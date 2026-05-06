from torch.utils.data import Dataset
import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import json
import random
import torch
import os
import os.path as osp
import validators
import pandas as pd
import pickle
import copy
import numpy as np
import string
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava import conversation as conversation_lib
from llava.model.memory import build_fold_train_visible_user_registry, resolve_description_field, concept_to_factor_id

from llava.mm_utils import tokenizer_image_token

local_rank = None

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    data_split_path: str = field(default=None,
                           metadata={"help": "Path to the training data split."})
    memory_data_path: str = field(default=None,
                           metadata={"help": "Path to the memory data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    multi_turn: int = 1
    injection_prompt_n_images: int = 4
    injection_description_prompt_type: str = "hard_moderate"
    injection_preference_prompt_type: str = "explicit"
    identity_mode: str = "sks"
    injection_compose_mode: str = "legacy"
    train_fold_start: int = 0
    train_fold_end: int = -1
    train_task_start: int = 0
    train_task_end: int = -1
    name_memory_enable: bool = False
    name_memory_use_concept_id: bool = False
    


def _identity_token(line, mode: str):
    if mode == "name":
        return str(line.get("name_for_prompt", line.get("name", "<sks>")))
    if mode == "id":
        try:
            return f"ID_{int(line.get('index')):05d}"
        except Exception:
            return "ID_UNKNOWN"
    return "<sks>"


def _replace_sks(text, token: str):
    if not isinstance(text, str):
        return text
    return text.replace("<sks>", token)


_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".ppm")
_LEGACY_BINARY_L2_CATEGORIES = {"awareness", "overconcept"}
_LEGACY_MCQ_L2_CATEGORIES = {"inconsistency"}


def _is_image_like_path(val) -> bool:
    if not isinstance(val, str):
        return False
    return val.strip().lower().endswith(_IMAGE_EXTS)


def _resolve_image_path(image_folder: str, rel_or_abs: str) -> str:
    p = str(rel_or_abs).strip()
    if not p:
        return p
    if osp.isabs(p):
        return p

    c1 = osp.normpath(osp.join(image_folder, p))
    if osp.exists(c1):
        return c1

    c2 = osp.normpath(osp.join(osp.dirname(image_folder), p))
    if osp.exists(c2):
        return c2
    return c1


def _content_to_prompt_text_and_images(content):
    if isinstance(content, list):
        parts = []
        image_paths = []
        for item in content:
            if _is_image_like_path(item):
                parts.append(DEFAULT_IMAGE_TOKEN)
                image_paths.append(str(item).strip())
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(x for x in parts if x).strip(), image_paths
    return str(content), []


def _content_to_prompt_text(content):
    # 兼容旧调用方：仍只返回 prompt 文本。
    text, _ = _content_to_prompt_text_and_images(content)
    return text


def _preprocess_one_image(image_path: str, processor, image_aspect_ratio: str):
    image = Image.open(image_path).convert('RGB')
    if image_aspect_ratio == 'pad':
        def expand2square(pil_img, background_color):
            width, height = pil_img.size
            if width == height:
                return pil_img
            elif width > height:
                result = Image.new(pil_img.mode, (width, width), background_color)
                result.paste(pil_img, (0, (width - height) // 2))
                return result
            else:
                result = Image.new(pil_img.mode, (height, height), background_color)
                result.paste(pil_img, ((height - width) // 2, 0))
                return result
        image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
    return processor.preprocess(image, return_tensors='pt')['pixel_values'][0]


def _collect_csv_injection_images(line):
    images = []
    for key in sorted([k for k in line.index if str(k).startswith("injection_image_")]):
        val = line[key]
        if pd.isna(val):
            continue
        sval = str(val).strip()
        if sval:
            images.append(sval)
    return images


def _build_final_query(line, question: str, identity_tok: str) -> str:
    l2_category = str(line.get("l2-category", "")).strip()

    # Preserve the original task routing and add a safe fallback for newer
    # cleaned binary tasks whose l2-category no longer matches the legacy names.
    if l2_category in _LEGACY_BINARY_L2_CATEGORIES:
        return f"\n{question}\n"

    has_options = l2_category in _LEGACY_MCQ_L2_CATEGORIES
    if not has_options:
        has_options = any(ch in line and not pd.isna(line[ch]) for ch in string.ascii_uppercase)

    if has_options:
        options = {
            cand: _replace_sks(line[cand], identity_tok)
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        options_prompt = "Options:\n"
        for key, item in options.items():
            options_prompt += f"{key}. {item}\n"

        final_query = f"\n{question}\n"
        if len(options):
            final_query += options_prompt
        return final_query

    return f"\n{question}\n"


def _limit_injection_images(images, max_images: int):
    """按训练参数裁剪 profile 注入图像数；默认保持原行为不裁剪。"""
    try:
        max_images = int(max_images)
    except Exception:
        return images
    if max_images < 0:
        return images
    return images[:max_images]


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        if data_args.memory_data_path is not None:
            list_memory_data_dict = json.load(open(data_args.memory_data_path, "r"))

            list_data_dict = list_data_dict + list_memory_data_dict
            
            random.shuffle(list_data_dict)

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict
    


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)



def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer_image_token(rou, tokenizer)) + len(tokenizer_image_token(conv.sep, tokenizer))
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)



def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation

################################################################################


class LazySupervisedConceptINCRDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, data_split_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedConceptINCRDataset, self).__init__()
        # list_data_dict = json.load(open(data_path, "r"))
        if data_args.multi_turn > 1:
            # for multi-turn conversation
            raise NotImplementedError
        else:
            # For single turn answering
            data_split = load(data_split_path)
            data_dframe = load(data_path)
        # if data_args.memory_data_path is not None:
        #     list_memory_data_dict = json.load(open(data_args.memory_data_path, "r"))

        #     list_data_dict = list_data_dict + list_memory_data_dict
            
        #     random.shuffle(list_data_dict)

        rank0_print("Formatting inputs...Skip in lazy mode")        
        self.stop_str = "</s>"

        self.tokenizer = tokenizer
        self.data_split = data_split
        self.all_data = data_dframe
        self.data_args = data_args
        self.task_idx = 0
        self.fold_idx = 0
        self._name_memory_registry_by_fold = {}

    def __len__(self):
        return len(self.data_split[self.fold_idx]["tasks"][self.task_idx]['train_idx'])

    def get_name_memory_registry(self, fold_idx: Optional[int] = None):
        fold_idx = self.fold_idx if fold_idx is None else int(fold_idx)
        if fold_idx not in self._name_memory_registry_by_fold:
            description_field = resolve_description_field(self.data_args.injection_description_prompt_type)
            self._name_memory_registry_by_fold[fold_idx] = build_fold_train_visible_user_registry(
                data_frame=self.all_data,
                data_split=self.data_split,
                fold_idx=fold_idx,
                image_folder=self.data_args.image_folder,
                description_field=description_field,
            )
        return self._name_memory_registry_by_fold[fold_idx]

    # @property
    # def lengths(self):
    #     """
    #     Docstring for lengths
    #     Compute the token lengths of the conversations in the dataset, considering both text and image tokens.
    #     :param self: Description
    #     """
    #     length_list = []
    #     if self.data_args.multi_turn > 1:
    #         raise NotImplementedError
    #     else:
    #         for sample in self.list_data_pdframe.itertuples():
    #             img_tokens = 128 if 'image' in sample else 0
    #             length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
    #     # for sample in self.list_data_dict:
    #     #     img_tokens = 128 if 'image' in sample else 0
    #     #     length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
    #     return length_list

    # @property
    # def modality_lengths(self):
    #     length_list = []
    #     # for sample in self.list_data_dict:
    #     for sample in self.list_data_pdframe.itertuples():
    #         cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
    #         cur_len = cur_len if 'image' in sample else -cur_len
    #         length_list.append(cur_len)
    #     return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        data_idx    = self.data_split[self.fold_idx]["tasks"][self.task_idx]['train_idx'][i]
        line        = self.all_data.iloc[data_idx]
    
        idx         = line['index']
        identity_tok = _identity_token(line, str(getattr(self.data_args, "identity_mode", "sks")))
        question    = _replace_sks(line['question'], identity_tok)
        answer      = line['answer']
        img_path    = line['image_path']
        preference  = _replace_sks(line['preference'], identity_tok)
        category    = line['category']
        l2_category = line['l2-category']
        name        = line['name']
        att         = line['attribute']
        answer      = project_labels_to_predictions(answer)
        if self.data_args.injection_description_prompt_type=="hard_moderate":
            description = _replace_sks(line['description_moderate'], identity_tok)
            des_type = 'description_moderate'
        elif self.data_args.injection_description_prompt_type=="hard_detailed":
            description = _replace_sks(line['description_detailed'], identity_tok)
            des_type = 'description_detailed'
        elif self.data_args.injection_description_prompt_type=="hard_simple":
            description = _replace_sks(line['description_simple'], identity_tok)
            des_type = 'description_simple'
        elif self.data_args.injection_description_prompt_type=="hard_super_detailed":
            description = _replace_sks(line['description_super_detailed'], identity_tok)
        elif self.data_args.injection_description_prompt_type=="image":
            description = ""
        else:
            raise NotImplementedError

        final_query = _build_final_query(line, question, identity_tok)

        name_memory_enable = bool(getattr(self.data_args, "name_memory_enable", False))
        registry = self.get_name_memory_registry(self.fold_idx) if name_memory_enable else None
        visible_name = str(line.get("name_for_prompt", line.get("name", ""))).strip()
        user_slot_id = int(registry["slot_by_name"].get(visible_name, 0)) if registry is not None else 0
        use_concept_id = bool(getattr(self.data_args, "name_memory_use_concept_id", False))
        concept_id = concept_to_factor_id(line.get("concept", "")) if (name_memory_enable and use_concept_id) else None
        router_task_label = None
        if name_memory_enable:
            category_norm = str(category).strip().lower()
            if category_norm == "preference":
                router_task_label = 0
            elif category_norm == "recognition":
                router_task_label = 1

        #########################################################################
        msgs = []
        prompt = ""
        compose_mode = str(getattr(self.data_args, "injection_compose_mode", "legacy"))

        if name_memory_enable:
            pass
        elif compose_mode == "image_text_des_pre":
            # 新模式：按 CSV 的 injection_image_* 字段组织 profile image 注入。
            csv_images = _limit_injection_images(
                _collect_csv_injection_images(line),
                getattr(self.data_args, "injection_prompt_n_images", 4),
            )
            profile_prompt = (description + " " + preference).strip()
            if csv_images:
                msgs.append({'type': 'image', 'role': 'user', 'content': csv_images + [profile_prompt]})
            else:
                msgs.append({'type': 'text', 'role': 'user', 'content': profile_prompt})
            msgs.append({'type': 'text', 'role': 'assistant', 'content': 'I got it.'})
        elif self.data_args.injection_description_prompt_type == "image":
            csv_images = _limit_injection_images(
                _collect_csv_injection_images(line),
                getattr(self.data_args, "injection_prompt_n_images", 4),
            )
            if not csv_images:
                csv_images = [
                    osp.join("../mmpb_clean", line['attribute'], "train", line['name'], f"{j}.png")
                    for j in range(int(getattr(self.data_args, "injection_prompt_n_images", 4)))
                ]
            msgs.append(
                {
                    'type': 'image',
                    'role': 'user',
                    'content': [img for img in csv_images] + [f"These images present {identity_tok}. {preference}"],
                }
            )
            msgs.append({'type': 'text', 'role': 'assistant', 'content': 'I got it.'})
        elif self.data_args.injection_preference_prompt_type == "image":
            pref_img_path = [osp.join("../MMPB", line['attribute'], "test", line['name'], "prefernece", "image", f"{j}.png") for j in range(4)]
            msgs.append(
                {
                    'type': 'image',
                    'role': 'user',
                    'content': [img_path for img_path in pref_img_path]
                    + [f"{description} The first two images show {identity_tok}'s entertainment preferences, and the last two his/her dislikes."],
                }
            )
            msgs.append({'type': 'text', 'role': 'assistant', 'content': 'I got it.'})
        else:
            profile_prompt = (description + " " + preference).strip()
            msgs.append({'type': 'text', 'role': 'user', 'content': profile_prompt})
            msgs.append({'type': 'text', 'role': 'assistant', 'content': 'I got it.'})

        # Multi-turn
        # for i in range(0, len(self.mt_data), 2) :
        #     msgs.append({'role': 'user', 'content': self.mt_data[i]["content"]}) 
        #     msgs.append({'role': 'assistant', 'content': self.mt_data[i+1]["content"]})

        # Eval query
        msgs.append({'type': 'text', 'role': 'user', 'content': "<image>\n" + final_query})
        ########################################################################
        prompt_image_paths = []
        for utter in msgs:
            prompt += "USER: " if utter["role"] == "user" else "ASSISTANT: "
            text_part, img_paths = _content_to_prompt_text_and_images(utter["content"])
            prompt += text_part
            prompt_image_paths.extend(img_paths)
            prompt += " " if utter["role"] == "user" else self.stop_str
        # assert msgs[-1]["role"] 
        prompt += "ASSISTANT: "
        full_train_text = prompt + " " + str(answer) + self.stop_str
        #########################################################################
        image_folder = self.data_args.image_folder
        processor = self.data_args.image_processor

        # query 图对应 final query 中的 <image>。
        query_abs = _resolve_image_path(image_folder, img_path)
        query_image = _preprocess_one_image(query_abs, processor, self.data_args.image_aspect_ratio)

        # profile image token 在前，query image token 在后。
        prompt_image_paths.append(img_path)
        image_list = []
        for p in prompt_image_paths:
            abs_p = _resolve_image_path(image_folder, p)
            try:
                if osp.normpath(abs_p) == osp.normpath(query_abs):
                    image_list.append(query_image.clone())
                else:
                    image_list.append(_preprocess_one_image(abs_p, processor, self.data_args.image_aspect_ratio))
            except Exception:
                image_list.append(query_image.clone())

        if len(image_list) == 1:
            image = image_list[0]
        else:
            image = torch.stack(image_list, dim=0)
        
        input_ids = tokenizer_image_token(
            full_train_text, 
            self.tokenizer, 
            IMAGE_TOKEN_INDEX, 
            return_tensors='pt'
        )
        prompt_ids = tokenizer_image_token(
            prompt + " ", # 加个空格通常为了对齐分词边界
            self.tokenizer, 
            IMAGE_TOKEN_INDEX, 
            return_tensors='pt'
        )
        prompt_len = prompt_ids.shape[0]
        labels = input_ids.clone()
        labels[:prompt_len] = IGNORE_INDEX
        #################################################################
        # if 'image' in sources[0]:
        #     image_file = self.list_data_dict[i]['image']
        #     image_folder = self.data_args.image_folder
        #     processor = self.data_args.image_processor
        #     image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
        #     if self.data_args.image_aspect_ratio == 'pad':
        #         def expand2square(pil_img, background_color):
        #             width, height = pil_img.size
        #             if width == height:
        #                 return pil_img
        #             elif width > height:
        #                 result = Image.new(pil_img.mode, (width, width), background_color)
        #                 result.paste(pil_img, (0, (width - height) // 2))
        #                 return result
        #             else:
        #                 result = Image.new(pil_img.mode, (height, height), background_color)
        #                 result.paste(pil_img, ((height - width) // 2, 0))
        #                 return result
        #         image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
        #         image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        #     else:
        #         image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        #     sources = preprocess_multimodal(
        #         copy.deepcopy([e["conversations"] for e in sources]),
        #         self.data_args)
        # else:
        #     sources = copy.deepcopy([e["conversations"] for e in sources])
        # data_dict = preprocess(
        #     sources,
        #     self.tokenizer,
        #     has_image=('image' in self.list_data_dict[i]))
        # if isinstance(i, int):
        #     data_dict = dict(input_ids=data_dict["input_ids"][0],
        #                      labels=data_dict["labels"][0])

        # # image exist in the data
        # if 'image' in self.list_data_dict[i]:
        #     data_dict['image'] = image
        # elif self.data_args.is_multimodal:
        #     # image does not exist in the data, but the model is multimodal
        #     crop_size = self.data_args.image_processor.crop_size
        #     data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        data_dict = dict(
            input_ids=input_ids,
            labels=labels,
            image=image
        )
        if name_memory_enable:
            data_dict["user_slot_id"] = torch.tensor(user_slot_id, dtype=torch.long)
            if concept_id is not None:
                data_dict["concept_id"] = torch.tensor(concept_id, dtype=torch.long)
            if router_task_label is not None:
                data_dict["router_task_label"] = torch.tensor(router_task_label, dtype=torch.long)
        return data_dict
    
    def concat_tilist(self, message):
        text, images = "", []
        for item in message:
            if item["type"] == "text":
                text += item["value"]
            elif item["type"] == "image":
                # text += " <image> "
                images.append(item["value"])
        return text, images

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        if 'user_slot_id' in instances[0]:
            batch['user_slot_id'] = torch.stack([instance['user_slot_id'] for instance in instances])
        if 'concept_id' in instances[0]:
            batch['concept_id'] = torch.stack([instance['concept_id'] for instance in instances])
        if any('router_task_label' in instance for instance in instances):
            router_task_labels = [
                instance.get('router_task_label', torch.tensor(-1, dtype=torch.long))
                for instance in instances
            ]
            batch['router_task_label'] = torch.stack(router_task_labels)

        return batch


def make_supervised_cvlmp_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedConceptINCRDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        data_split_path=data_args.data_split_path,
        data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def load(f, fmt=None):
    def load_pkl(pth):
        return pickle.load(open(pth, 'rb'))

    def load_json(pth):
        return json.load(open(pth, 'r', encoding='utf-8'))

    def load_jsonl(f):
        lines = open(f, encoding='utf-8').readlines()
        lines = [x.strip() for x in lines]
        if lines[-1] == '':
            lines = lines[:-1]
        data = [json.loads(x) for x in lines]
        return data

    def load_xlsx(f):
        return pd.read_excel(f)

    def load_csv(f):
        return pd.read_csv(f)

    def load_tsv(f):
        return pd.read_csv(f, sep='\t')
    
    def LMUDataRoot():
        if 'LMUData' in os.environ and osp.exists(os.environ['LMUData']):
            return os.environ['LMUData']
        home = osp.expanduser('~')
        root = osp.join(home, 'LMUData')
        os.makedirs(root, exist_ok=True)
        return root

    import validators
    if validators.url(f):
        tgt = osp.join(LMUDataRoot(), 'files', osp.basename(f))
        if not osp.exists(tgt):
            # download_file(f, tgt)
            raise Exception("No dataset found.")
        f = tgt

    handlers = dict(pkl=load_pkl, json=load_json, jsonl=load_jsonl, xlsx=load_xlsx, csv=load_csv, tsv=load_tsv)
    if fmt is not None:
        return handlers[fmt](f)

    suffix = f.split('.')[-1]
    return handlers[suffix](f)


def project_predictions_to_labels(pred):
    if not isinstance(pred, str):
        return 99
    pred = pred.strip().lower()
    return {
        "a": 0,
        "b": 1,
        "c": 2,
        "d": 3,
        "yes": 4,
        "no": 5,
    }.get(pred, 99)

def project_labels_to_predictions(label):
    try:
        label = int(float(label))
    except Exception:
        return 'fail'
    return {
        0: 'a',
        1: 'b',
        2: "c",
        3: "d",
        4: "yes",
        5: "no",
    }.get(label, 'fail')
