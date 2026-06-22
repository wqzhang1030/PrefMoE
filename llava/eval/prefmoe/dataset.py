import pandas as pd
import os
import numpy as np
import string
import json
import hashlib
import os.path as osp
from pathlib import Path
from typing import List, Tuple
import torch
from PIL import Image
from llava.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token

stop_str = '</s>'

# 内置兜底 transcript：10-turn（20 条 utterances）中性对话，不涉及人物/偏好/识别任务。
_FALLBACK_GENERIC_CONVERSATION_10TURN = [
    {'role': 'user', 'content': 'Hi there. Could we do a quick casual chat?'},
    {'role': 'assistant', 'content': 'Sure. I can chat briefly on general topics.'},
    {'role': 'user', 'content': 'What is a simple way to start a focused morning?'},
    {'role': 'assistant', 'content': 'Pick one small task, finish it first, then plan the next steps.'},
    {'role': 'user', 'content': 'How can I keep notes tidy during the day?'},
    {'role': 'assistant', 'content': 'Use short bullet points with clear timestamps and one topic per line.'},
    {'role': 'user', 'content': 'Any tip for reducing distractions while reading?'},
    {'role': 'assistant', 'content': 'Set a short timer, silence alerts, and summarize one idea after each section.'},
    {'role': 'user', 'content': 'What is a practical break routine for long work blocks?'},
    {'role': 'assistant', 'content': 'Stand up, stretch briefly, drink water, then resume with one clear next action.'},
    {'role': 'user', 'content': 'How should I phrase a polite follow-up message?'},
    {'role': 'assistant', 'content': 'Keep it short, mention context, ask one clear question, and thank them.'},
    {'role': 'user', 'content': 'What helps with checking details before submission?'},
    {'role': 'assistant', 'content': 'Use a small checklist: names, numbers, dates, and formatting consistency.'},
    {'role': 'user', 'content': 'Can you suggest a quick end-of-day wrap-up?'},
    {'role': 'assistant', 'content': 'List completed items, unresolved items, and the first task for tomorrow.'},
    {'role': 'user', 'content': 'How do I keep communication clear in team chats?'},
    {'role': 'assistant', 'content': 'State objective first, provide context second, and end with an explicit request.'},
    {'role': 'user', 'content': 'Thanks. One final reminder for staying organized?'},
    {'role': 'assistant', 'content': 'Review priorities once daily and keep current tasks visible in one place.'},
]

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

_GENERIC_CONV_CANDIDATE_PATHS = (
    str(_PROJECT_ROOT / 'data' / 'multi_turn' / 'generic_text' / 'multi_turn_conversation.json'),
    osp.join(osp.dirname(__file__), 'eval_demo', 'generic_multi_turn_conversation.json'),
    osp.join(osp.dirname(__file__), 'generic_multi_turn_conversation.json'),
)

_LEGACY_BINARY_L2_CATEGORIES = {'awareness', 'overconcept'}
_LEGACY_MCQ_L2_CATEGORIES = {'inconsistency'}

def _identity_token(data_point, args):
    mode = str(getattr(args, 'identity_mode', 'sks'))
    if mode == 'name':
        return str(data_point.get('name_for_prompt', data_point.get('name', '<sks>')))
    if mode == 'id':
        try:
            return f"ID_{int(data_point.get('index')):05d}"
        except Exception:
            return 'ID_UNKNOWN'
    return '<sks>'


def _replace_sks(text, token):
    if not isinstance(text, str):
        return text
    return text.replace('<sks>', token)


_IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.ppm')


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

    # 常规路径：相对于 image_folder。
    c1 = osp.normpath(osp.join(image_folder, p))
    if osp.exists(c1):
        return c1

    # Fallback relative to the parent of image_folder for MMPB clean layouts.
    c2 = osp.normpath(osp.join(osp.dirname(image_folder), p))
    if osp.exists(c2):
        return c2
    return c1


def _content_to_prompt_text_and_images(content) -> Tuple[str, List[str]]:
    """把 message content 转成 prompt 文本，并提取其中的图片路径（按顺序）。"""
    if isinstance(content, list):
        parts: List[str] = []
        image_paths: List[str] = []
        for item in content:
            if _is_image_like_path(item):
                parts.append(DEFAULT_IMAGE_TOKEN)
                image_paths.append(str(item).strip())
            elif isinstance(item, str):
                parts.append(item)
        text = " ".join(x for x in parts if x).strip()
        return text, image_paths
    return str(content), []


def _content_to_prompt_text(content):
    # 保留原函数名供其他调用方复用。
    text, _ = _content_to_prompt_text_and_images(content)
    return text


def _render_prompt_from_msgs(msgs, stop_str: str):
    prompt = ""
    prompt_image_paths: List[str] = []
    for utter in msgs:
        prompt += "USER: " if utter["role"] == "user" else "ASSISTANT: "
        text_part, img_paths = _content_to_prompt_text_and_images(utter["content"])
        prompt += text_part
        prompt_image_paths.extend(img_paths)
        prompt += " " if utter["role"] == "user" else stop_str
    return prompt, prompt_image_paths


def _build_query_span_mask(
    *,
    input_ids: torch.Tensor,
    tokenizer,
    prompt_before_query: str,
    prompt_through_query: str,
) -> torch.Tensor:
    query_start = int(tokenizer_image_token(
        prompt_before_query,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors='pt',
    ).shape[0])
    query_end = int(tokenizer_image_token(
        prompt_through_query,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors='pt',
    ).shape[0])
    seq_len = int(input_ids.shape[0])
    query_start = max(0, min(query_start, seq_len))
    query_end = max(query_start, min(query_end, seq_len))
    query_span_mask = torch.zeros_like(input_ids, dtype=torch.long)
    query_span_mask[query_start:query_end] = 1
    return query_span_mask


def _collect_csv_injection_images(data_point):
    images = []
    keys = sorted([k for k in data_point.index if str(k).startswith("injection_image_")])
    for key in keys:
        val = data_point[key]
        if pd.isna(val):
            continue
        sval = str(val).strip()
        if sval:
            images.append(sval)
    return images


def _build_final_query(data_point, question, identity_tok):
    l2_category = str(data_point.get('l2-category', '')).strip()

    # Preserve the original task routing and add a safe fallback for newer
    # cleaned binary tasks whose l2-category no longer matches the legacy names.
    if l2_category in _LEGACY_BINARY_L2_CATEGORIES:
        return f'\n{question}\n'

    has_options = l2_category in _LEGACY_MCQ_L2_CATEGORIES
    if not has_options:
        has_options = any(ch in data_point and not pd.isna(data_point[ch]) for ch in string.ascii_uppercase)

    if has_options:
        options = {
            cand: _replace_sks(data_point[cand], identity_tok)
            for cand in string.ascii_uppercase
            if cand in data_point and not pd.isna(data_point[cand])
        }
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'

        final_query = f'\n{question}\n'
        if len(options):
            final_query += options_prompt
        return final_query

    return f'\n{question}\n'


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


def _is_generic_conversation_enabled(args):
    """判断是否启用 generic multi-turn。

    约束：
    - 只有 `--generic-conversation-enable` 为真且 `n_turn > 0` 才生效。
    - 默认关闭，保证 0-turn 行为与历史完全一致。
    """
    enabled = bool(getattr(args, 'generic_conversation_enable', False))
    n_turn = int(getattr(args, 'generic_conversation_n_turn', 0) or 0)
    return enabled and n_turn > 0


def _normalize_generic_records(records):
    """把多种 transcript 结构归一成 [{role, content}, ...]。"""
    out = []
    for i, item in enumerate(records):
        role = ''
        content = ''
        if isinstance(item, dict):
            role = str(item.get('role', '')).strip().lower()
            content = item.get('content', '')
        else:
            content = item
        if role not in {'user', 'assistant'}:
            role = 'user' if (i % 2 == 0) else 'assistant'
        if not isinstance(content, str):
            content = str(content)
        content = content.strip()
        if not content:
            continue
        out.append({'role': role, 'content': content})
    return out


def _flatten_generic_payload(payload):
    """解析 transcript 文件内容，兼容 dict/list 嵌套结构。"""
    if isinstance(payload, dict):
        if 'conversation' in payload:
            return _flatten_generic_payload(payload['conversation'])
        return _normalize_generic_records([payload])

    if isinstance(payload, list):
        if not payload:
            return []
        if all(isinstance(x, dict) and 'content' in x for x in payload):
            return _normalize_generic_records(payload)

        flat = []
        for part in payload:
            if isinstance(part, list):
                flat.extend(_normalize_generic_records(part))
            elif isinstance(part, dict):
                if 'conversation' in part and isinstance(part['conversation'], list):
                    flat.extend(_flatten_generic_payload(part['conversation']))
                elif 'content' in part:
                    flat.extend(_normalize_generic_records([part]))
        return flat

    return []


def _load_generic_transcript_records(args):
    """加载 generic transcript，优先外部路径，失败时回退到内置 10-turn。"""
    candidate_paths = []
    user_path = str(getattr(args, 'generic_conversation_path', '') or '').strip()
    if user_path:
        candidate_paths.append(user_path)
    candidate_paths.extend(_GENERIC_CONV_CANDIDATE_PATHS)

    for p in candidate_paths:
        try:
            if p and osp.exists(p):
                with open(p, 'r', encoding='utf-8') as f:
                    payload = json.load(f)
                records = _flatten_generic_payload(payload)
                if records:
                    return records
        except Exception:
            # 文件读取失败不影响主流程，继续尝试下一个候选路径。
            pass

    return list(_FALLBACK_GENERIC_CONVERSATION_10TURN)


def _build_generic_conversation_msgs(args):
    """构建要插入 prompt 的 generic 对话消息列表。"""
    if not _is_generic_conversation_enabled(args):
        return []

    n_turn = int(getattr(args, 'generic_conversation_n_turn', 0) or 0)
    seed = int(getattr(args, 'generic_conversation_seed', 0) or 0)
    required_utterances = max(0, n_turn * 2)
    if required_utterances <= 0:
        return []

    records = _load_generic_transcript_records(args)
    if not records:
        return []

    if len(records) < required_utterances:
        repeat = (required_utterances + len(records) - 1) // len(records)
        picked = (records * repeat)[:required_utterances]
    elif len(records) > required_utterances:
        max_start = len(records) - required_utterances
        start = 0 if max_start <= 0 else (abs(seed) % (max_start + 1))
        picked = records[start:start + required_utterances]
    else:
        picked = records

    # 保证 user/assistant 交替。若源数据 role 异常，按位置兜底修正。
    msgs = []
    for i, rec in enumerate(picked):
        role = str(rec.get('role', '')).strip().lower()
        if role not in {'user', 'assistant'}:
            role = 'user' if (i % 2 == 0) else 'assistant'
        msgs.append(
            {
                'type': 'text',
                'role': role,
                'content': str(rec.get('content', '')).strip(),
            }
        )

    if len(msgs) % 2 == 1:
        msgs = msgs[:-1]
    return msgs


def _sha1_text(x: str) -> str:
    return hashlib.sha1(str(x).encode('utf-8')).hexdigest()


def _sha1_msgs(msgs) -> str:
    try:
        payload = json.dumps(msgs, ensure_ascii=False, sort_keys=True)
    except Exception:
        payload = str(msgs)
    return hashlib.sha1(payload.encode('utf-8')).hexdigest()


def build_prompt(data_point, tokenizer, image_processor, args):
    idx         = data_point['index']
    identity_tok = _identity_token(data_point, args)
    question    = _replace_sks(data_point['question'], identity_tok)
    answer      = data_point['answer']
    img_path    = data_point['image_path']
    preference  = _replace_sks(data_point['preference'], identity_tok)
    category    = data_point['category']
    l2_category = data_point['l2-category']
    name        = data_point['name']
    att         = data_point['attribute']
    answer      = project_labels_to_predictions(answer)
    if args.injection_description_prompt_type=="hard_moderate":
        description = _replace_sks(data_point['description_moderate'], identity_tok)
        des_type = 'description_moderate'
    elif args.injection_description_prompt_type=="hard_detailed":
        description = _replace_sks(data_point['description_detailed'], identity_tok)
        des_type = 'description_detailed'
    elif args.injection_description_prompt_type=="hard_simple":
        description = _replace_sks(data_point['description_simple'], identity_tok)
        des_type = 'description_simple'
    elif args.injection_description_prompt_type=="hard_super_detailed":
        description = _replace_sks(data_point['description_super_detailed'], identity_tok)
    elif args.injection_description_prompt_type=="image":
        description = ""
    else:
        raise NotImplementedError

    final_query = _build_final_query(data_point, question, identity_tok)

    #########################################################################
    drop_profile = bool(getattr(args, 'drop_profile_in_test', False) or getattr(args, 'use_name_memory', False))
    if drop_profile:
        description = ""
        preference = ""

    msgs = []
    if not drop_profile:
        compose_mode = str(getattr(args, "injection_compose_mode", "legacy"))
        if compose_mode == "image_text_des_pre":
            csv_images = _collect_csv_injection_images(data_point)
            profile_prompt = (description + " " + preference).strip()
            if csv_images:
                msgs.append({'type': 'image', 'role': 'user', 'content': csv_images + [profile_prompt]})
            else:
                msgs.append({'type': 'text', 'role': 'user', 'content': profile_prompt})
            msgs.append({'type': 'text', 'role': 'assistant', 'content': 'I got it.'})
        elif args.injection_description_prompt_type == "image":
            inj_img_path = [
                osp.join("../mmpb_clean", data_point['attribute'], "train", data_point['name'], f"{i}.png")
                for i in range(args.injection_prompt_n_images)
            ]
            msgs.append(
                {
                    'type': 'image',
                    'role': 'user',
                    'content': [img_path for img_path in inj_img_path] + [f"These images present {identity_tok}. {preference}"],
                }
            )
            msgs.append({'type': 'text', 'role': 'assistant', 'content': 'I got it.'})
        elif args.injection_preference_prompt_type == "image":
            pref_img_path = [
                osp.join("../MMPB", data_point['attribute'], "test", data_point['name'], "prefernece", "image", f"{i}.png")
                for i in range(4)
            ]
            msgs.append(
                {
                    'type': 'image',
                    'role': 'user',
                    'content': [
                        img_path for img_path in pref_img_path
                    ]
                    + [f"{description} The first two images show {identity_tok}'s entertainment preferences, and the last two his/her dislikes."],
                }
            )
            msgs.append({'type': 'text', 'role': 'assistant', 'content': 'I got it.'})
        else:
            profile_prompt = (description + " " + preference).strip()
            msgs.append({'type': 'text', 'role': 'user', 'content': profile_prompt})
            msgs.append({'type': 'text', 'role': 'assistant', 'content': 'I got it.'})

    # Multi-turn (可选，默认关闭)：
    # - 这里是新增分支；turns==0 时不进入，0-turn 路径保持原样。
    # - 插入位置固定在 injection 与 final query 之间。
    generic_conversation_msgs = []
    if _is_generic_conversation_enabled(args):
        generic_conversation_msgs = _build_generic_conversation_msgs(args)
        if generic_conversation_msgs:
            msgs.extend(generic_conversation_msgs)

    # Eval query
    msgs.append({'type': 'text', 'role': 'user', 'content': "<image>\n" + final_query})
    ########################################################################
    prompt_before_query, _ = _render_prompt_from_msgs(msgs[:-1], stop_str)
    prompt, prompt_image_paths = _render_prompt_from_msgs(msgs, stop_str)
    # assert msgs[-1]["role"] 
    prompt += "ASSISTANT: "
    full_train_text = prompt + " " + str(answer) + stop_str
    prompt_through_query = prompt[:-len("ASSISTANT: ")]
    #########################################################################
    image_folder = args.image_folder
    processor = image_processor

    # query 图始终存在，且与最终 "<image>" 一一对应。
    query_abs = _resolve_image_path(image_folder, img_path)
    query_image = _preprocess_one_image(query_abs, processor, args.image_aspect_ratio)

    # prompt 中 profile 注入产生的 image token 在前；query image 在最后。
    prompt_image_paths.append(img_path)
    image_list = []
    for p in prompt_image_paths:
        abs_p = _resolve_image_path(image_folder, p)
        try:
            if osp.normpath(abs_p) == osp.normpath(query_abs):
                image_list.append(query_image.clone())
            else:
                image_list.append(_preprocess_one_image(abs_p, processor, args.image_aspect_ratio))
        except Exception:
            # profile 路径坏掉时，用 query 图兜底，保证 image token 数和张量数一致。
            image_list.append(query_image.clone())

    if len(image_list) == 1:
        image = image_list[0]
    else:
        image = torch.stack(image_list, dim=0)
    
    input_ids = tokenizer_image_token(
        full_train_text, 
        tokenizer, 
        IMAGE_TOKEN_INDEX, 
        return_tensors='pt'
    )
    prompt_ids = tokenizer_image_token(
        prompt + " ", # 加个空格通常为了对齐分词边界
        tokenizer, 
        IMAGE_TOKEN_INDEX, 
        return_tensors='pt'
    )
    prompt_len = prompt_ids.shape[0]
    query_span_mask = _build_query_span_mask(
        input_ids=input_ids,
        tokenizer=tokenizer,
        prompt_before_query=prompt_before_query,
        prompt_through_query=prompt_through_query,
    )
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
        image=image,
        query_span_mask=query_span_mask,
    )

    # dryrun 调试信息：
    # - 仅在显式开启时写入，不影响默认 0-turn 与常规运行输出。
    if int(getattr(args, 'generic_conversation_dryrun', 0) or 0) > 0:
        data_dict['_debug_msgs'] = msgs
        data_dict['_debug_prompt_text'] = prompt
        data_dict['_debug_prompt_sha1'] = _sha1_text(prompt)
        data_dict['_debug_msgs_sha1'] = _sha1_msgs(msgs)
        data_dict['_debug_total_messages'] = int(len(msgs))
        data_dict['_debug_generic_utterances'] = int(len(generic_conversation_msgs))
        data_dict['_debug_generic_turns'] = int(len(generic_conversation_msgs) // 2)

    return data_dict



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
