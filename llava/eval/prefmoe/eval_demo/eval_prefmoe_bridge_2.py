import argparse
import glob
import hashlib
import logging
import os
import re
import sys
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig


# ==============================
# 模块：路径与依赖初始化
# ==============================
# 职责：
# 1) 计算当前脚本所在目录，并把项目根目录加入 sys.path。
# 2) 统一从项目包中导入评测所需函数与模型构建工具。
# 3) 该模块只做“运行环境准备”，不处理任何业务数据。
# 4) 如果路径插入失败，后续 import 会直接报错并中断。
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CUR_DIR, '../../../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from llava.constants import IGNORE_INDEX 
from llava.eval.prefmoe.dataset import build_prompt
from llava.eval.prefmoe.eval_demo.metrics import evaluate_mmpb_no_gpt, report_acc
from llava.eval.prefmoe.utils import dump, load
from llava.mm_utils import get_model_name_from_path_v2
from llava.model.builder import load_pretrained_model_v2
from llava.model.memory import build_fold_train_visible_user_registry, resolve_description_field, concept_to_factor_id
from llava.utils import disable_torch_init


# ==============================
# 模块：运行时工具函数
# ==============================
# 职责：
# 1) 提供模型 task 推断、输出目录创建、raw 可读化等通用函数。
# 2) 这些函数不持有模型状态，输入输出都是纯数据结构。
# 3) 该模块被 test/evaluate 两条主流程共同复用。
# 4) 出错通常表现为路径不存在、字段缺失、类型转换失败。
def infer_model_task(model_path: str, fallback_task: int) -> int:
    """从 checkpoint 路径中推断 model_task。

    解析规则：
    - 优先匹配路径里的 `task_<数字>`，例如 `.../fold_0_task_7`。
    - 若未匹配成功，回退到调用方传入的 `fallback_task`。

    参数：
    - model_path: 模型目录路径。
    - fallback_task: 兜底 task id。

    返回：
    - int，模型任务编号。
    """
    match = re.search(r'task_(\d+)', str(model_path))
    return int(match.group(1)) if match else int(fallback_task)


def ensure_pair_dirs(output_root: str, fold: int, model_task: int, eval_task: int) -> Tuple[str, str, str, str]:
    """创建单个 (model_task, eval_task) 评测对所需目录。

    目录结构：
    - pairs/fold_x/mt_xx_et_xx : 每个 pair 的 pkl 预测文件。
    - raw/fold_x               : 每个 pair 的原始样本输出（含 prediction）。
    - scores/fold_x            : 每个 pair 的聚合分数。

    参数：
    - output_root: 当前 run 的输出根目录。
    - fold: fold 编号。
    - model_task: 模型任务编号（训练到哪一任务）。
    - eval_task: 被测试的任务编号。

    返回：
    - pair_tag: 例如 `mt_03_et_01`。
    - pair_root/raw_root/score_root: 三类产物目录绝对路径。
    """
    pair_tag = f'mt_{model_task:02d}_et_{eval_task:02d}'
    pair_root = os.path.join(output_root, 'pairs', f'fold_{fold}', pair_tag)
    raw_root = os.path.join(output_root, 'raw', f'fold_{fold}')
    score_root = os.path.join(output_root, 'scores', f'fold_{fold}')
    os.makedirs(pair_root, exist_ok=True)
    os.makedirs(raw_root, exist_ok=True)
    os.makedirs(score_root, exist_ok=True)
    return pair_tag, pair_root, raw_root, score_root


def _append_suffix_once(root: str, suffix_tag: str) -> str:
    """给输出目录追加一次后缀，避免重复拼接同名标签。"""
    root = str(root)
    tag = str(suffix_tag or '').strip()
    if not tag:
        return root
    marker = f'__{tag}'
    return root if marker in root else f'{root}{marker}'


def _apply_generic_conversation_output_suffix(args):
    """当 multi-turn 开启时，为输出目录追加独立后缀。

    约束：
    - 默认关闭时不改 output_root，保证历史 baseline 输出路径不变。
    - 开启后统一追加 `__gconv_{n}turn_s{seed}`，避免覆盖 0-turn 结果。
    """
    enabled = bool(getattr(args, 'generic_conversation_enable', False))
    n_turn = int(getattr(args, 'generic_conversation_n_turn', 0) or 0)
    seed = int(getattr(args, 'generic_conversation_seed', 0) or 0)
    if enabled and n_turn > 0:
        args.output_root = _append_suffix_once(args.output_root, f'gconv_{n_turn}turn_s{seed}')


def augment_output_root(base_root: str, suffix_parts: List[str]) -> str:
    """批量给输出目录追加后缀标签（仅追加一次）。"""
    root = str(base_root)
    for tag in suffix_parts:
        if not tag:
            continue
        root = _append_suffix_once(root, str(tag))
    return root


def norm_group_token(x: str) -> str:
    """规范化 group 名称，统一用于 scope 匹配。"""
    return str(x).strip().lower().replace(' ', '_').replace('-', '_')


def sample_groups(row: pd.Series) -> Set[str]:
    """提取样本可命中的 group 集合。"""
    cat = norm_group_token(row.get('category', ''))
    l2 = norm_group_token(row.get('l2-category', ''))
    groups: Set[str] = set()
    if cat:
        groups.add(cat)
    if l2:
        groups.add(l2)
    if cat and l2:
        groups.add(f'{l2}__{cat}')
    return groups


def primary_group(row: pd.Series) -> str:
    """返回日志展示用主 group，优先 l2__category。"""
    cat = norm_group_token(row.get('category', ''))
    l2 = norm_group_token(row.get('l2-category', ''))
    if cat and l2:
        return f'{l2}__{cat}'
    return l2 or cat or 'unknown'


def parse_scope(scope: str) -> Optional[Set[str]]:
    """解析 ablation scope 文本；返回 None 表示 all。"""
    raw = str(scope or 'all').strip()
    if not raw or raw.lower() == 'all':
        return None

    raw_low = raw.lower()
    if raw_low.startswith('only_groups'):
        raw = raw[len('only_groups') :].strip()
        if raw.startswith('='):
            raw = raw[1:].strip()

    parts = [norm_group_token(x) for x in raw.split(',') if str(x).strip()]
    return set(parts) if parts else None


def scope_hit(row: pd.Series, scope_groups: Optional[Set[str]]) -> bool:
    """判断样本是否命中 scope。"""
    if scope_groups is None:
        return True
    return len(sample_groups(row).intersection(scope_groups)) > 0


def stable_unit(seed: int, sample_id: int, salt: str = '') -> float:
    """生成稳定随机数 [0,1)，不依赖全局随机状态。"""
    text = f'{int(seed)}::{int(sample_id)}::{salt}'
    val = int(hashlib.sha256(text.encode('utf-8')).hexdigest()[:16], 16)
    return val / float(16**16)


def stable_index(seed: int, sample_id: int, salt: str, n: int) -> int:
    """生成稳定随机索引 [0,n)。"""
    if n <= 1:
        return 0
    text = f'{int(seed)}::{int(sample_id)}::{salt}'
    val = int(hashlib.sha256(text.encode('utf-8')).hexdigest()[:16], 16)
    return int(val % n)


def infer_setting(args) -> str:
    """根据 drop 开关推断 setting 名称。"""
    drop_profile = bool(getattr(args, 'drop_profile_in_test', False))
    drop_desc = bool(getattr(args, 'drop_description_in_test', False))
    drop_pref = bool(getattr(args, 'drop_preference_in_test', False))

    if drop_profile:
        return 'role_only'
    if drop_desc and (not drop_pref):
        return 'role_pref'
    if (not drop_desc) and drop_pref:
        return 'role_desc'
    return 'role_desc_pref'


def apply_name_replacement(samples: pd.DataFrame, args) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """在运行时生成 name replacement 结果，不改原始数据文件。"""
    mode = str(getattr(args, 'name_replace_mode', 'none')).lower()
    if mode == 'none':
        raise ValueError('apply_name_replacement should only be called when mode != none')
    if mode not in {'fixed', 'random', 'neutral', 'shuffle'}:
        raise ValueError(f'Unsupported name_replace_mode: {mode}')

    ratio = float(getattr(args, 'name_replace_ratio', 1.0))
    ratio = max(0.0, min(1.0, ratio))
    seed = int(getattr(args, 'name_replace_seed', 0))
    scope_groups = parse_scope(getattr(args, 'name_replace_scope', 'all'))

    df = samples.copy()
    if 'index' not in df.columns:
        df = df.reset_index().rename(columns={'index': 'index'})

    name_pool = sorted(
        {
            str(x).strip()
            for x in df.get('name', pd.Series(dtype=str)).fillna('')
            if str(x).strip() != ''
        }
    )

    rows: List[Dict[str, object]] = []
    candidates: List[int] = []
    sample_ids = [int(x) for x in df['index'].tolist()]
    row_by_id = {int(r['index']): r for _, r in df.iterrows()}

    for sid in sample_ids:
        row = row_by_id[sid]
        original_name = str(row.get('name', '')).strip()
        group = primary_group(row)
        hit_scope = scope_hit(row, scope_groups)
        hit_ratio = stable_unit(seed, sid, 'name_ratio') < ratio if ratio < 1.0 else True
        should_replace = bool(hit_scope and hit_ratio)

        if should_replace:
            candidates.append(sid)

        rows.append(
            {
                'sample_id': sid,
                'group': group,
                'original_name': original_name,
                'scope_hit': int(hit_scope),
                'ratio_hit': int(hit_ratio),
                'should_replace': int(should_replace),
            }
        )

    shuffled_map: Dict[int, str] = {}
    if mode == 'shuffle' and candidates:
        cand_names = [str(row_by_id[sid].get('name', '')).strip() for sid in candidates]
        g = torch.Generator()
        g.manual_seed(seed)
        perm = torch.randperm(len(candidates), generator=g).tolist()
        for i, sid in enumerate(candidates):
            shuffled_map[sid] = cand_names[perm[i]]

    fixed_name = 'FixedUser'
    neutral_name = 'ENTITY_X'
    replace_map: Dict[int, str] = {}

    for rec in rows:
        sid = int(rec['sample_id'])
        orig = str(rec['original_name'])
        should = bool(rec['should_replace'])
        replaced = orig

        if should:
            if mode == 'fixed':
                replaced = fixed_name
            elif mode == 'neutral':
                replaced = neutral_name
            elif mode == 'shuffle':
                replaced = shuffled_map.get(sid, orig)
            elif mode == 'random':
                if len(name_pool) > 1:
                    cands = [x for x in name_pool if x != orig]
                    if cands:
                        ridx = stable_index(seed, sid, 'name_random', len(cands))
                        replaced = cands[ridx]

        rec['replaced_name'] = replaced
        rec['name_replaced'] = int(replaced != orig)
        rec['name_replace_mode'] = mode
        rec['name_replace_ratio'] = ratio
        rec['name_replace_seed'] = seed
        replace_map[sid] = replaced

    df['name_for_prompt'] = [replace_map.get(int(x), '') for x in df['index'].tolist()]
    log_df = pd.DataFrame(rows)
    return df, log_df


def build_vision_ablation_plan(samples: pd.DataFrame, args) -> Dict[int, Dict[str, int]]:
    """构建视觉消融计划，不直接处理图像 Tensor。"""
    mode = str(getattr(args, 'vision_ablation_mode', 'none')).lower()
    if mode == 'none':
        return {}
    if mode not in {'no_image', 'shuffle_image'}:
        raise ValueError(f'Unsupported vision_ablation_mode: {mode}')

    df = samples.copy()
    if 'index' not in df.columns:
        df = df.reset_index().rename(columns={'index': 'index'})

    scope_groups = parse_scope(getattr(args, 'vision_ablation_scope', 'all'))
    seed = int(getattr(args, 'vision_ablation_seed', 0))

    sample_ids = [int(x) for x in df['index'].tolist()]
    row_by_id = {int(r['index']): r for _, r in df.iterrows()}
    selected = [sid for sid in sample_ids if scope_hit(row_by_id[sid], scope_groups)]

    plan: Dict[int, Dict[str, int]] = {}

    if mode == 'no_image':
        for sid in sample_ids:
            hit = int(sid in selected)
            plan[sid] = {
                'is_ablated': hit,
                'is_shuffled': 0,
                'donor_sample_id': sid,
            }
        return plan

    donor_map: Dict[int, int] = {}
    if selected:
        g = torch.Generator()
        g.manual_seed(seed)
        perm = torch.randperm(len(selected), generator=g).tolist()
        for i, sid in enumerate(selected):
            donor_map[sid] = int(selected[perm[i]])

    for sid in sample_ids:
        if sid in donor_map:
            donor = int(donor_map[sid])
            plan[sid] = {
                'is_ablated': 1,
                'is_shuffled': int(donor != sid),
                'donor_sample_id': donor,
            }
        else:
            plan[sid] = {
                'is_ablated': 0,
                'is_shuffled': 0,
                'donor_sample_id': sid,
            }

    return plan


def _resolve_oneshot_key_value(row: pd.Series, key_field: str, row_idx: int):
    """解析 one-shot 分组键。

    优先级：
    1) 用户显式字段存在 -> 直接使用该列。
    2) key_field=concept 时，依次回退 concept/name/attribute。
    3) 最终回退 index（每行唯一，不会相互屏蔽）。
    """
    kf = str(key_field or 'concept').strip()
    if kf and kf in row.index:
        val = row.get(kf)
        if pd.notna(val):
            sval = str(val).strip()
            if sval:
                return sval, kf

    if kf == 'concept':
        for cand in ('concept', 'name', 'attribute'):
            if cand in row.index:
                val = row.get(cand)
                if pd.notna(val):
                    sval = str(val).strip()
                    if sval:
                        return sval, cand

    if 'name' in row.index:
        val = row.get('name')
        if pd.notna(val):
            sval = str(val).strip()
            if sval:
                return sval, 'name'

    if 'index' in row.index:
        return f"index::{int(row.get('index'))}", 'index'
    return f"row::{int(row_idx)}", 'row'


def _apply_oneshot_profile_injection_flags(df: pd.DataFrame, key_field: str):
    """给样本打 one-shot 注入标记。

    返回：
    - flagged_df: 新增 `_oneshot_allow_injection` 列。
    - resolved_from: 实际使用的键来源（concept/name/attribute/index/...）。
    - n_keys: 唯一键个数（理论上等于注入次数）。
    """
    out = df.copy()
    seen = set()
    flags = []
    resolved_from = None

    for i, (_, row) in enumerate(out.iterrows()):
        key, src = _resolve_oneshot_key_value(row, key_field, i)
        if resolved_from is None:
            resolved_from = src
        if key in seen:
            flags.append(False)
        else:
            flags.append(True)
            seen.add(key)

    out['_oneshot_allow_injection'] = flags
    return out, (resolved_from or str(key_field or 'concept')), int(len(seen))


def _identity_token_for_row(row: pd.Series, identity_mode: str) -> str:
    """按 identity_mode 生成单条样本的身份 token。

    规则：
    - name: 优先用 `name_for_prompt`（用于 name 消融后的名字），没有则退回 `name`。
    - id:   使用 `index` 生成 `ID_00012` 形式的 token。
    - sks:  固定返回 `<sks>`。

    参数：
    - row: 含样本字段的 Series。
    - identity_mode: `sks` / `name` / `id`。

    返回：
    - str，可用于替换文本中的 `<sks>`。
    """
    mode = str(identity_mode or 'sks').strip().lower()
    if mode == 'name':
        return str(row.get('name_for_prompt', row.get('name', '<sks>')))
    if mode == 'id':
        try:
            return f"ID_{int(row.get('index')):05d}"
        except Exception:
            return 'ID_UNKNOWN'
    return '<sks>'


def _render_raw_with_identity(df: pd.DataFrame, identity_mode: str) -> pd.DataFrame:
    """将 raw 输出中的 `<sks>` 替换为当前评测模式对应身份 token。

    说明：
    - 评分逻辑读取的是 `prediction/answer` 归一化后的 hit，不依赖这个替换。
    - 这个函数只影响 `raw.csv` 的可读性，方便人工检查样本。
    - 当 `identity_mode == 'sks'` 时直接返回原 DataFrame，不做复制。

    参数：
    - df: 含原始文本列和 prediction 的 DataFrame。
    - identity_mode: `sks` / `name` / `id`。

    返回：
    - DataFrame：仅文本展示列可能被替换。
    """
    mode = str(identity_mode or 'sks').strip().lower()
    if mode == 'sks':
        return df

    out = df.copy()
    # 只替换这些会出现 <sks> 的文本列；不存在的列自动跳过。
    candidate_cols = [
        'question',
        'preference',
        'description_simple',
        'description_moderate',
        'description_detailed',
        'description_super_detailed',
    ] + [chr(x) for x in range(ord('A'), ord('Z') + 1)]
    text_cols = [c for c in candidate_cols if c in out.columns]
    if not text_cols:
        return out

    for i in out.index:
        tok = _identity_token_for_row(out.loc[i], mode)
        for c in text_cols:
            val = out.at[i, c]
            if isinstance(val, str) and '<sks>' in val:
                out.at[i, c] = val.replace('<sks>', tok)
    return out


def build_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """创建统一格式的 logger。

    行为：
    - 始终输出到 stdout。
    - 传入 `log_file` 时额外写文件日志。
    - 每次调用会清空同名 logger 现有 handler，避免重复打印。

    参数：
    - name: logger 名称。
    - log_file: 日志文件路径；为空时仅控制台输出。

    返回：
    - logging.Logger 实例。
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


def maybe_init_wandb(args, logger):
    """按命令行开关初始化 wandb。

    触发条件：
    - `--use-wandb` 为真，且当前 rank == 0。

    返回：
    - 初始化成功：wandb run 对象。
    - 关闭或失败：None。

    注意：
    - 初始化失败不会中断主流程，只写 warning。
    """
    if (not args.use_wandb) or args.rank != 0:
        return None
    try:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            mode=args.wandb_mode,
            reinit=True,
        )
        logger.info('WandB initialized.')
        return run
    except Exception as e:
        logger.warning(f'WandB init failed, fallback to no-wandb: {e}')
        return None


# ==============================
# 模块：模型适配层
# ==============================
# 职责：
# 1) 把底层 LLaVA 模型封装成项目内统一的 generate 接口。
# 2) 处理 token decode 兼容问题，避免非法 token id 触发崩溃。
# 3) 该模块只负责“推理接口适配”，不处理分数与文件存储。
# 4) 输入为 build_prompt 产物（含 input_ids/labels/image），输出纯文本预测。
def load_plain_lora_fallback_model(model_path: str, model_base: str, tokenizer, device, dtype):
    """Load a plain LLaVA+LoRA model without name-memory wrapping.

    This is used only as an eval-time fallback when the wrapped name-memory path
    returns an empty string. The fallback keeps the same tokenizer and image
    pathway, but bypasses the name-memory wrapper/runtime prefix logic.
    """
    from llava.model import LlavaLlamaForCausalLM
    from PrefMoE.peft import PeftModel

    load_kwargs = {
        'low_cpu_mem_usage': True,
        'config': AutoConfig.from_pretrained(model_path),
        'torch_dtype': dtype,
    }
    if str(device).startswith('cuda'):
        load_kwargs['device_map'] = 'auto'
    else:
        load_kwargs['device_map'] = {'': str(device)}

    model = LlavaLlamaForCausalLM.from_pretrained(model_base, **load_kwargs)

    non_lora_path = os.path.join(model_path, 'non_lora_trainables.bin')
    if os.path.exists(non_lora_path):
        non_lora_trainables = torch.load(non_lora_path, map_location='cpu')
        non_lora_trainables = {
            (k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()
        }
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {
                (k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()
            }
        model.load_state_dict(non_lora_trainables, strict=False)

    model = PeftModel.from_pretrained(model, model_path)
    model = model.merge_and_unload()
    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=torch.float16)
    model.eval()
    return model


class HFMessageModelAdapter:
    """LLaVA 推理适配器。

    参数：
    - tokenizer: HuggingFace tokenizer。
    - model: 已加载好的 LLaVA 模型。
    - max_new_tokens: 生成最大长度。

    关键假设：
    - `message` 字典至少包含 `input_ids`、`labels`、`image`。
    - `labels` 中用 IGNORE_INDEX 标记 prompt 区间。
    """

    def __init__(
        self,
        tokenizer,
        model,
        max_new_tokens: int = 32,
        model_path: str = '',
        model_base: str = '',
        enable_plain_fallback: bool = False,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.max_new_tokens = int(max_new_tokens)
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        # tokenizer 词表上界；decode 前用于过滤非法 token id。
        self._vocab_size = int(getattr(tokenizer, 'vocab_size', 0) or 0)
        self._model_path = str(model_path or '')
        self._model_base = str(model_base or '')
        self._enable_plain_fallback = bool(enable_plain_fallback)
        self._plain_fallback_model = None

    def _safe_decode(self, token_ids) -> str:
        """安全解码 token ids，过滤非法 id 后再 decode。

        输入：
        - token_ids: list / Tensor，通常是一段生成 token。

        返回：
        - str，解码后的文本；失败时返回空字符串。

        失败处理：
        - 非法类型、负数 token、超词表 token 会被跳过。
        - tokenizer.decode 抛异常时返回空字符串。
        """
        if token_ids is None:
            return ''
        if hasattr(token_ids, 'detach'):
            ids = token_ids.detach().cpu().tolist()
        else:
            ids = list(token_ids)

        clean_ids = []
        for x in ids:
            try:
                xid = int(x)
            except Exception:
                continue
            if xid < 0:
                continue
            if self._vocab_size > 0 and xid >= self._vocab_size:
                continue
            clean_ids.append(xid)

        if not clean_ids:
            return ''

        try:
            return self.tokenizer.decode(clean_ids, skip_special_tokens=True).strip()
        except Exception:
            return ''

    def generate(self, message, dataset=None, model=None):
        """执行单样本生成。

        输入：
        - message['input_ids']: 1D Tensor，包含 prompt + answer。
        - message['labels']:    1D Tensor，prompt 段为 IGNORE_INDEX。
        - message['image']:     3D Tensor，shape 通常为 [3, H, W]。

        返回：
        - str，模型文本输出。

        关键步骤：
        1) 由 labels 中 IGNORE_INDEX 计算 prompt 长度。
        2) 调用 model.generate（不采样，贪心）。
        3) 优先解码生成段；若为空，再退回解码整句。
        """
        prompt_len = int((message['labels'] == IGNORE_INDEX).sum().item())
        if prompt_len <= 0:
            # 兼容异常样本：如果 labels 没有正确标注，退化为整句作为 prompt。
            input_ids = message['input_ids']
            prompt_len = int(input_ids.shape[0])
        else:
            input_ids = message['input_ids'][:prompt_len]

        with torch.inference_mode():
            generate_kwargs = dict(
                input_ids=input_ids.unsqueeze(0).to(self.device),
                images=message['image'].unsqueeze(0).to(device=self.device, dtype=self.dtype),
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )
            if 'user_slot_id' in message:
                generate_kwargs['user_slot_id'] = message['user_slot_id'].view(1).to(self.device)
            if 'concept_id' in message:
                generate_kwargs['concept_id'] = message['concept_id'].view(1).to(self.device)
            output_ids = self.model.generate(**generate_kwargs)

        gen_ids = output_ids[0, prompt_len:]
        pred = self._safe_decode(gen_ids)
        if pred:
            return pred

        if self._enable_plain_fallback:
            fallback_pred = self._generate_plain_fallback(
                input_ids=input_ids,
                image_tensor=message['image'],
            )
            if fallback_pred:
                return fallback_pred
        # Do not fall back to the full prompt; an empty generation should stay empty.
        return ''

    def _generate_plain_fallback(self, input_ids: torch.Tensor, image_tensor: torch.Tensor) -> str:
        """Retry generation with a plain merged LLaVA+LoRA model.

        The wrapped name-memory inference path can occasionally return empty
        generations for adapter-style checkpoints. When that happens, use a
        lazily loaded plain model as a narrow eval-time fallback instead of
        returning an empty string.
        """
        if not self._model_path or not self._model_base:
            return ''
        if self._plain_fallback_model is None:
            self._plain_fallback_model = load_plain_lora_fallback_model(
                model_path=self._model_path,
                model_base=self._model_base,
                tokenizer=self.tokenizer,
                device=self.device,
                dtype=self.dtype,
            )

        with torch.inference_mode():
            output_ids = self._plain_fallback_model.generate(
                input_ids=input_ids.unsqueeze(0).to(self.device),
                images=image_tensor.unsqueeze(0).to(device=self.device, dtype=self.dtype),
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )
        gen_ids = output_ids[0, input_ids.shape[0] :]
        return self._safe_decode(gen_ids)


def load_model_adapter(
    model_path: str,
    model_base: str,
    max_new_tokens: int,
    enable_plain_lora_fallback: bool = False,
):
    """加载模型并包装为项目统一推理接口。

    参数：
    - model_path: 当前 pair 使用的 checkpoint 路径。
    - model_base: 基座模型路径（如 vicuna）。
    - max_new_tokens: 生成上限。

    返回：
    - tokenizer
    - image_processor
    - HFMessageModelAdapter

    说明：
    - 会根据 checkpoint 内容补全 model_name（如 llava/lora）。
    - 失败时会直接抛异常，由上层流程终止当前运行。
    """
    disable_torch_init()
    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path_v2(model_path)
    has_lora_adapter = os.path.exists(os.path.join(model_path, 'adapter_model.bin'))

    # 兼容命名：如果路径名不含 llava，但配置里是 llava，补上后缀。
    if 'llava' not in model_name.lower():
        try:
            cfg = AutoConfig.from_pretrained(model_path)
            if getattr(cfg, 'model_type', '').lower() == 'llava':
                model_name = f'{model_name}_llava'
        except Exception:
            pass

    # 兼容 LoRA 命名：用于匹配 builder 的加载分支。
    if has_lora_adapter and 'lora' not in model_name.lower():
        model_name = f'{model_name}_lora'

    tokenizer, model, image_processor, _ = load_pretrained_model_v2(model_path, model_base, model_name)
    model.eval()
    enable_plain_fallback = bool(
        enable_plain_lora_fallback
        and has_lora_adapter
        and getattr(getattr(model, 'config', None), 'use_name_memory', False)
    )
    adapter = HFMessageModelAdapter(
        tokenizer=tokenizer,
        model=model,
        max_new_tokens=max_new_tokens,
        model_path=model_path,
        model_base=model_base,
        enable_plain_fallback=enable_plain_fallback,
    )
    return tokenizer, image_processor, adapter


# ==============================
# 模块：数据视图适配层
# ==============================
# 职责：
# 1) 在不改原 dataset.build_prompt 的前提下，注入评测期开关逻辑。
# 2) 支持四种 eval prompt 变体：role_only / role_desc / role_pref / role_desc_pref。
# 3) 对样本字段做“临时裁剪”后再调用原 build_prompt，保证训练逻辑不受影响。
# 4) 输入是单条样本行，输出是模型可直接推理的 message dict。
class BridgeDatasetView:
    """对 `llava.eval.prefmoe.dataset.build_prompt` 的轻量包装。

    参数：
    - data_df: 当前评测子集 DataFrame（通常是某个 eval_task 的 test 集）。
    - tokenizer: tokenizer。
    - image_processor: 图像预处理器。
    - args: CLI 参数对象。

    关键字段开关：
    - drop_profile_in_test
    - drop_description_in_test
    - drop_preference_in_test
    """

    def __init__(self, data_df, tokenizer, image_processor, args):
        self.data = data_df
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.args = args
        self.profile_injection_mode = str(getattr(args, 'profile_injection_mode', 'per_question')).strip().lower()
        self.drop_profile_in_test = bool(getattr(args, 'drop_profile_in_test', False))
        self.drop_description_in_test = bool(getattr(args, 'drop_description_in_test', False))
        self.drop_preference_in_test = bool(getattr(args, 'drop_preference_in_test', False))
        self.name_memory_registry = None
        if bool(getattr(args, 'use_name_memory', False)):
            description_field = resolve_description_field(getattr(args, 'injection_description_prompt_type', 'hard_moderate'))
            self.name_memory_registry = build_fold_train_visible_user_registry(
                data_frame=getattr(args, '_name_memory_full_df'),
                data_split=getattr(args, '_name_memory_split'),
                fold_idx=getattr(args, 'fold', 0),
                image_folder=getattr(args, 'image_folder'),
                description_field=description_field,
            )

    @staticmethod
    def _strip_question_tail_description(question: str) -> str:
        """去掉 question 尾部拼接的描述段。

        输入：
        - question: 原始问题文本。

        返回：
        - str，截断后的问题。

        用途：
        - 在 drop description 的评测模式下，防止 description 从 question 尾部泄漏。
        """
        if not isinstance(question, str):
            return question

        markers = [
            '\nDescription:',
            '\ndescription:',
            '\n描述：',
            ' Description:',
            ' description:',
        ]
        new_q = question
        for mk in markers:
            pos = new_q.find(mk)
            if pos != -1:
                new_q = new_q[:pos].strip()
        return new_q

    def build_prompt(self, data_point, data_df=None):
        """根据评测开关构造单条样本 prompt。

        输入：
        - data_point: DataFrame 中一行（Series）。
        - data_df: 保留接口兼容，当前函数不依赖该参数。

        返回：
        - dict，至少包含 `input_ids`/`labels`/`image`。

        步骤：
        1) 若三个 drop 开关都为 False，直接调用原 build_prompt（最快路径）。
        2) 否则复制样本行并按开关清空 description/preference 字段。
        3) 调用原 build_prompt，保持 tokenization 与图像处理逻辑一致。
        """
        allow_profile = bool(data_point.get('_oneshot_allow_injection', True))
        oneshot_drop = self.profile_injection_mode == 'one_shot' and not allow_profile

        effective_drop_profile = self.drop_profile_in_test or oneshot_drop
        effective_drop_description = self.drop_description_in_test or effective_drop_profile
        effective_drop_preference = self.drop_preference_in_test or effective_drop_profile

        if not effective_drop_profile and not effective_drop_description and not effective_drop_preference:
            struct = build_prompt(data_point, self.tokenizer, self.image_processor, self.args)
        else:
            dp = data_point.copy()

            # 关闭 profile 或 description 时，四档描述都清空，并尝试清理 question 尾部描述残留。
            if effective_drop_description:
                for col in (
                    'description_simple',
                    'description_moderate',
                    'description_detailed',
                    'description_super_detailed',
                ):
                    if col in dp:
                        dp[col] = ''
                if 'question' in dp:
                    dp['question'] = self._strip_question_tail_description(dp['question'])

            # 关闭 profile 或 preference 时，偏好字段清空。
            if effective_drop_preference and 'preference' in dp:
                dp['preference'] = ''

            struct = build_prompt(dp, self.tokenizer, self.image_processor, self.args)

        if self.name_memory_registry is not None:
            visible_name = str(data_point.get('name_for_prompt', data_point.get('name', ''))).strip()
            struct['user_slot_id'] = torch.tensor(
                int(self.name_memory_registry['slot_by_name'].get(visible_name, 0)),
                dtype=torch.long,
            )
            if bool(getattr(self.args, 'name_memory_use_concept_id', False)):
                struct['concept_id'] = torch.tensor(
                    concept_to_factor_id(data_point.get('concept', '')),
                    dtype=torch.long,
                )
        return struct


# ==============================
# 模块：主流程控制（test / evaluate）
# ==============================
# 职责：
# 1) 管理单个 (model_task, eval_task) pair 的完整生命周期。
# 2) test 阶段执行推理并落盘 rank pkl；evaluate 阶段合并并评分。
# 3) 封装日志、模型惰性加载、输出文件命名等运行细节。
# 4) 外部脚本可通过循环调用该类完成下三角矩阵评测。
class PrefMoEBridge2:
    """Bridge-2 评测执行器（基线版）。"""

    def __init__(self, args, logger, wandb_run=None):
        """初始化当前 pair 的运行上下文。

        输入：
        - args: 命令行参数对象。
        - logger: 已配置 logger。
        - wandb_run: wandb run 或 None。

        主要产物：
        - self.subset: 当前 eval_task 的测试子集 DataFrame。
        - self.results_file/raw_file/score_file: 当前 pair 的输出路径。

        失败情形：
        - data_split_path/data_path 不可读会抛异常。
        - fold/task 越界会在索引阶段抛异常。
        """
        self.args = args
        self.logger = logger
        self.wandb_run = wandb_run

        self.rank = int(args.rank)
        self.world_size = int(args.world_size)

        # 从 split 文件中取当前 fold + eval_task 的 test 索引。
        self.data_split = load(args.data_split_path)
        self.data_dframe = load(args.data_path)
        self.test_idx = list(self.data_split[args.fold]['tasks'][args.eval_task]['test_idx'])
        if args.max_samples > 0:
            self.test_idx = self.test_idx[: args.max_samples]

        self.subset = self.data_dframe.iloc[self.test_idx].copy()
        if 'index' not in self.subset.columns:
            # 后续逻辑依赖 `index` 作为样本唯一键。
            self.subset = self.subset.reset_index().rename(columns={'index': 'index'})

        self.profile_injection_mode = str(getattr(args, 'profile_injection_mode', 'per_question')).strip().lower()
        self.oneshot_key_field = str(getattr(args, 'oneshot_key_field', 'concept')).strip()
        if self.profile_injection_mode not in {'per_question', 'one_shot'}:
            raise ValueError(f'unsupported profile_injection_mode: {self.profile_injection_mode}')
        if self.profile_injection_mode == 'one_shot':
            self.subset, resolved_from, n_keys = _apply_oneshot_profile_injection_flags(
                self.subset, self.oneshot_key_field
            )
            n_injected = int(self.subset['_oneshot_allow_injection'].sum())
            self.logger.info(
                f'Profile injection mode: one_shot (key_field={self.oneshot_key_field}, resolved_from={resolved_from}, '
                f'unique_keys={n_keys}, injected_samples={n_injected}/{len(self.subset)})'
            )
        else:
            self.subset['_oneshot_allow_injection'] = True
            self.logger.info('Profile injection mode: per_question (baseline behavior).')

        self.pair_tag, self.pred_root, self.raw_root, self.score_root = ensure_pair_dirs(
            args.output_root, args.fold, args.model_task, args.eval_task
        )

        # rank pkl 文件名格式：{rank}_{world_size}_{dataset_name}.pkl
        self.results_file = os.path.join(self.pred_root, f'{self.rank}_{self.world_size}_{self.args.dataset_name}.pkl')
        self.raw_file = os.path.join(self.raw_root, f'{self.pair_tag}_raw.csv')
        self.score_file = os.path.join(self.score_root, f'{self.pair_tag}_score.csv')

        self.model = None
        self.dataset = None

        # 记录当前 prompt 控制开关，便于排查 run 配置不一致。
        if bool(getattr(args, 'drop_profile_in_test', False)):
            self.logger.info('Test prompt mode: profile fields are DISABLED.')
        else:
            self.logger.info('Test prompt mode: profile fields are ENABLED.')

        if bool(getattr(args, 'drop_description_in_test', False)):
            self.logger.info('Test prompt mode: drop description fields is ENABLED.')
        else:
            self.logger.info('Test prompt mode: drop description fields is DISABLED.')

        if bool(getattr(args, 'drop_preference_in_test', False)):
            self.logger.info('Test prompt mode: drop preference fields is ENABLED.')
        else:
            self.logger.info('Test prompt mode: drop preference fields is DISABLED.')

        self.logger.info(f'Test prompt identity mode: {getattr(args, "identity_mode", "sks")}')

    def _lazy_load_model(self):
        """惰性加载模型与数据视图。

        说明：
        - 仅在第一次调用 test 时加载。
        - evaluate 阶段只读 pkl，不需要模型。
        """
        if self.model is not None:
            return

        tokenizer, image_processor, model_adapter = load_model_adapter(
            model_path=self.args.model_path,
            model_base=self.args.model_base,
            max_new_tokens=self.args.max_new_tokens,
            enable_plain_lora_fallback=self.args.enable_plain_lora_fallback,
        )
        self.args.use_name_memory = bool(getattr(getattr(model_adapter.model, 'config', None), 'use_name_memory', False))
        self.args._name_memory_full_df = self.data_dframe
        self.args._name_memory_split = self.data_split
        self.model = model_adapter
        self.dataset = BridgeDatasetView(self.subset, tokenizer, image_processor, self.args)

    def _build_struct_with_temp_mturn_args(self, row: pd.Series, enable: bool, n_turn: int):
        """临时覆写 multi-turn 参数，构造对比用 prompt 结构。

        用途：
        - dryrun 阶段对比 OFF(0-turn) 与 ON(n-turn) 的最终消息和哈希。
        - 覆写仅在函数内部生效，返回前恢复原参数。
        """
        old_enable = getattr(self.args, 'generic_conversation_enable', False)
        old_turn = getattr(self.args, 'generic_conversation_n_turn', 0)
        try:
            setattr(self.args, 'generic_conversation_enable', bool(enable))
            setattr(self.args, 'generic_conversation_n_turn', int(n_turn))
            return self.dataset.build_prompt(row, self.subset)
        finally:
            setattr(self.args, 'generic_conversation_enable', old_enable)
            setattr(self.args, 'generic_conversation_n_turn', old_turn)

    @staticmethod
    def _extract_prompt_debug(struct) -> dict:
        """从 build_prompt 结果提取可序列化的调试字段。"""
        return {
            'prompt_sha1': str(struct.get('_debug_prompt_sha1', '')),
            'messages_sha1': str(struct.get('_debug_msgs_sha1', '')),
            'total_messages': int(struct.get('_debug_total_messages', 0) or 0),
            'generic_turns': int(struct.get('_debug_generic_turns', 0) or 0),
            'generic_utterances': int(struct.get('_debug_generic_utterances', 0) or 0),
            'messages': struct.get('_debug_msgs', []),
            'prompt_text': struct.get('_debug_prompt_text', ''),
        }

    def _run_generic_conversation_dryrun(self) -> bool:
        """打印并落盘 OFF/ON 两套消息构造对比，随后退出 test。

        返回：
        - True: 已执行 dryrun，调用方应直接结束 test。
        - False: 未开启 dryrun，继续常规推理。
        """
        dryrun = int(getattr(self.args, 'generic_conversation_dryrun', 0) or 0)
        if dryrun <= 0:
            return False

        self._lazy_load_model()
        if len(self.subset) == 0:
            self.logger.info('[MultiTurnDryRun] empty subset, skip.')
            return True

        row = self.subset.iloc[0]
        current_enabled = bool(getattr(self.args, 'generic_conversation_enable', False))
        current_turns = int(getattr(self.args, 'generic_conversation_n_turn', 0) or 0)
        if not current_enabled or current_turns <= 0:
            current_enabled = False
            current_turns = 0

        struct_off = self._build_struct_with_temp_mturn_args(row, enable=False, n_turn=0)
        struct_on = self._build_struct_with_temp_mturn_args(row, enable=current_enabled, n_turn=current_turns)

        off_dbg = self._extract_prompt_debug(struct_off)
        on_dbg = self._extract_prompt_debug(struct_on)

        self.logger.info(
            f'[MultiTurnDryRun][OFF] prompt_sha1={off_dbg["prompt_sha1"]} '
            f'messages_sha1={off_dbg["messages_sha1"]} '
            f'total_messages={off_dbg["total_messages"]} generic_turns={off_dbg["generic_turns"]}'
        )
        self.logger.info(
            f'[MultiTurnDryRun][ON ] enabled={int(current_enabled)} turns={current_turns} '
            f'prompt_sha1={on_dbg["prompt_sha1"]} messages_sha1={on_dbg["messages_sha1"]} '
            f'total_messages={on_dbg["total_messages"]} generic_turns={on_dbg["generic_turns"]}'
        )

        preview_file = os.path.join(self.pred_root, f'{self.pair_tag}_multiturn_dryrun.json')
        dump(
            {
                'sample_id': int(row['index']),
                'off': off_dbg,
                'on': on_dbg,
                'note': '0-turn path remains the original baseline path; ON inserts generic turns only.',
            },
            preview_file,
        )
        self.logger.info(f'[MultiTurnDryRun] dumped prompt comparison to {preview_file}')
        self.logger.info('Dry-run finished. Exit without inference.')
        return True

    def _log_acc(self, acc):
        """统一打印并可选上报评测指标。"""
        self.logger.info(
            f'The evaluation of model {self.args.model_name} x dataset {self.args.dataset_name} has finished! '
        )
        self.logger.info('Evaluation Results:')

        if isinstance(acc, dict):
            self.logger.info(str(acc))
            if self.wandb_run is not None:
                try:
                    import wandb

                    wandb.log(acc)
                except Exception:
                    pass
            return

        if isinstance(acc, pd.DataFrame):
            self.logger.info('\n' + acc.to_string(index=False))
            if self.wandb_run is not None and len(acc) > 0:
                try:
                    import wandb

                    row = acc.iloc[0].to_dict()
                    log_row = {k: v for k, v in row.items() if k != 'split'}
                    wandb.log(log_row)
                except Exception:
                    pass

    def test(self, fold_id=None, task_id=None):
        """Stage-1：执行推理并写 rank 级 pkl。

        输入：
        - fold_id/task_id: 仅用于进度条展示，不参与计算。

        返回：
        - dict[int, str]：sample_index -> prediction 文本。

        数据流：
        1) 按 rank/world_size 切分子样本。
        2) 对每条样本 build_prompt -> model.generate。
        3) 周期性 dump 到 results_file，防止长任务中断丢进度。
        """
        if self._run_generic_conversation_dryrun():
            return {}

        self._lazy_load_model()

        sheet_indices = list(range(self.rank, len(self.subset), self.world_size))
        data = self.subset.iloc[sheet_indices]
        lt = len(sheet_indices)
        res = {}

        pbar = tqdm(range(lt))
        for i in pbar:
            idx = int(data.iloc[i]['index'])
            struct = self.dataset.build_prompt(data.iloc[i], data)
            response = self.model.generate(
                message=struct,
                dataset=self.args.dataset_name,
                model=self.args.model_name,
            )
            torch.cuda.empty_cache()

            if self.args.verbose:
                print(response, flush=True)

            res[idx] = response
            if (i + 1) % self.args.save_every == 0 or (i + 1) == lt:
                dump(res, self.results_file)

            if fold_id is not None and task_id is not None:
                pbar.set_description(f'Fold {fold_id + 1} Task {task_id + 1} | Test')
            else:
                pbar.set_description(f'Test {self.pair_tag}')
            pbar.set_postfix(response=str(response)[:120])

        return res

    def evaluate(self, results_file=None, circular=False):
        """Stage-2：合并预测、计算 hit/acc、写 raw 与 score。

        输入：
        - results_file/circular: 兼容旧接口保留，当前实现未使用。

        返回：
        - DataFrame 或 dict（由 report_acc 返回类型决定）。

        异常：
        - 如果任一 rank 的 pkl 缺失，抛 FileNotFoundError。

        注意：
        - 仅 rank0 执行 evaluate；其他 rank 直接返回。
        """
        if self.rank != 0:
            self.logger.info('Skip evaluate on non-zero rank.')
            return None

        # 1) 收集所有 rank 的预测 pkl。
        data_all = {}
        missing_files = []
        for i in range(self.world_size):
            pkl_file = os.path.join(self.pred_root, f'{i}_{self.world_size}_{self.args.dataset_name}.pkl')
            if not os.path.exists(pkl_file):
                missing_files.append(pkl_file)
                continue
            data_all.update(load(pkl_file))

        if missing_files:
            raise FileNotFoundError(f'Missing rank pkl files: {missing_files[:5]}')

        # 2) 按 subset 顺序组装 data_with_pred，保持样本顺序稳定。
        data_with_pred = self.subset.iloc[0:0].copy()
        predictions = []
        missing_indices = []
        for x in self.subset['index']:
            xi = int(x)
            if xi in data_all:
                data_with_pred = pd.concat(
                    [data_with_pred, self.subset.loc[self.subset['index'] == xi]],
                    ignore_index=True,
                )
                predictions.append(str(data_all[xi]))
            else:
                missing_indices.append(xi)

        if missing_indices:
            self.logger.warning(
                f'Missing predictions for {len(missing_indices)} samples, examples={missing_indices[:10]}'
            )

        data_with_pred['prediction'] = predictions

        # image 张量列不参与 CSV 序列化，移除可减少输出体积。
        if 'image' in data_with_pred.columns:
            data_with_pred.pop('image')

        # 3) 评分：先得到 hit，再汇总 acc。
        data = evaluate_mmpb_no_gpt(data_with_pred=data_with_pred, meta_data=self.subset)

        # 评分口径不变；raw 仅做可读性替换。
        raw_out = _render_raw_with_identity(data, getattr(self.args, 'identity_mode', 'sks'))
        raw_out.to_csv(self.raw_file, index=False)

        acc = report_acc(data)
        if isinstance(acc, pd.DataFrame):
            acc.to_csv(self.score_file, index=False)

        self._log_acc(acc)
        return acc


# ==============================
# 模块：Ablation 图像加载工具
# ==============================
# 职责：
# 1) 以与主流程一致的方式读取并预处理单张图。
# 2) 给 vision ablation 的 donor 图替换提供统一入口。
# 3) 该函数不做评分，也不改样本字段。
def _load_image_tensor(image_folder: str, rel_path: str, image_processor, image_aspect_ratio: str = 'pad') -> torch.Tensor:
    """读取并预处理单张图像。"""
    image = Image.open(os.path.join(image_folder, rel_path)).convert('RGB')

    if image_aspect_ratio == 'pad':
        def expand2square(pil_img, background_color):
            width, height = pil_img.size
            if width == height:
                return pil_img
            if width > height:
                result = Image.new(pil_img.mode, (width, width), background_color)
                result.paste(pil_img, (0, (width - height) // 2))
                return result
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

        image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))

    return image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]


# ==============================
# 模块：Name Ablation 执行器
# ==============================
# 职责：
# 1) 复用基线 test/evaluate 逻辑，仅在样本侧注入 name_for_prompt。
# 2) 落盘 name_replacement_log，便于后续分析替换覆盖与影响。
# 3) 保持 hit/acc 口径完全不变。
class PrefMoEBridge2NameAblation(PrefMoEBridge2):
    """Name 替换实验执行器。"""

    def __init__(self, args, logger, wandb_run=None):
        super().__init__(args=args, logger=logger, wandb_run=wandb_run)

        mode = str(getattr(args, 'name_replace_mode', 'none')).lower()
        if mode == 'none':
            raise ValueError('name-ablation mode requires --name_replace_mode != none')

        self.setting = infer_setting(args)
        self.name_mode = mode
        self.name_log_file = os.path.join(self.pred_root, 'name_replacement_log.csv')
        self.name_log_rank_file = os.path.join(self.pred_root, f'name_replacement_log_rank{self.rank}.csv')

        self.subset, self.name_plan_df = apply_name_replacement(self.subset, args)

        self.logger.info(
            f'Name replacement enabled: mode={self.name_mode}, '
            f'ratio={float(getattr(args, "name_replace_ratio", 1.0)):g}, '
            f'seed={int(getattr(args, "name_replace_seed", 0))}, '
            f'scope={getattr(args, "name_replace_scope", "all")}'
        )

    def _run_name_dryrun(self, n: int):
        """打印前 N 条替换结果并退出。"""
        n = min(max(int(n), 0), len(self.subset))
        if n <= 0:
            return

        self.logger.info(f'[NameDryRun] showing first {n} samples')
        for _, row in self.subset.iloc[:n].iterrows():
            sid = int(row['index'])
            original = str(row.get('name', ''))
            replaced = str(row.get('name_for_prompt', original))
            question = str(row.get('question', '')).replace('<sks>', replaced)
            snippet = question[:180].replace('\n', ' ')
            self.logger.info(
                f'[NameDryRun] sample_id={sid} group={primary_group(row)} '
                f'original_name={original} replaced_name={replaced} snippet={snippet}'
            )

    def _write_rank_log(self):
        """写当前 rank 的 name 替换日志。"""
        sheet_indices = list(range(self.rank, len(self.subset), self.world_size))
        local_df = self.subset.iloc[sheet_indices].copy()

        rows: List[Dict[str, object]] = []
        for _, row in local_df.iterrows():
            sid = int(row['index'])
            original = str(row.get('name', ''))
            replaced = str(row.get('name_for_prompt', original))
            rows.append(
                {
                    'sample_id': sid,
                    'group': primary_group(row),
                    'model_task': int(self.args.model_task),
                    'eval_task': int(self.args.eval_task),
                    'setting': self.setting,
                    'original_name': original,
                    'replaced_name': replaced,
                    'name_replaced': int(original != replaced),
                    'name_replace_mode': self.name_mode,
                    'name_replace_ratio': float(getattr(self.args, 'name_replace_ratio', 1.0)),
                    'name_replace_seed': int(getattr(self.args, 'name_replace_seed', 0)),
                }
            )

        pd.DataFrame(rows).to_csv(self.name_log_rank_file, index=False)

    def test(self, fold_id=None, task_id=None):
        """执行 test，并追加 name 日志。"""
        if self._run_generic_conversation_dryrun():
            return {}

        dryrun = int(getattr(self.args, 'name_replace_dryrun', 0))
        if dryrun > 0:
            self._run_name_dryrun(dryrun)
            self.logger.info('Dry-run finished. Exit without inference.')
            return {}

        res = super().test(fold_id=fold_id, task_id=task_id)
        self._write_rank_log()

        if self.world_size == 1:
            os.replace(self.name_log_rank_file, self.name_log_file)
        return res

    def evaluate(self, results_file=None, circular=False):
        """执行 evaluate，并在多卡场景合并 rank 日志。"""
        out = super().evaluate(results_file=results_file, circular=circular)
        if self.rank == 0 and self.world_size > 1:
            logs = sorted(glob.glob(os.path.join(self.pred_root, 'name_replacement_log_rank*.csv')))
            if logs:
                all_df = pd.concat([pd.read_csv(x) for x in logs], ignore_index=True)
                all_df.to_csv(self.name_log_file, index=False)
        return out


# ==============================
# 模块：Vision Ablation 执行器
# ==============================
# 职责：
# 1) 复用基线流程，在 test 阶段按计划替换 struct['image']。
# 2) 支持 no_image / shuffle_image，并记录 donor 来源。
# 3) 保持评分逻辑不变，仅改变视觉输入。
class PrefMoEBridge2VisionAblation(PrefMoEBridge2):
    """视觉输入消融执行器。"""

    def __init__(self, args, logger, wandb_run=None):
        super().__init__(args=args, logger=logger, wandb_run=wandb_run)

        mode = str(getattr(args, 'vision_ablation_mode', 'none')).lower()
        if mode == 'none':
            raise ValueError('vision-ablation mode requires --vision_ablation_mode != none')

        self.setting = infer_setting(args)
        self.vision_mode = mode
        self.vision_plan = build_vision_ablation_plan(self.subset, args)
        self.vision_log_file = os.path.join(self.pred_root, 'vision_ablation_log.csv')
        self.vision_log_rank_file = os.path.join(self.pred_root, f'vision_ablation_log_rank{self.rank}.csv')

        self._row_by_id = {int(r['index']): r for _, r in self.subset.iterrows()}
        self._image_cache: Dict[int, torch.Tensor] = {}
        self._dummy_image: Optional[torch.Tensor] = None

        self.logger.info(
            f'Vision ablation enabled: mode={self.vision_mode}, '
            f'seed={int(getattr(args, "vision_ablation_seed", 0))}, '
            f'scope={getattr(args, "vision_ablation_scope", "all")}'
        )

    def _get_image_by_sample_id(self, sample_id: int) -> torch.Tensor:
        """按样本 id 获取图像 Tensor（带缓存）。"""
        sid = int(sample_id)
        if sid in self._image_cache:
            return self._image_cache[sid].clone()

        if self.dataset is None:
            self._lazy_load_model()

        row = self._row_by_id[sid]
        image = _load_image_tensor(
            image_folder=self.args.image_folder,
            rel_path=str(row['image_path']),
            image_processor=self.dataset.image_processor,
            image_aspect_ratio=self.args.image_aspect_ratio,
        )
        self._image_cache[sid] = image
        return image.clone()

    def _apply_vision_ablation_to_struct(self, struct: Dict[str, torch.Tensor], sample_id: int):
        """对单条样本 struct 应用视觉消融。"""
        plan = self.vision_plan.get(int(sample_id))
        if not plan or int(plan.get('is_ablated', 0)) == 0:
            return struct, 0, 0, int(sample_id)

        if self.vision_mode == 'no_image':
            if self._dummy_image is None or tuple(self._dummy_image.shape) != tuple(struct['image'].shape):
                self._dummy_image = torch.zeros_like(struct['image'])
            struct['image'] = self._dummy_image.clone()
            return struct, 1, 0, int(sample_id)

        donor_id = int(plan.get('donor_sample_id', sample_id))
        struct['image'] = self._get_image_by_sample_id(donor_id)
        return struct, 1, int(plan.get('is_shuffled', 0)), donor_id

    def _run_vision_dryrun(self, n: int):
        """打印前 N 条视觉消融计划并退出。"""
        n = min(max(int(n), 0), len(self.subset))
        if n <= 0:
            return

        self.logger.info(f'[VisionDryRun] showing first {n} samples')
        for _, row in self.subset.iloc[:n].iterrows():
            sid = int(row['index'])
            plan = self.vision_plan.get(sid, {'is_ablated': 0, 'is_shuffled': 0, 'donor_sample_id': sid})
            self.logger.info(
                f'[VisionDryRun] sample_id={sid} group={primary_group(row)} '
                f'is_ablated={int(plan.get("is_ablated", 0))} '
                f'is_shuffled={int(plan.get("is_shuffled", 0))} '
                f'donor_sample_id={int(plan.get("donor_sample_id", sid))}'
            )

    def _write_rank_log(self, rows: List[Dict[str, object]]):
        """写当前 rank 的视觉消融日志。"""
        pd.DataFrame(rows).to_csv(self.vision_log_rank_file, index=False)

    def test(self, fold_id=None, task_id=None):
        """执行 test，并在推理前替换视觉输入。"""
        if self._run_generic_conversation_dryrun():
            return {}

        dryrun = int(getattr(self.args, 'vision_ablation_dryrun', 0))
        if dryrun > 0:
            self._run_vision_dryrun(dryrun)
            self.logger.info('Dry-run finished. Exit without inference.')
            return {}

        self._lazy_load_model()

        sheet_indices = list(range(self.rank, len(self.subset), self.world_size))
        data = self.subset.iloc[sheet_indices]
        lt = len(sheet_indices)
        res = {}
        log_rows: List[Dict[str, object]] = []

        pbar = tqdm(range(lt))
        for i in pbar:
            row = data.iloc[i].copy()
            idx = int(row['index'])

            struct = self.dataset.build_prompt(row, data)
            self._image_cache[idx] = struct['image'].clone()

            struct, is_ablated, is_shuffled, donor_id = self._apply_vision_ablation_to_struct(struct, idx)

            response = self.model.generate(
                message=struct,
                dataset=self.args.dataset_name,
                model=self.args.model_name,
            )
            torch.cuda.empty_cache()

            if self.args.verbose:
                print(response, flush=True)

            res[idx] = response
            log_rows.append(
                {
                    'sample_id': idx,
                    'group': primary_group(row),
                    'model_task': int(self.args.model_task),
                    'eval_task': int(self.args.eval_task),
                    'setting': self.setting,
                    'vision_ablation_mode': self.vision_mode,
                    'vision_ablation_seed': int(getattr(self.args, 'vision_ablation_seed', 0)),
                    'is_ablated': int(is_ablated),
                    'is_shuffled': int(is_shuffled),
                    'donor_sample_id': int(donor_id),
                }
            )

            if (i + 1) % self.args.save_every == 0 or (i + 1) == lt:
                dump(res, self.results_file)

            if fold_id is not None and task_id is not None:
                pbar.set_description(f'Fold {fold_id + 1} Task {task_id + 1} | Test')
            else:
                pbar.set_description(f'Test {self.pair_tag}')
            pbar.set_postfix(response=str(response)[:120])

        self._write_rank_log(log_rows)
        if self.world_size == 1:
            os.replace(self.vision_log_rank_file, self.vision_log_file)
        return res

    def evaluate(self, results_file=None, circular=False):
        """执行 evaluate，并在多卡场景汇总 rank 日志。"""
        out = super().evaluate(results_file=results_file, circular=circular)
        if self.rank == 0 and self.world_size > 1:
            logs = sorted(glob.glob(os.path.join(self.pred_root, 'vision_ablation_log_rank*.csv')))
            if logs:
                all_df = pd.concat([pd.read_csv(x) for x in logs], ignore_index=True)
                all_df.to_csv(self.vision_log_file, index=False)
        return out


# ==============================
# 模块：CLI 参数构建
# ==============================
# 职责：
# 1) 定义 test/evaluate 两个子命令及共享参数。
# 2) 保留与历史脚本一致的参数名，避免批处理脚本失效。
# 3) 参数默认值即“主线 baseline”配置，不改变现有跑法。
# 4) 解析失败时 argparse 会直接退出并返回非 0 状态。
def build_parser():
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description='PrefMoE Bridge-2 (baseline: role-only eval modes)')
    sub = parser.add_subparsers(dest='cmd', required=True)

    def add_shared(sp):
        """为 test/evaluate 注入共享参数。"""
        sp.add_argument('--model-base', type=str, default='./checkpoints/vicuna-7b-v1.5')
        sp.add_argument('--model-name', type=str, default='prefmllm')
        sp.add_argument('--fold', type=int, required=True)
        sp.add_argument('--eval-task', type=int, required=True)
        sp.add_argument('--model-task', type=int, default=-1)
        sp.add_argument('--data-split-path', type=str, required=True)
        sp.add_argument('--data-path', type=str, required=True)
        sp.add_argument('--dataset-name', type=str, default='MMPB clean')
        sp.add_argument('--image-folder', type=str, default='./data')
        sp.add_argument('--identity-mode', type=str, default='sks', choices=['sks', 'name', 'id'])
        sp.add_argument('--injection-description-prompt-type', type=str, default='hard_moderate')
        sp.add_argument('--injection-preference-prompt-type', type=str, default='explicit')
        sp.add_argument('--injection-compose-mode', type=str, default='legacy', choices=['legacy', 'image_text_des_pre'])
        sp.add_argument('--injection-prompt-n-images', type=int, default=4)
        sp.add_argument('--image-aspect-ratio', type=str, default='pad')
        sp.add_argument('--name_memory_use_concept_id', action='store_true')

        # 四种主线 eval 模式控制：
        # role_only  : --drop-profile-in-test
        # role_desc  : --drop-preference-in-test
        # role_pref  : --drop-description-in-test
        # role_desc_pref: 三个 drop 都不加
        sp.add_argument('--drop-profile-in-test', action='store_true')
        sp.add_argument('--drop-description-in-test', action='store_true')
        sp.add_argument('--drop-preference-in-test', action='store_true')
        sp.add_argument('--profile-injection-mode', type=str, default='per_question', choices=['per_question', 'one_shot'])
        sp.add_argument('--oneshot-key-field', type=str, default='concept')

        # Generic multi-turn（评测可选）：
        # - 默认关闭（0-turn），此时 prompt/输入/评分与历史行为完全一致。
        # - 开启后仅在 injection 与 final query 之间插入中性对话，不改评分口径。
        # - 0-turn: 不加参数；10-turn: --generic-conversation-enable --generic-conversation-n-turn 10
        sp.add_argument('--generic-conversation-enable', action='store_true')
        sp.add_argument('--generic-conversation-n-turn', type=int, default=0)
        sp.add_argument('--generic-conversation-seed', type=int, default=0)
        sp.add_argument('--generic-conversation-path', type=str, default='')
        sp.add_argument('--generic-conversation-dryrun', type=int, default=0)

        # Name ablation（默认 none，不影响 baseline）
        sp.add_argument('--name_replace_mode', type=str, default='none', choices=['none', 'fixed', 'random', 'neutral', 'shuffle'])
        sp.add_argument('--name_replace_ratio', type=float, default=1.0)
        sp.add_argument('--name_replace_seed', type=int, default=0)
        sp.add_argument('--name_replace_scope', type=str, default='all')
        sp.add_argument('--name_replace_dryrun', type=int, default=0)

        # Vision ablation（默认 none，不影响 baseline）
        sp.add_argument('--vision_ablation_mode', type=str, default='none', choices=['none', 'no_image', 'shuffle_image'])
        sp.add_argument('--vision_ablation_seed', type=int, default=0)
        sp.add_argument('--vision_ablation_scope', type=str, default='all')
        sp.add_argument('--vision_ablation_dryrun', type=int, default=0)

        sp.add_argument('--output-root', type=str, required=True)
        sp.add_argument('--max-samples', type=int, default=-1)
        sp.add_argument('--rank', type=int, default=0)
        sp.add_argument('--world-size', type=int, default=1)
        sp.add_argument('--verbose', action='store_true')
        sp.add_argument('--enable-plain-lora-fallback', action='store_true')
        sp.add_argument('--log-file', type=str, default='')
        sp.add_argument('--use-wandb', action='store_true')
        sp.add_argument('--wandb-project', type=str, default='prefmllm-eval')
        sp.add_argument('--wandb-name', type=str, default='')
        sp.add_argument('--wandb-mode', type=str, default='offline', choices=['offline', 'online', 'disabled'])

    p_test = sub.add_parser('test', help='Stage-1: run inference and dump rank pkl')
    add_shared(p_test)
    p_test.add_argument('--model-path', type=str, required=True)
    p_test.add_argument('--save-every', type=int, default=20)
    p_test.add_argument('--max-new-tokens', type=int, default=32)

    p_eval = sub.add_parser('evaluate', help='Stage-2: merge rank pkl and score')
    add_shared(p_eval)
    # 兼容批处理脚本：允许 evaluate 接收 --model-path，但不会使用。
    p_eval.add_argument('--model-path', type=str, default='')
    p_eval.add_argument('--save-every', type=int, default=20)
    p_eval.add_argument('--max-new-tokens', type=int, default=32)

    return parser


# ==============================
# 模块：程序入口
# ==============================
# 职责：
# 1) 解析参数并校验 test/evaluate 所需最小字段。
# 2) 初始化日志、可选 wandb、主执行器实例。
# 3) 按子命令分发到 test 或 evaluate。
# 4) 确保 wandb 在 finally 中关闭，避免后台进程残留。
def main():
    """脚本入口函数。"""
    parser = build_parser()
    args = parser.parse_args()

    if args.model_task < 0 and args.cmd == 'test':
        args.model_task = infer_model_task(args.model_path, args.eval_task)
    elif args.model_task < 0 and args.cmd == 'evaluate':
        raise ValueError('evaluate stage requires explicit --model-task')

    _apply_generic_conversation_output_suffix(args)
    if str(getattr(args, 'profile_injection_mode', 'per_question')).strip().lower() == 'one_shot':
        kf = str(getattr(args, 'oneshot_key_field', 'concept')).strip() or 'concept'
        args.output_root = augment_output_root(args.output_root, [f'oneshot_{kf}'])

    # 统一入口下仅允许一种 ablation 模式激活，避免语义重叠。
    name_mode = str(getattr(args, 'name_replace_mode', 'none')).lower()
    vision_mode = str(getattr(args, 'vision_ablation_mode', 'none')).lower()
    if name_mode != 'none' and vision_mode != 'none':
        raise ValueError('Only one ablation mode can be enabled at a time: name OR vision.')

    runner_cls = PrefMoEBridge2
    log_prefix = 'bridge2'
    if name_mode != 'none':
        ratio = float(getattr(args, 'name_replace_ratio', 1.0))
        seed = int(getattr(args, 'name_replace_seed', 0))
        args.output_root = augment_output_root(args.output_root, [f'namerepl_{name_mode}_r{ratio:g}_s{seed}'])
        runner_cls = PrefMoEBridge2NameAblation
        log_prefix = 'bridge2_nameabl'
    elif vision_mode != 'none':
        seed = int(getattr(args, 'vision_ablation_seed', 0))
        args.output_root = augment_output_root(args.output_root, [f'visabl_{vision_mode}_s{seed}'])
        runner_cls = PrefMoEBridge2VisionAblation
        log_prefix = 'bridge2_visabl'

    log_name = f'{log_prefix}_fold{args.fold}_mt{args.model_task}_et{args.eval_task}_r{args.rank}w{args.world_size}'
    logger = build_logger(log_name, args.log_file if args.log_file else None)
    wandb_run = maybe_init_wandb(args, logger)

    if args.cmd == 'evaluate' and str(getattr(args, 'model_path', '')).strip():
        logger.warning('`--model-path` is ignored in evaluate stage.')

    runner = runner_cls(args=args, logger=logger, wandb_run=wandb_run)

    try:
        if args.cmd == 'test':
            runner.test(fold_id=args.fold, task_id=args.eval_task)
        elif args.cmd == 'evaluate':
            runner.evaluate()
        else:
            raise ValueError(args.cmd)
    finally:
        if wandb_run is not None:
            try:
                import wandb

                wandb.finish()
            except Exception:
                pass


if __name__ == '__main__':
    main()
