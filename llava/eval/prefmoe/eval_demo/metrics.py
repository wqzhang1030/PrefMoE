# bridge -> utils.get_hit -> metrics.score_prediction -> metrics.answer_to_text
import re
import string
import copy as cp
from typing import Any, Dict
from collections import defaultdict

import numpy as np
import pandas as pd


# ==============================
# 模块：标签与同义词常量
# ==============================
# 职责：
# 1) 定义数据集数值标签到文本标签的映射。
# 2) 定义 yes/no 同义词集合，用于判定题归一化。
# 3) 常量只描述映射规则，不包含任何计算逻辑。
# 4) 若修改这些集合，会直接影响 hit 判定结果。
LABEL_TO_TEXT = {
    0: 'a',
    1: 'b',
    2: 'c',
    3: 'd',
    4: 'yes',
    5: 'no',
}

YES_VARIANTS = {'yes', 'y', 'yeah', 'yep', 'sure', 'affirmative', 'ok', 'okay'}
NO_VARIANTS = {'no', 'n', 'nope', 'nah', 'negative', '0', 'false'}


# ==============================
# 模块：基础归一化函数
# ==============================
# 职责：
# 1) 将 answer/prediction 统一为可比较的标准标签。
# 2) 对无效输入做保守处理：返回空串或首词，避免抛异常。
# 3) 这些函数被后续 score_prediction 全流程复用。
# 4) 该模块不直接计算 hit，只负责格式标准化。
def _safe_float_to_int(value: Any):
    """把可数值化输入安全转为 int。

    参数：
    - value: 任意对象，常见为数字、字符串、NaN。

    返回：
    - int 或 None（无法转换时）。
    """
    try:
        return int(float(value))
    except Exception:
        return None


def answer_to_text(answer_raw: Any) -> str:
    """把数据集 answer 归一化到 `a/b/c/d/yes/no`。

    处理顺序：
    1) 先尝试按数值标签映射（0..5）。
    2) 再尝试按文本直接匹配（大小写不敏感）。

    参数：
    - answer_raw: 原始答案字段。

    返回：
    - 标准化标签字符串；无法识别返回空串。
    """
    val = _safe_float_to_int(answer_raw)
    if val is not None and val in LABEL_TO_TEXT:
        return LABEL_TO_TEXT[val]

    text = str(answer_raw).strip().lower()
    if text in {'a', 'b', 'c', 'd', 'yes', 'no'}:
        return text
    return ''


def normalize_yes_no(text: str) -> str:
    """将判定题预测归一化到 yes/no 或首词。

    规则：
    - 取首词，去掉标点。
    - 首词命中 YES/NO 同义词则返回 yes/no。
    - 否则返回首词本身（便于后续比对失败定位）。
    """
    text = str(text).lower().strip()
    first_word = text.split()[0] if text else ''
    first_word = re.sub(r'[^\w\s]', '', first_word)

    if first_word in YES_VARIANTS:
        return 'yes'
    if first_word in NO_VARIANTS:
        return 'no'
    return first_word


# ==============================
# 模块：多选题解析（inconsistency）
# ==============================
# 职责：
# 1) 从模型自由文本中提取 A/B/C/D 选项。
# 2) 先走严格的 option/text 推断，再按官方 exact-matching 语义兜底为 Z。
# 3) 保持与 VLMEvalKit 行为对齐，避免评测口径漂移。
# 4) 该模块只返回规范化预测，不计算 hit。
def build_choices(item: Dict[str, Any]) -> Dict[str, str]:
    """从样本行抽取可用选项字典。

    输入：
    - item: 样本字典，可能包含 A/B/C/D... 列。

    返回：
    - dict，如 {'A': '...', 'B': '...'}。
    """
    ret: Dict[str, str] = {}
    for ch in string.ascii_uppercase:
        if ch in item and not pd.isna(item[ch]):
            ret[ch] = str(item[ch])
    return ret


def can_infer_option(answer: str, choices: Dict[str, str]):
    """优先通过显式选项字符推断答案（A/B/C...）。

    返回：
    - 单个选项字符（如 'A'）
    - 'Z'（拒答/无效回答）
    - False（无法确定）
    """
    answer = str(answer)
    if 'Failed to obtain answer via API' in answer:
        return False

    reject_to_answer = [
        "Sorry, I can't help with images of people yet.",
        "I can't process this file.",
        "I'm sorry, but without the image provided",
        'Cannot determine the answer',
    ]
    for err in reject_to_answer:
        if err in answer:
            return 'Z'

    def count_choice(splits, cand_choices, prefix='', suffix=''):
        cnt = 0
        for c in cand_choices:
            if prefix + c + suffix in splits:
                cnt += 1
        return cnt

    answer_mod = cp.copy(answer)
    chars = '.()[],:;!*#{}'
    for c in chars:
        answer_mod = answer_mod.replace(c, ' ')

    raw_splits = [x.strip() for x in answer_mod.split()]
    # 兼容模型输出小写单字母（a/b/c/d）的情况。
    splits = [tok.upper() if len(tok) == 1 and tok.isalpha() else tok for tok in raw_splits]
    count = count_choice(splits, choices)

    if count == 1:
        for ch in choices:
            if ch in splits:
                return ch
    elif count == 0 and count_choice(splits, {'Z', ''}) == 1:
        return 'Z'
    return False


def can_infer_text(answer: str, choices: Dict[str, str]):
    """通过选项文本内容反推选项字符。

    示例：
    - 回答中包含唯一一个选项文本，则返回对应选项字母。
    """
    answer = str(answer).lower()
    norm_choices = {k: str(v).lower() for k, v in choices.items()}

    cands = []
    for k, v in norm_choices.items():
        if v in answer:
            cands.append(k)
    if len(cands) == 1:
        return cands[0]
    return False


def can_infer(answer: str, choices: Dict[str, str]):
    """综合推断：先 option，再 text。"""
    copt = can_infer_option(str(answer), choices)
    return copt if copt else can_infer_text(str(answer), choices)


def extract_characters_regex(s, choices=('(A)', '(B)', '(C)', '(D)', '(E)')):
    """正则抽取选项字符（保留历史实现，当前主流程未使用）。"""
    if isinstance(s, dict):
        s = ''
    s = str(s).strip()

    answer_prefixes = [
        'The best answer is',
        'The correct answer is',
        'The answer is',
        'The answer',
        'The best option is',
        'The correct option is',
        'Best answer:',
        'Best option:',
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, '')

    if len(s.split()) > 10 and not re.search('[ABCDE]', s):
        return ''

    matches = re.search(r'[ABCDE]', s)
    if matches is None:
        for choice in choices:
            if s.lower() in choice.lower():
                return choice[1]
        return ''
    return matches[0]


def normalize_mcq_prediction(pred_raw: str, data_point: Dict[str, Any]) -> str:
    """inconsistency 题型预测归一化。

    规则：
    - 无选项列时返回空串。
    - can_infer 成功返回小写选项字母。
    - can_infer 失败返回 'z'，与官方 exact-matching 对齐。
    """
    choices = build_choices(data_point)
    if not choices:
        return ''

    inferred = can_infer(pred_raw, choices)
    if inferred:
        return str(inferred).strip().lower()

    # 与官方 exact_matching（model=None）保持一致：失败直接记 Z。
    return 'z'


def normalize_prediction(pred_raw: str, l2_category: str, data_point: Dict[str, Any]) -> str:
    """按 l2-category 选择归一化策略。"""
    l2 = str(l2_category).strip().lower()
    if l2 == 'inconsistency':
        return normalize_mcq_prediction(pred_raw, data_point)
    if l2 in {'awareness', 'overconcept'}:
        return normalize_yes_no(pred_raw)
    return str(pred_raw).strip().lower()


# ==============================
# 模块：单条打分与聚合
# ==============================
# 职责：
# 1) 单样本：产出 gt_norm / pred_norm / hit。
# 2) 多样本：汇总 overall、l2-category、category 维度统计。
# 3) 所有 hit 都是严格相等比较，不做语义相似度。
# 4) 聚合函数返回 DataFrame，方便直接落 CSV。
def score_prediction(pred_raw: str, answer_raw: Any, l2_category: str, data_point: Dict[str, Any]) -> Dict[str, Any]:
    """单条样本打分。

    返回字段：
    - gt_norm: 归一化标准答案。
    - pred_norm: 归一化模型输出。
    - hit: 0/1，严格字符串匹配且非空。
    """
    gt_norm = answer_to_text(answer_raw)
    pred_norm = normalize_prediction(pred_raw, l2_category, data_point)
    hit = int(pred_norm == gt_norm and pred_norm != '')

    return {
        'gt_norm': gt_norm,
        'pred_norm': pred_norm,
        'hit': hit,
    }


def _safe_mean(series: pd.Series):
    """安全均值：空序列返回 NaN。"""
    if len(series) == 0:
        return np.nan
    return float(series.mean())


def aggregate_scores(df: pd.DataFrame) -> pd.DataFrame:
    """聚合为一行简表：overall + l2-category + 可选 category。"""
    out = {
        'overall': _safe_mean(df['hit']),
        'n_samples': int(len(df)),
    }

    for cat in ['inconsistency', 'awareness', 'overconcept']:
        mask = df['l2-category'].astype(str).str.lower() == cat
        out[cat] = _safe_mean(df.loc[mask, 'hit'])
        out[f'n_{cat}'] = int(mask.sum())

    if 'category' in df.columns:
        for cat in sorted(df['category'].dropna().astype(str).unique().tolist()):
            mask = df['category'].astype(str) == cat
            out[f'category::{cat}'] = _safe_mean(df.loc[mask, 'hit'])

    return pd.DataFrame([out])


# ==============================
# 模块：MMPB 报表口径（兼容 Bridge2）
# ==============================
# 职责：
# 1) 复刻项目现有 report_acc 输出风格，供 bridge2 直接复用。
# 2) 产出 split / Overall / 多级组合列。
# 3) 列名保持历史兼容，避免下游脚本解析失败。
# 4) 该模块不做文件 IO，只返回 DataFrame。
def report_acc_mmpb(df: pd.DataFrame) -> pd.DataFrame:
    """按 MMPB 口径汇总准确率表。"""
    res = defaultdict(list)

    work_df = df.copy()
    if 'split' in work_df.columns:
        splits = list(set(work_df['split']))
        res['split'] = splits
    else:
        work_df['split'] = ['none'] * len(work_df)
        res['split'] = ['none']

    # Overall
    res['Overall'] = [np.mean(work_df[work_df['split'] == sp]['hit']) for sp in res['split']]

    # category
    if 'category' in work_df.columns:
        categories = sorted(work_df['category'].dropna().unique())
        for cat in categories:
            sub_df = work_df[work_df['category'] == cat]
            res[str(cat)] = [np.mean(sub_df[sub_df['split'] == sp]['hit']) for sp in res['split']]

    # attribute
    if 'attribute' in work_df.columns:
        attributes = sorted(work_df['attribute'].dropna().unique())
        for attr in attributes:
            sub_df = work_df[work_df['attribute'] == attr]
            res[str(attr)] = [np.mean(sub_df[sub_df['split'] == sp]['hit']) for sp in res['split']]

    # l2-category
    if 'l2-category' in work_df.columns:
        l2_cats = sorted(work_df['l2-category'].dropna().unique())
        for l2 in l2_cats:
            sub_df = work_df[work_df['l2-category'] == l2]
            res[str(l2)] = [np.mean(sub_df[sub_df['split'] == sp]['hit']) for sp in res['split']]

    # category + l2-category
    if 'category' in work_df.columns and 'l2-category' in work_df.columns:
        grouped = work_df.groupby(['category', 'l2-category'])
        for (cat, l2_cat), sub_df in grouped:
            name = f'{cat} + {l2_cat}'
            res[name] = [np.mean(sub_df[sub_df['split'] == sp]['hit']) for sp in res['split']]

    # category + attribute
    if 'category' in work_df.columns and 'attribute' in work_df.columns:
        grouped = work_df.groupby(['category', 'attribute'])
        for (cat, attr), sub_df in grouped:
            name = f'{cat} + {attr}'
            res[name] = [np.mean(sub_df[sub_df['split'] == sp]['hit']) for sp in res['split']]

    # category + l2-category + concept
    if 'category' in work_df.columns and 'l2-category' in work_df.columns and 'concept' in work_df.columns:
        df_filtered = work_df.dropna(subset=['concept'])
        grouped = df_filtered.groupby(['category', 'l2-category', 'concept'])
        for (cat, l2_cat, concept), sub_df in grouped:
            name = f'{cat} + {l2_cat} + {concept}'
            res[name] = [np.mean(sub_df[sub_df['split'] == sp]['hit']) for sp in res['split']]

    # category + attribute + l2-category + target
    if (
        'category' in work_df.columns
        and 'l2-category' in work_df.columns
        and 'target' in work_df.columns
        and 'attribute' in work_df.columns
    ):
        df_filtered = work_df.dropna(subset=['target'])
        grouped = df_filtered.groupby(['category', 'attribute', 'l2-category', 'target'])
        for (cat, attr, l2_cat, tgt), sub_df in grouped:
            name = f'{cat} + +{attr} + {l2_cat} + {tgt}'
            res[name] = [np.mean(sub_df[sub_df['split'] == sp]['hit']) for sp in res['split']]

    return pd.DataFrame(res)


def evaluate_mmpb_no_gpt(data_with_pred: pd.DataFrame, meta_data: pd.DataFrame = None) -> pd.DataFrame:
    """Bridge2 入口：对带 prediction 的 DataFrame 逐条评分。

    参数：
    - data_with_pred: 至少包含 `prediction` 和 `answer` 的 DataFrame。
    - meta_data: 兼容旧接口保留，当前实现不使用。

    返回：
    - 原 DataFrame 的副本，并新增：
      - gt_norm
      - pred_norm
      - hit
      - get_hit（历史别名）
    """
    work_df = data_with_pred.copy().reset_index(drop=True)
    gt_list = []
    pred_list = []
    hit_list = []

    for _, row in work_df.iterrows():
        scored = score_prediction(
            pred_raw=row.get('prediction', ''),
            answer_raw=row.get('answer', ''),
            l2_category=row.get('l2-category', ''),
            data_point=row.to_dict(),
        )
        gt_list.append(scored['gt_norm'])
        pred_list.append(scored['pred_norm'])
        hit_list.append(int(scored['hit']))

    work_df['gt_norm'] = gt_list
    work_df['pred_norm'] = pred_list
    work_df['hit'] = hit_list
    work_df['get_hit'] = work_df['hit']
    return work_df


def report_acc(df: pd.DataFrame) -> pd.DataFrame:
    """Bridge2 兼容入口：统一走 MMPB 汇总口径。"""
    return report_acc_mmpb(df)
