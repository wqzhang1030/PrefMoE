#作为评测接口层引入，桥接打分load/dump/get_hit/report_acc
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
import math
import csv
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava import conversation as conversation_lib

from llava.mm_utils import tokenizer_image_token

local_rank = None

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


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]



class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)):
            return None
        return json.JSONEncoder.default(self, obj)

def dump(data, f, **kwargs):
    def dump_pkl(data, pth, **kwargs):
        pickle.dump(data, open(pth, 'wb'))

    def dump_json(data, pth, **kwargs):
        json.dump(data, open(pth, 'w'), indent=4, ensure_ascii=False, cls=NumpyEncoder)

    def dump_jsonl(data, f, **kwargs):
        lines = [json.dumps(x, ensure_ascii=False, cls=NumpyEncoder) for x in data]
        with open(f, 'w', encoding='utf8') as fout:
            fout.write('\n'.join(lines))

    def dump_xlsx(data, f, **kwargs):
        data.to_excel(f, index=False, engine='xlsxwriter')

    def dump_csv(data, f, quoting=csv.QUOTE_ALL):
        data.to_csv(f, index=False, encoding='utf-8', quoting=quoting)

    def dump_tsv(data, f, quoting=csv.QUOTE_ALL):
        data.to_csv(f, sep='\t', index=False, encoding='utf-8', quoting=quoting)

    handlers = dict(pkl=dump_pkl, json=dump_json, jsonl=dump_jsonl, xlsx=dump_xlsx, csv=dump_csv, tsv=dump_tsv)
    suffix = f.split('.')[-1]
    return handlers[suffix](data, f, **kwargs)


def get_hit(df: pd.DataFrame) -> pd.DataFrame:
    """
    为样本表增加 `gt_norm/pred_norm/hit` 三列。
    评分细节统一复用 eval_demo.metrics，避免两套规则漂移。
    """
    from llava.eval.prefmoe.eval_demo.metrics import score_prediction

    out = df.copy()
    if len(out) == 0:
        if 'gt_norm' not in out.columns:
            out['gt_norm'] = []
        if 'pred_norm' not in out.columns:
            out['pred_norm'] = []
        if 'hit' not in out.columns:
            out['hit'] = []
        return out

    gt_norm_list = []
    pred_norm_list = []
    hit_list = []

    for _, row in out.iterrows():
        row_dict = row.to_dict()
        pred_raw = row_dict.get('prediction', '')
        ans_raw = row_dict.get('answer', '')
        l2_cat = row_dict.get('l2-category', '')
        score = score_prediction(pred_raw, ans_raw, l2_cat, row_dict)
        gt_norm_list.append(score['gt_norm'])
        pred_norm_list.append(score['pred_norm'])
        hit_list.append(score['hit'])

    out['gt_norm'] = gt_norm_list
    out['pred_norm'] = pred_norm_list
    out['hit'] = hit_list
    return out


def report_acc(df: pd.DataFrame) -> pd.DataFrame:
    """
    对齐 MMPB report_acc 口径（split/Overall/分组列）。
    """
    from llava.eval.prefmoe.eval_demo.metrics import report_acc_mmpb

    return report_acc_mmpb(df)
