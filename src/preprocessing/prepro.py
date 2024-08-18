# -*- coding: utf-8 -*-
import os, sys, re
import pandas as pd
import ast
import sys
# sys.path.append("..")
from utils import read_json, write_json


def merge_template_and_param(temp, params):
    if isinstance(params, str):
        params = ast.literal_eval(params)
    sep = "<*>"
    temp0 = temp
    for param in params:
        temp = temp.replace(sep, param, 1)
    return temp


def split_summary(summary):
    splits = summary.split(';')
    if len(splits) >= 2:
        return splits[0], splits[1]
    else:
        return splits[0], ''


def clean_logs(logs):
    return ' '.join([i for i in logs.split(' ') if len(i)>0])


def data_df_json_to_summary(data_json):
    # data_json = read_json(os.path.join(args.save_result_dir, "data.json"))
    data = []
    for item in data_json:
        # templates = item['templates']
        # # params = [ast.literal_eval(i) for i in item['parameters']]
        # params = item['parameters']
        #
        # ### Option 1: merge all logs
        # merged_logs = "\n".join([merge_template_and_param(t, p) for t, p in zip(templates, params)])
        # ### Option 2: merge logs and down sample, for chatgpt inference
        # # merged_logs = [merge_template_and_param(t, p) for t, p in zip(templates, params)]
        # # merged_logs = down_sample_logs(merged_logs, item['summary'])
        # # merged_logs = "\n".join(merged_logs)

        merged_logs = '\n'.join(item['raw_log'])

        if 'summary' in item:
            # offline setting
            description, param_string = split_summary(item['summary'])
        else:
            # online setting
            description, param_string = '', ''

        data.append({"logs": merged_logs,
                     "summary": item['summary'],
                     "description": description,
                     "param_string": param_string,
                     "raw_log": item['raw_log']})
    return data


import random
def down_sample_logs(logs, summary):
    description, param_string = split_summary(summary)
    if param_string != "":
        needed = [i for i, log in enumerate(logs) if param_string in log and description in log]
    else:
        needed = [i for i, log in enumerate(logs) if description in log]
    other_indexes = [i for i in range(len(logs)) if i not in needed]
    threshold = 40
    if len(needed) < threshold:
        appened = random.sample(other_indexes, min(threshold-len(needed), len(other_indexes)))
        needed += appened
    needed.sort()
    return [logs[i] for i in needed]

