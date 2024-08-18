import os, sys, re
import pandas as pd
import ast
from preprocessing.prepro import merge_template_and_param



def clean_chatgpt(string):
    string = ' '.join([i for i in string.split(' ') if i != ""])
    if len(string) > 0:
        string = string[1:] if string[0] in ["\"", "\'"] else string
        string = string[:-1] if string[-1] in ["\"", "\'"] else string
    return string


def post_split_description_and_param(summary_string):
    desc_string = summary_string.split('\n')[0]
    param_string = summary_string.split('\n')[1] if len(summary_string.split('\n')) != 1 else ""  # in case no '\n'
    description = ':'.join(desc_string.split(":")[1:]).strip()  # if no ":", just output ""
    parameter = ':'.join(param_string.split(":")[1:]).strip()
    return description, parameter


def post_process_summary(data_json):
    desc_ref, desc_hypo, param_ref, param_hypo = [], [], [], []
    for item in data_json:
        _desc, _param = [], []
        for generation in item['generation']:
            if len(generation) == 0:
                continue
            d1, p1 = post_split_description_and_param(generation.lower())
            _desc.append(clean_chatgpt(d1))
            _param.append(clean_chatgpt(p1))
        desc_hypo.append(_desc)
        param_hypo.append(_param)
        d2 = item['summary'].split(';')[0].lower()
        p2 = item['summary'].split(';')[1].lower() if ";" in item['summary'] else ""
        desc_ref.append(clean_chatgpt(d2))
        param_ref.append(clean_chatgpt(p2))
    return desc_ref, desc_hypo, param_ref, param_hypo




