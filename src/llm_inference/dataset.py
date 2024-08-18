# coding=utf-8

from __future__ import absolute_import

import copy
import os
import torch
import ast
import random
import logging
import numpy as np
from torch.utils.data import TensorDataset

from utils import read_json, write_json
from preprocessing.prepro import merge_template_and_param, split_summary, clean_logs

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class Example(object):
    """A single training/test example."""
    def __init__(self, idx, logs, desc, param):
        self.idx = idx
        self.logs = logs
        self.desc = desc
        self.param = param


def read_examples(filename):
    """Read examples from filename."""
    # sep_token "</s>"; cls_token: "<s>"; unk_token "[UNK]"; pad_token "<pad>"; mask_token:"<mask>"

    examples = []
    data_json = read_json(filename)
    for idx, item in enumerate(data_json):
        # templates = item['templates']
        # params = item['parameters']
        # # params = [ast.literal_eval(i) for i in item['parameters']]
        # merged_logs = "\n".join([merge_template_and_param(t, p) for t, p in zip(templates, params)])
        # # merged_logs = "</s>".join([merge_template_and_param(t, p) for t, p in zip(templates, params)])
        merged_logs = '\n'.join(item['raw_log'])

        merged_logs = clean_logs(merged_logs)
        # description, param_string = split_summary(item['summary'])
        description, param_string = split_summary(item.get('summary_anno', item['summary']))  # TODO update ad then change this line back

        examples.append(
            Example(
                idx=idx,
                logs=merged_logs,
                desc=description,
                param=param_string,
            )
        )

    return examples


class InputFeaturesQA(object):
    def __init__(self,
                 example_id,
                 input_ids,
                 desc_start_position,
                 desc_end_position,
                 param_start_position,
                 param_end_position,
                 is_possible,
    ):
        self.example_id = example_id
        self.input_ids = input_ids
        self.desc_start_position = desc_start_position
        self.desc_end_position = desc_end_position
        self.param_start_position = param_start_position
        self.param_end_position = param_end_position
        self.is_possible = is_possible


class InputFeaturesSeq2Seq(object):
    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids


class InputFeaturesPT(object):
    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 desc_start_ids,
                 desc_end_ids,
                 param_start_ids,
                 param_end_ids,
    ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.desc_start_ids = desc_start_ids
        self.desc_end_ids = desc_end_ids
        self.param_start_ids = param_start_ids
        self.param_end_ids = param_end_ids


def find_all_occurrences_string(string, substr):
    occurrences = []
    start = 0
    while True:
        index = string.find(substr, start)
        if index == -1:
            break
        occurrences.append(index)
        start = index + len(substr)
    return occurrences


def find_start_end_position(long_ids, short_ids):
    """
        To avoid that the first or last token can not match the original logs (due to the possible space or punc token),
        and thus return a null list of start_position and end position, we can cut the original short tokens and match
        the remaining tokens. Each time we cut one token and to find
    """
    offset_short = len(short_ids)
    left_cut, right_cut = 0, 0
    iter_flag = False
    start_positions = kmp_search(long_ids, short_ids)
    if len(start_positions) < 1:
        iter_flag = True
        start_positions = kmp_search(long_ids, short_ids[1:])
        if len(start_positions) > 0:
            left_cut += 1
        else:
            start_positions = kmp_search(long_ids, short_ids[:-1])
            right_cut += 1

    end_positions = [i + offset_short for i in start_positions]
    return [(i-left_cut, j+right_cut) for i, j in zip(start_positions, end_positions)], iter_flag


def truncate_ids(input_ids, max_length=510, stride=100):
    if len(input_ids) <= max_length:
        return [input_ids]
    else:
        output_list = []
        for i in range(0, len(input_ids), max_length-stride):
            input_list_slice = input_ids[i:i + max_length]
            output_list.append(input_list_slice)
        return output_list


def update_start_end_position_after_truncate(positions, max_length=510, stride=100):
    new_positions = []
    for i, position in enumerate(positions):
        start_index, end_index = position
        chunk_start_index = i * stride
        while end_index - start_index > max_length:
            new_positions.append((start_index - chunk_start_index, start_index - chunk_start_index + max_length))
            start_index += stride
        new_positions.append((start_index - chunk_start_index, end_index - chunk_start_index))
    return new_positions


def get_continuous_ranges(input_ids, label_index):
    ranges = []
    i = 0
    n = len(input_ids)
    while i < n:
        if input_ids[i] != label_index:
            i += 1
            continue
        j = i
        while j < n and input_ids[j] == label_index:
            j += 1
        ranges.append((i, j))
        i = j
    return ranges


def get_start_end_position_by_continuous_ranges(input_ids, label_index):
    ranges = get_continuous_ranges(input_ids, label_index)
    if len(ranges) == 1:
        return ranges[0]
    elif len(ranges) == 0:
        return -1, -1
    else:
        return max(ranges, key=lambda x: x[1] - x[0])


def convert_examples_to_features_pt(examples, tokenizer, args):
    features = []
    special_tokens_map = {"i-desc": tokenizer.convert_tokens_to_ids(["i-desc"])[0],
                          "i-param": tokenizer.convert_tokens_to_ids(["i-param"])[0], }
    # _temp1 = []
    for example_index, example in enumerate(examples):
        # if example_index==8:
        #     print()
        # ## tokenize and align labels
        log_tokens = tokenizer.tokenize(example.logs)  # TODO: can be optimized
        log_ids = tokenizer.convert_tokens_to_ids(log_tokens)
        # _temp1.append(len(log_ids))
        desc_tokens = tokenizer.tokenize(example.desc)
        desc_ids = tokenizer.convert_tokens_to_ids(desc_tokens)
        if len(example.param) > 0:
            param_tokens = tokenizer.tokenize(example.param)
            param_ids = tokenizer.convert_tokens_to_ids(param_tokens)

        positions_desc, iter_desc = find_start_end_position(log_ids, desc_ids)
        if iter_desc:
            logger.warning(f"[positions_desc][DESCPTION] iter_desc {iter_desc}")
        if len(positions_desc) == 0:
            logger.warning(f"[positions_desc][LOG] start tokens {tokenizer.convert_tokens_to_string(log_tokens)}")
            logger.warning(f"[positions_desc][DES] start tokens {tokenizer.convert_tokens_to_string(desc_tokens)}")
        if len(example.param) > 0:
            positions_param, iter_param = find_start_end_position(log_ids, param_ids)
            if iter_param:
                logger.warning(f"[positions_param][PARAMETER] iter_param {iter_param}")
            if len(positions_param) == 0:
                logger.warning(f"[positions_param][LOG] start tokens {tokenizer.convert_tokens_to_string(log_tokens)}")
                logger.warning(f"[positions_param][PAR] start tokens {tokenizer.convert_tokens_to_string(param_tokens)}")

        source_ids, target_ids = copy.deepcopy(log_ids), copy.deepcopy(log_ids)
        for s1, e1 in positions_desc:
            target_ids[s1: e1] = tokenizer.convert_tokens_to_ids(["i-desc"]) * (e1 - s1)
        if len(example.param) > 0:
            for s1, e1 in positions_param:
                target_ids[s1: e1] = tokenizer.convert_tokens_to_ids(["i-param"]) * (e1 - s1)
        source_tokens = tokenizer.convert_ids_to_tokens(source_ids)
        target_tokens = tokenizer.convert_ids_to_tokens(target_ids)

        all_source_ids = truncate_ids(source_ids, max_length=args.max_source_length - 2, stride=100)
        all_target_ids = truncate_ids(target_ids, max_length=args.max_source_length - 2, stride=100)
        # # add token
        all_source_ids = [[tokenizer.cls_token_id] + s + [tokenizer.eos_token_id] for s in all_source_ids]
        all_target_ids = [[tokenizer.cls_token_id] + s + [tokenizer.eos_token_id] for s in all_target_ids]
        # # pad
        all_source_ids = [s if len(s)==args.max_source_length else s+[tokenizer.pad_token_id]*(args.max_source_length-len(s)) for s in all_source_ids]
        all_target_ids = [s if len(s)==args.max_source_length else s+[tokenizer.pad_token_id]*(args.max_source_length-len(s)) for s in all_target_ids]

        positions_desc = [get_start_end_position_by_continuous_ranges(s, special_tokens_map['i-desc']) for s in all_target_ids]
        positions_param = [get_start_end_position_by_continuous_ranges(s, special_tokens_map['i-param']) for s in all_target_ids]

        # positions_desc = update_start_end_position_after_truncate(positions_desc)
        # positions_desc = [(i+1, j+1) for i, j in positions_desc]
        # positions_param = update_start_end_position_after_truncate(positions_param)
        # positions_param = [(i+1, j+1) for i, j in positions_param]

        if example_index < 2:
            logger.info("*** Example ***")
            logger.info(f"idx: {example.idx}")
            logger.info("source_tokens: {}".format([x.replace('\u0120', '_') for x in source_tokens]))
            logger.info(f"source_ids: {' '.join(map(str, source_ids))}")
            logger.info("target_tokens: {}".format([x.replace('\u0120', '_') for x in target_tokens]))
            logger.info(f"target_ids: {' '.join(map(str, target_ids))}")

        for idx, (source_ids, target_ids) in enumerate(zip(all_source_ids, all_target_ids)):
            features.append(
                InputFeaturesPT(
                     f"{example_index}-{idx}",
                     source_ids,
                     target_ids,
                     positions_desc[idx][0],
                     positions_desc[idx][1],
                     positions_param[idx][0],
                     positions_param[idx][1],
                )
            )
    # _temp2 = [(item.example_id, item.desc_start_ids, item.desc_end_ids, item.param_start_ids, item.param_end_ids) for item in features]
    return features


def convert_features_to_dataset_pt(features):
    source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
    target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
    desc_start_ids = torch.tensor([f.desc_start_ids for f in features], dtype=torch.int)
    desc_end_ids = torch.tensor([f.desc_end_ids for f in features], dtype=torch.int)
    param_start_ids = torch.tensor([f.param_start_ids for f in features], dtype=torch.int)
    param_end_ids = torch.tensor([f.param_end_ids for f in features], dtype=torch.int)
    tensor_data = TensorDataset(source_ids, target_ids,
                                desc_start_ids, desc_end_ids,
                                param_start_ids, param_end_ids)
    return tensor_data


def prepare_squad_qa_data(args):
    data_files = []
    surfix = "_squad"
    question_desc = "What is the most important/representative description strings in the following logs?"
    question_param = "What is the most important/representative parameters in the following logs?"

    if args.train_filename is not None:
        args.train_filename += surfix
        print(f"Using training data for squad qa: {args.train_filename}")
        if not os.path.exists(args.train_filename):
            data_files.append(args.train_filename)
    if args.dev_filename is not None:
        args.dev_filename += surfix
        print(f"Using dev data for squad qa: {args.dev_filename}")
        if not os.path.exists(args.dev_filename):
            data_files.append(args.dev_filename)
    if args.test_filename is not None:
        args.test_filename += surfix
        print(f"Using test data for squad qa: {args.test_filename}")
        if not os.path.exists(args.test_filename):
            data_files.append(args.test_filename)
    print(f"DATAFILES for squad qa: {data_files}")

    for data_file in data_files:
        examples = read_examples(data_file[:-len(surfix)])
        new_examples = []
        if args.qa_add_desc:
            # question_desc = "What is the most important/representative description strings in the following logs?"
            answer_positions_desc = [find_all_occurrences_string(item.logs, item.desc) if len(item.desc)>0 else [] for item in examples]
            new_examples_d = [{'id': str(item.idx)+'-desc',
                               'question': question_desc,
                               'context': item.logs,
                               'answers': {'text': [item.desc]*len(ans),
                                           'answer_start': ans}} for item, ans in zip(examples, answer_positions_desc)]
            new_examples += new_examples_d
        if args.qa_add_param:
            # question_param = "What is the most important/representative parameters in the following logs?"
            answer_positions_param = [find_all_occurrences_string(item.logs, item.param) if len(item.param)>0 else [] for item in examples]
            if data_file==args.test_filename:
                new_examples_p = [{'id': str(item.idx)+'-param',
                                   'question': question_param,
                                   'context': item.logs,
                                   'answers': {'text': [item.param]*len(ans),
                                               'answer_start': ans}} for item, ans in zip(examples, answer_positions_param)]
            else:
                new_examples_p = [{'id': str(item.idx)+'-param',
                                   'question': question_param.format(f" of '{item.desc}'"),
                                   'context': item.logs,
                                   'answers': {'text': [item.param]*len(ans),
                                               'answer_start': ans}} for item, ans in zip(examples, answer_positions_param)]
            new_examples += new_examples_p
        write_json(new_examples, data_file)
    return args


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_lps_array(sequential_list):
    m = len(sequential_list)
    lps = [0] * m
    i = 1
    j = 0
    while i < m:
        if sequential_list[i] == sequential_list[j]:
            j += 1
            lps[i] = j
            i += 1
        elif j != 0:
            j = lps[j-1]
        else:
            lps[i] = 0
            i += 1
    return lps


def kmp_search(ordered_list, sequential_list):
    n = len(ordered_list)
    m = len(sequential_list)
    lps = compute_lps_array(sequential_list)
    i = 0
    j = 0
    indices = []
    while i < n:
        if ordered_list[i] == sequential_list[j]:
            i += 1
            j += 1
        if j == m:
            indices.append(i-j)
            j = lps[j-1]
        elif i < n and ordered_list[i] != sequential_list[j]:
            if j != 0:
                j = lps[j-1]
            else:
                i += 1
    return indices

