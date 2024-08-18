import os
import random

from utils import read_json, write_json
from preprocessing.prepro import merge_template_and_param
from log_selection.textual_similarity import find_most_similar_embedding_Nd


def check_summary_in(log, summary):
    pieces = summary.split(';')
    pieces = [i.strip() for i in pieces]
    if all([piece in log for piece in pieces]):
        return True
    else:
        return False


def select_logs_for_all(args, selection_method):

    data_files = []
    if args.train_filename is not None:
        args.train_filename += "_sel"+selection_method
        if not os.path.exists(args.train_filename):
            data_files.append(args.train_filename)
    if args.dev_filename is not None:
        args.dev_filename += "_sel"+selection_method
        if not os.path.exists(args.dev_filename):
            data_files.append(args.dev_filename)
    if args.test_filename is not None:
        args.test_filename += "_sel"+selection_method
        if not os.path.exists(args.test_filename):
            data_files.append(args.test_filename)
    print(f"DATAFILES for log selection: {data_files}")

    LEN_BEFORE, LEN_AFTER, FILES, SUM_ALL, SUM_ERROR, IDXS = [], [], [], [], [], []
    for data_file in data_files:
        data = read_json(data_file[:-(4+len(selection_method))])
        length_before, length_after = [], []
        summary_in_logs_all, summary_in_logs_error = [], []
        new_data = []
        new_data_idx = []
        for item in data:
            new_item = item.copy()

            # log_contents = [merge_template_and_param(temp, param) for temp, param in zip(item['templates'], item['parameters'])]
            log_contents = item['raw_log']
            new_logs_idx = select_logs(log_contents, item['levels'], method=selection_method)

            new_item['raw_log'] = [item['raw_log'][idx] for idx in new_logs_idx]
            # new_item['templates'] = [item['templates'][idx] for idx in new_logs_idx]
            # new_item['parameters'] = [item['parameters'][idx] for idx in new_logs_idx]
            new_item['levels'] = [item['levels'][idx] for idx in new_logs_idx]
            new_data.append(new_item)
            new_data_idx.append(new_logs_idx)

            summary_in_logs_all.append(check_summary_in('\n'.join(item['raw_log']), item['summary']))
            summary_in_logs_error.append(check_summary_in('\n'.join(new_item['raw_log']), item['summary']))
            length_before.append(len(item['raw_log']))
            length_after.append(len(new_item['raw_log']))
        write_json(new_data, data_file)
        LEN_BEFORE.append(length_before)
        LEN_AFTER.append(length_after)
        FILES.append(data_file)
        SUM_ALL.append(summary_in_logs_all)
        SUM_ERROR.append(summary_in_logs_error)
        IDXS.append(new_data_idx)

    for length_before, length_after, data_file, summary_in_logs_all, summary_in_logs_error, new_data_idx in zip(LEN_BEFORE, LEN_AFTER, FILES, SUM_ALL, SUM_ERROR, IDXS):
        print(data_file)
        print(f"{sum(length_before)} -> {sum(length_after)}, {sum(summary_in_logs_all)} -> {sum(summary_in_logs_error)}")
        print(f"{[[i, new_data_idx[i]] for i, flag in enumerate(summary_in_logs_error) if not flag]}")
        print()

    return args


def find_highest_level(levels):
    """ Log level order: FATAL > ERROR > WARN > INFO > DEBUG > TRACE,
        Given a list of level string, return the highest level in the list
    """
    level_order = ['FATAL', 'ERROR', 'WARN', 'INFO', 'DEBUG', 'TRACE']
    level_idx = [level_order.index(level) if level in level_order else 5 for level in levels]
    return level_order[min(level_idx)]


def select_logs(logs, levels=None, times=None, method='Highest'):

    if method == 'Error' and levels is not None:
        new_logs_idx = [idx for idx, level in enumerate(levels) if level == 'ERROR']
    elif method == 'Highest' and levels is not None:
        highest_level = find_highest_level(levels)
        new_logs_idx = [idx for idx, level in enumerate(levels) if level == highest_level]
    elif method == 'Time' and levels is not None and times is not None:
        # TODO
        new_logs_idx = list(range(len(logs)))
    elif method == 'HighGroup' and levels is not None:
        highest_level = find_highest_level(levels)
        new_logs_idx = [idx for idx, level in enumerate(levels) if level == highest_level]
        wanted_idx_group = [max(0, i-1) for i in new_logs_idx]
        new_logs_idx = sorted(set(new_logs_idx + wanted_idx_group))
    elif 'HighSimilar' in method and levels is not None:
        plm = method.split('-')[1]
        highest_level = find_highest_level(levels)
        new_logs_idx = [idx for idx, level in enumerate(levels) if level == highest_level]
        sorted_idx, similarity = find_most_similar_embedding_Nd(logs, new_logs_idx, plm, return_sim=True)
        # sorted_idx = find_most_similar_embedding_Nd(logs, new_logs_idx, return_sim=False)
        wanted_idx_sim = [i for i in sorted_idx if i>0.9][:1]
        new_logs_idx = sorted(set(new_logs_idx + wanted_idx_sim))
    elif 'HighGroupSimilar' in method and levels is not None:
        plm = method.split('-')[1]
        highest_level = find_highest_level(levels)
        new_logs_idx = [idx for idx, level in enumerate(levels) if level == highest_level]
        sorted_idx, similarity = find_most_similar_embedding_Nd(logs, new_logs_idx, plm, return_sim=True)
        # sorted_idx = find_most_similar_embedding_Nd(logs, new_logs_idx, return_sim=False)
        wanted_idx_sim = [i for i in sorted_idx if i>0.9][:1]
        wanted_idx_group = [max(0, i-1) for i in new_logs_idx]
        new_logs_idx = sorted(set(new_logs_idx + wanted_idx_sim + wanted_idx_group))
    elif 'random' in method:
        new_logs_idx = sorted(random.sample(list(range(len(logs))), int(len(logs)*0.2)))
    else:
        new_logs_idx = list(range(len(logs)))
    return new_logs_idx


