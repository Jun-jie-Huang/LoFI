import os
import sys


def read_log_from_file(file_name):
    with open(file_name, 'r', encoding='utf-8') as fp:
        lines = fp.read().split("\n")
    return lines

import json
def write_json(data, path):
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, indent=1)
def read_json(path):
    with open(path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    return data


import pickle
def write_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def read_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


import csv
def csv_reader(path):
    with open(path, 'r', encoding='utf-8') as fp:
        reader = csv.reader(fp)
        data = [i for i in reader]
    return data
def csv_writer(path, header, data):
    with open(path, 'w', encoding='utf-8_sig', newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(header)
        writer.writerows(data)


# import fcntl
# import csv
# def write_experiment_results(outfile, result_dict):
#     # Write the results to a CSV file
#     # row_to_save = vars(args)  ## add args
#     # row_to_save.update(result)
#
#     with open(outfile, "a", newline="") as csvfile:
#         # Acquire a file write lock
#         fcntl.flock(csvfile.fileno(), fcntl.LOCK_EX)
#
#         fieldnames = list(result_dict.keys())
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#
#         # If this is the first row, write the header
#         if csvfile.tell() == 0:
#             writer.writeheader()
#
#         # Write the row for this experiment
#         writer.writerow(result_dict)
#         print(f"Experiment record saved to {outfile}.")
#
#         # Release the file write lock
#         fcntl.flock(csvfile.fileno(), fcntl.LOCK_UN)

import csv
import sys
if sys.platform == "win32":
    import msvcrt
else:
    import fcntl
def write_experiment_results(outfile, result_dict):
    # Write the results to a CSV file
    # row_to_save = vars(args)  ## add args
    # row_to_save.update(result)

    with open(outfile, "a", newline="") as csvfile:
        if sys.platform == "win32":
            # Acquire a file write lock on Windows
            msvcrt.locking(csvfile.fileno(), msvcrt.LK_LOCK, 1)
        else:
            # Acquire a file write lock on Linux
            fcntl.flock(csvfile.fileno(), fcntl.LOCK_EX)

        fieldnames = list(result_dict.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # If this is the first row, write the header
        if csvfile.tell() == 0:
            writer.writeheader()

        # Write the row for this experiment
        writer.writerow(result_dict)
        print(f"Experiment record saved to {outfile}.")

        if sys.platform == "win32":
            # Release the file write lock on Windows
            msvcrt.locking(csvfile.fileno(), msvcrt.LK_UNLOCK, 1)
        else:
            # Release the file write lock on Linux
            fcntl.flock(csvfile.fileno(), fcntl.LOCK_UN)



import time
def write_qa_results_to_csv(results, args):
    result_dict = {
        'RunID': args.run_id,
        'Second': args.log_second_window,
        'Model': args.model_name,
        'ModelInitial': args.model_name_or_path,
        'WithLogSelection': args.log_selection,
        'LS_METHOD': args.selection_method,
        'Epo': args.num_train_epochs,
        'Bz': args.train_batch_size,
        'LR': args.learning_rate,
        'Seed': args.seed,
        'FinishTime': time.asctime(time.localtime(time.time())),
        'Desc-HasAns': '',
        'Num-Desc-HasAns': int(results.get('test_DESC_HasAns_total', 0)),
        'EM-Desc-HasAns': round(results.get('test_DESC_HasAns_exact', -1), 2),
        'F1-Desc-HasAns': round(results.get('test_DESC_HasAns_f1', -1), 2),
        'Param-HasAns': '',
        'Num-Param-HasAns': int(results.get('test_PARAM_HasAns_total', 0)),
        'EM-Param-HasAns': round(results.get('test_PARAM_HasAns_exact', -1), 2),
        'F1-Param-HasAns': round(results.get('test_PARAM_HasAns_f1', -1), 2),
        'Param-NoAns': '',
        'Num-Param-NoAns': int(results.get('test_PARAM_NoAns_total', 0)),
        'EM-Param-NoAns': round(results.get('test_PARAM_NoAns_exact', -1), 2),
        'F1-Param-NoAns': round(results.get('test_PARAM_NoAns_f1', -1), 2),
        'OutPath': args.output_dir,
    }
    append_csv_path = os.path.join(args.save_result_dir, 'summary_results_fewshot_qa_new.csv')
    write_experiment_results(append_csv_path, result_dict)



