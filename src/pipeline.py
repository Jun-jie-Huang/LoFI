import os, sys, re, time
import yaml

from utils import write_qa_results_to_csv
from config import config
from log_selection.selection_strategy import select_logs_for_all
from llm_inference.dataset import set_seed, prepare_squad_qa_data
from llm_inference.run_qa import train_eval_qa_model


def main():
    args = config.load_args()
    set_seed(args.seed+1)

    # # Step1: log selection
    if args.log_selection:
        print("[LSSS] Start log selection, method:", args.selection_method)
        args = select_logs_for_all(args, args.selection_method)
        print("[LSSS]", os.path.basename(args.train_filename), os.path.basename(args.dev_filename), os.path.basename(args.test_filename))

    # # Step2: extract fault-indicating information from the selected logs
    if args.do_extraction:
        args = prepare_squad_qa_data(args)
        results = train_eval_qa_model(args)

        write_qa_results_to_csv(results, args)

        with open(os.path.join(args.output_dir, 'args.yaml'), 'w') as f:
            yaml.dump(vars(args), f)


if __name__ == "__main__":
    main()



