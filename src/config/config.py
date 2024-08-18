import argparse
import os
import sys
import yaml


def common_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default=0, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--model_specification", type=str, default="", help="what do you want to say about this model")
    parser.add_argument("--prefix", type=str, default="")

    parser.add_argument("--save_result_dir", type=str, default="../saved_results")
    parser.add_argument("--training_data_dir", type=str, default="../data/Apache", help="The directory where the training and testing data are saved.")
    parser.add_argument("--output_dir", default=None, type=str, help="The output directory where the model predictions and checkpoints will be written.")

    # # log summary overall args
    # # Step2: Log extraction
    # # Step2.1: extraction args: log selection
    parser.add_argument("--log_selection", action='store_true', default=False, help="Whether to select log to feed to qa model.")
    parser.add_argument("--selection_method", default='HighSimilar-unixcoder', type=str, help=".", choices=['Error', 'Highest', 'Time', 'HighGroup', 'random',
                                                                                                          'HighSimilar-bert', 'HighSimilar-roberta', 'HighSimilar-codebert', 'HighSimilar-unixcoder', 'HighSimilar-labse',
                                                                                                          'HighGroupSimilar-bert', 'HighGroupSimilar-roberta', 'HighGroupSimilar-codebert', 'HighGroupSimilar-unixcoder', 'HighGroupSimilar-labse',])
    # # Step2.2: extraction args: prepare squad data
    # parser.add_argument("--qa_train_with_log_selection", type=str, default="", help="Train qa model with log_selection", choices=['Train', 'N'])
    # parser.add_argument("--qa_test_with_log_selection", type=str, default="", help="Test qa model with log_selection", choices=['Test', 'N'])
    parser.add_argument("--qa_add_desc", action='store_true', default=True, help="Whether add qa pairs of description to qa model.")
    parser.add_argument("--qa_add_param", action='store_true', default=True, help="Whether add qa pairs of parameters to qa model.")
    parser.add_argument("--overwrite_squad_dir", action='store_true', default=True, help="Whether to overwrite split of the dataset.")

    # # Step2.3: extraction args
    parser.add_argument("--extraction_method", type=str, default="fewshot_qa", help="extraction method", choices=["fewshot_qa", "no"])
    parser.add_argument("--do_extraction", action='store_true', default=False, help="Whether to select log to feed to qa model.")

    return parser



def add_log_parser_args(parser):
    parser.add_argument("--log_step_size", default=11, type=int)
    parser.add_argument("--log_second_window", default=11, type=int)

    return parser


def add_model_args(parser):
    parser.add_argument("--model_name", type=str, default="unixcoder")
    # parser.add_argument("--model_name", type=str, default="robertaPreWin10Step10")
    parser.add_argument('--model_name_or_path', type=str, default='./pretrained_models/unixcoder-base', help='model path')
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")

    parser.add_argument("--train_filename", default=None, type=str, help="The train filename.")
    parser.add_argument("--dev_filename", default=None, type=str, help="The dev filename.")
    parser.add_argument("--test_filename", default=None, type=str, help="The test filename.")

    parser.add_argument("--max_source_length", default=512, type=int, help="Maximum number of source tokens.")
    parser.add_argument("--stride", default=100, type=int, help="Stride to split source tokens.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    # parser.add_argument("--max_source_length", default=64, type=int,
    #                     help="The maximum total source sequence length after tokenization. Sequences longer "
    #                          "than this will be truncated, sequences shorter will be padded.")
    # parser.add_argument("--max_target_length", default=32, type=int,
    #                     help="The maximum total target sequence length after tokenization. Sequences longer "
    #                          "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--overwrite_output_dir", action='store_true', default=True,
                        help="Whether to overwrite saved model and re training.")
    parser.add_argument("--do_train", action='store_true', default=True,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', default=True,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true', default=True,
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument("--train_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    # parser.add_argument("--beam_size", default=10, type=int,
    #                     help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    # parser.add_argument("--max_grad_norm", default=1.0, type=float,
    #                     help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=200, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_step", default=10, type=int,
                        help="Logging steps.")
    parser.add_argument("--eval_steps", default=30, type=int,
                        help="Evaluation steps.")
    # parser.add_argument("--save_steps", default=30, type=int,
    #                     help="Save model steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",)
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Whether to evaluate during training",)
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--n_best_size", default=20, type=int, help="The total number of n-best predictions to generate when looking for an answer.")
    parser.add_argument("--max_answer_length", default=50, type=int, help="The maximum length of an answer that can be generated. This is needed because the start and end predictions are not conditioned on one another.")
    parser.add_argument("--version_2_with_negative", action='store_true', default=True, help="If true, some of the examples do not have an answer.")
    parser.add_argument("--null_score_diff_threshold", default=0.0, type=float, help="The threshold used to select the null answer: if the best answer has a score that is less than the score of the null answer minus this threshold, the null answer is selected for this example. Only useful when `version_2_with_negative=True")
    parser.add_argument("--update_train_examples_prediction", action='store_true', default=True, help="If true, do prediction and save results on training set.")

    parser.add_argument("--max_train_samples", default=None, type=int, help="max number of examples for training.")
    parser.add_argument("--max_eval_samples", default=None, type=int, help="max number of examples for eval.")
    parser.add_argument("--max_test_samples", default=None, type=int, help="max number of examples for test.")

    return parser


def load_args():
    parser = common_args()
    parser = add_log_parser_args(parser)
    parser = add_model_args(parser)
    args = parser.parse_args()

    print(f"[config][Offline] Do Extraction")
    # if args.do_extraction:
    args.train_filename = os.path.join(args.training_data_dir, 'train.json' if not args.train_filename else args.train_filename)
    args.dev_filename = os.path.join(args.training_data_dir, 'dev.json' if not args.dev_filename else args.dev_filename)
    args.test_filename = os.path.join(args.training_data_dir, 'test.json' if not args.test_filename else args.test_filename)
    print(f"[config][Extraction] train/test files saved at {args.training_data_dir}")

    args.prefix += f"qa_model/{args.model_name}-id{args.run_id}-Second{args.log_second_window}Step{args.log_step_size}"
    if args.log_selection:
        args.prefix += f"-LS{args.selection_method}"
    args.output_dir = os.path.join(args.save_result_dir, args.prefix) if args.output_dir is None else args.output_dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print(f"[config][Extraction] Model saved to {args.output_dir}")
    with open(os.path.join(args.output_dir, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f)

    return args

