import argparse
from transformers import TrainingArguments


def add_model_args(parser=None):
    if not parser:
        parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="roberta")
    parser.add_argument('--model_name_or_path', type=str, default='../pretrained_models/roberta-base', help='model path')
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")

    parser.add_argument("--train_filename", default=None, type=str,
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default=None, type=str,
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default=None, type=str,
                        help="The test filename. Should contain the .jsonl files for this task.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    return parser


def add_qa_args(parser):

    parser.add_argument("--model_name", type=str, default="roberta")
    parser.add_argument('--model_name_or_path', type=str, default='./pretrained_models/roberta-base', help='model path')
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--training_data_dir", default=None, type=str,
                        help="The directory where the training and testing data are saved.")

    # parser.add_argument("--loss_weight", default=5, type=int, help="weights to enhance summaries.")
    parser.add_argument("--max_source_length", default=512, type=int, help="Maximum number of source tokens.")
    parser.add_argument("--stride", default=100, type=int, help="Stride to split source tokens.")
    parser.add_argument("--n_best_size", default=20, type=int, help="The total number of n-best predictions to generate when looking for an answer.")
    parser.add_argument("--max_answer_length", default=50, type=int, help="The maximum length of an answer that can be generated. This is needed because the start and end predictions are not conditioned on one another.")
    parser.add_argument("--version_2_with_negative", action='store_true', default=True, help="If true, some of the examples do not have an answer.")
    parser.add_argument("--null_score_diff_threshold", default=0.0, type=float, help="The threshold used to select the null answer: if the best answer has a score that is less than the score of the null answer minus this threshold, the null answer is selected for this example. Only useful when `version_2_with_negative=True")

    parser.add_argument("--train_filename", default=None, type=str, help="The train filename.")
    parser.add_argument("--dev_filename", default=None, type=str, help="The dev filename.")
    parser.add_argument("--test_filename", default=None, type=str, help="The test filename.")

    parser.add_argument("--max_train_samples", default=None, type=int, help="max number of examples for training.")
    parser.add_argument("--max_eval_samples", default=None, type=int, help="max number of examples for eval.")
    parser.add_argument("--max_test_samples", default=None, type=int, help="max number of examples for test.")

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
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    # parser.add_argument("--beam_size", default=10, type=int,
    #                     help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    # parser.add_argument("--max_grad_norm", default=1.0, type=float,
    #                     help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_step", default=10, type=int,
                        help="Logging steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",)
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Whether to evaluate during training",)
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    return parser


def transfer_qa_training_args(args):
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_step,
        learning_rate=args.learning_rate,
        overwrite_output_dir=args.overwrite_output_dir,
        do_train=args.do_train,
        do_eval=args.do_eval,
        do_predict=args.do_test,
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        save_total_limit=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy='steps',
        logging_strategy='steps',
        # report_to=['tensorboard'],
        seed=args.seed
    )
    return training_args