import argparse
import logging
import os
import random
import glob
from finetuned_squad.examples.run_squad import evaluate, set_seed, to_list, load_and_cache_examples

from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForQuestionAnswering, BertTokenizer)
#                                  XLMConfig, XLMForQuestionAnswering,
#                                  XLMTokenizer, XLNetConfig,
#                                  XLNetForQuestionAnswering,
#                                  XLNetTokenizer)

def set_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--predict_file", default="/home/gene/qademo/tmp/eval.json", type=str)
    parser.add_argument("--model_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="/home/gene/clouddrive", type=str)
    parser.add_argument("--output_dir", default="/home/gene/clouddrive", type=str)
    parser.add_argument("--cache_dir", default="", type=str)
    parser.add_argument('--version_2_with_negative', action='store_true')
    parser.add_argument('--null_score_diff_threshold', type=float, default=0.0)

    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--doc_stride", default=128, type=int)
    parser.add_argument("--max_query_length", default=64, type=int)
    parser.add_argument("--do_eval", action='store_true', default=True)
    parser.add_argument("--do_train", action='store_true', default=False)
    parser.add_argument("--do_lower_case", action='store_true', default=True)

    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int)
    parser.add_argument("--n_best_size", default=5, type=int)
    parser.add_argument("--max_answer_length", default=30, type=int)
    parser.add_argument("--verbose_logging", action='store_true')
    parser.add_argument("--no_cuda", action='store_true', default=True)
    parser.add_argument('--overwrite_output_dir', action='store_true')
    parser.add_argument('--overwrite_cache', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument("--local_rank", type=int, default=-1,)
    
    return parser.parse_args()

args = set_args()
print(args)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) \
                  for conf in (BertConfig, XLNetConfig, XLMConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForQuestionAnswering, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
}

args.device = "cpu"
set_seed(args)

args.model_type = args.model_type.lower()
config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
config = config_class.from_pretrained(args.model_name_or_path)
tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
#model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

model = model_class.from_pretrained(args.output_dir)#checkpoint)
model.to(args.device)

evaluate(args, model, tokenizer, prefix="")
