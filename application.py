from flask import Flask, render_template, url_for, redirect
#from forms import UploadForm
from flask import request
from werkzeug import secure_filename
import json
import re
import os
import argparse
import random
import glob
from finetuned_squad.examples.run_squad import evaluate, set_seed, to_list, load_and_cache_examples

from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForQuestionAnswering, BertTokenizer,
                                  XLMConfig, XLMForQuestionAnswering,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForQuestionAnswering,
                                  XLNetTokenizer)


app = Flask(__name__)
app.debug=True
WDIR = os.path.dirname(os.path.realpath(__file__))
SECRET_KEY = os.urandom(32)
TMP_DIR   = WDIR + '/tmp/'
SQUAD_DIR = WDIR + '/finetuned_squad'
PAR_DIR = os.path.dirname(WDIR)
MODEL_DIR = PAR_DIR + '/clouddrive'
ALLOWED_EXTENSIONS = set(['txt'])

app.config['SECRET_KEY'] = SECRET_KEY
app.config['TMP_DIR'] = TMP_DIR
app.config['SQUAD_DIR'] = SQUAD_DIR
app.config['MODEL_DIR'] = MODEL_DIR

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

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

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) \
                  for conf in (BertConfig, XLNetConfig, XLMConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForQuestionAnswering, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
}

args.device = "cpu"
set_seed(args)


@app.route("/")
@app.route("/home", methods=['GET', 'POST'])

def home():
    if request.method == 'POST':
        txt_file = request.files['file']
        txt_query = request.form['question']
        print(txt_query)
        if txt_file and allowed_file(txt_file.filename):
            filename = secure_filename(txt_file.filename)
            full_filename = os.path.join(app.config['TMP_DIR'], filename)
            txt_file.save(full_filename)
            
            #return redirect(url_for('index'))
            with open(full_filename, 'r') as f:
                text = f.read()
                text = re.sub("\[\d+\]", "", text)
                text = re.sub("\n", " ", text)

            return render_template('home.html', content=text)

        if txt_query:
            fc=request.form['context']
            fq=request.form['question']
            js  = """{"data": [{"paragraphs": [{ "qas": [{ "question":"", "id": ""}]}]}]}"""
            jsn = json.loads(js)
            jsn["data"][0]["paragraphs"][0]["qas"][0]["question"]=fq
            jsn["data"][0]["paragraphs"][0]["qas"][0]["id"]="1"
            jsn["data"][0]["paragraphs"][0]["context"]=fc
            
            eval_file = os.path.join(app.config['TMP_DIR'], 'eval.json')
            
            with open(eval_file, 'w') as f:
                json.dump(jsn, f)

            #os.system("cd "+app.config['SQUAD_DIR']+"; ./pred_squad.sh")
            args.model_type = args.model_type.lower()
            config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
            config = config_class.from_pretrained(args.model_name_or_path)
            tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)

            model = model_class.from_pretrained(args.output_dir)#checkpoint)
            model.to(args.device)

            evaluate(args, model, tokenizer, prefix="")

            with open(app.config['MODEL_DIR']+"/nbest_predictions_.json", 'r') as g:
                ans = json.load(g)
            
            a0 = ans['1'][0]['text']
            a1 = ans['1'][1]['text']
            p0 = ans['1'][0]['probability']
            p1 = ans['1'][1]['probability']

            if a0 == '':
                answer = "I'm " + str(round(p0*100, 2)) + \
                        "% certain the answer cannot be found, but " + \
                        "this might be relevant to your query:\n" + a1
            else: 
                answer = a0 + " ..................with " + str(round(p0*100,2)) + "% certainty."

            return render_template('home.html', 
                                    content=fc, 
                                    question=answer)

    return render_template('home.html', content='Select file and click Upload')


if __name__ == '__main__':
    app.run(debug=True)
