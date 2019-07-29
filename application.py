from flask import Flask, render_template, url_for, redirect
#from forms import UploadForm
from flask import request
from werkzeug import secure_filename
import json
import re
import os

app = Flask(__name__)
app.debug=True
WDIR = dir_path = os.path.dirname(os.path.realpath(__file__))
SECRET_KEY = os.urandom(32)
TMP_DIR   = WDIR + '/tmp/'
SQUAD_DIR = WDIR + '/finetuned_squad'
MODEL_DIR = WDIR + '/models'
ALLOWED_EXTENSIONS = set(['txt'])

app.config['SECRET_KEY'] = SECRET_KEY
app.config['TMP_DIR'] = TMP_DIR
app.config['SQUAD_DIR'] = SQUAD_DIR
app.config['MODEL_DIR'] = MODEL_DIR

#@app.route("/")
#@app.route("/home")
#def home():
#    return render_template('home.html', posts=posts)


#@app.route("/about")
#def about():
#    return render_template('about.html', title='About')

#@app.route('/upload', methods=['GET', 'POST'])
#def index():
#    form = UploadForm()
#    if request.method == 'POST' and form.validate_on_submit():
#        input_file = request.files['input_file']
#        with open("yourfile", "r") as f:
#            content = f.read()
#            
#        return render_template("upload.html", content=content)
#        # Do stuff
#    else:
#        return render_template('upload.html', form=form)
    
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route("/")
@app.route("/home", methods=['GET', 'POST'])
#@app.route("/get_file", methods=['GET', 'POST'])
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

            os.system("cd "+app.config['SQUAD_DIR']+"; ./pred_squad.sh")
            
            with open(app.config['MODEL_DIR']+"/nbest_predictions_.json", 'r') as g:
                ans = json.load(g)
            
            a0 = ans['1'][0]['text']
            a1 = ans['1'][1]['text']
            p0 = ans['1'][0]['probability']
            p1 = ans['1'][1]['probability']

            if a0 == '':
                answer = "I'm " + str(round(p0*100, 2)) + \
                        "% certain the answer cannot be found, but " + \
                        "this might be a relevant to your query:\n" + a1
            else: 
                answer = a0 + " ..................with " + str(round(p0*100,2)) + "% certainty."

            return render_template('home.html', 
                                    content=fc, 
                                    question=answer)

    return render_template('home.html', content='Select file and click Upload')


if __name__ == '__main__':
    app.run(debug=True)
