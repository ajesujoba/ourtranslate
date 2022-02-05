import os
from flask import Flask, render_template
from flask import request


import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("Davlan/mt5-small-pcm-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Davlan/mt5-small-pcm-en")

def answer_question(source_text):
    '''
    Takes a `question` string and an `answer` string and tries to identify 
    the words within the `answer` that can answer the question. Prints them out.
    '''

    features = tokenizer([source_text], return_tensors='pt')
    output = model.generate(**features)
    target_text = tokenizer.decode(output[0])
    
    return target_text


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
  
    if request.method == 'POST':
      form = request.form
      result = []
      source_text = form['paragraph']
      #ques = form['question']
      #result.append(form['question'])
      result.append(answer_question(source_text))
      result.append(form['paragraph'])

      return render_template("index.html",result = result)

    return render_template("index.html")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
