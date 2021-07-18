import os
from flask import Flask, render_template
from flask import request

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2",)

model = AutoModelForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")

def answer_question(question, answer_text):
    '''
    Takes a `question` string and an `answer` string and tries to identify 
    the words within the `answer` that can answer the question. Prints them out.
    '''
    token_indices = tokenizer.encode(question, answer_text)
    sep_index = token_indices.index(tokenizer.sep_token_id)
    
    seg_one = sep_index + 1
    seg_two = len(token_indices) - seg_one
    segment_ids = [0]*seg_one + [1]*seg_two
    
    start_scores, end_scores = model(torch.tensor([token_indices]), token_type_ids=torch.tensor([segment_ids]))
    answer_begin = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)
    
    indices_tokens = tokenizer.convert_ids_to_tokens(token_indices)
    
    answer = indices_tokens[answer_begin:answer_end+1]
    answer = [word.replace("▁","") if word.startswith("▁") else word for word in answer]
    answer = " ".join(answer).replace("[CLS]","").replace("[SEP]","").replace(" ##","")
    
    return answer


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
  
    if request.method == 'POST':
      form = request.form
      result = []
      bert_abstract = form['paragraph']
      question = form['question']
      result.append(form['question'])
      result.append(answer_question(question, bert_abstract))
      result.append(form['paragraph'])

      return render_template("index.html",result = result)

    # answer = answer_question(question, bert_abstract)

    # return answer
    return render_template("index.html")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
