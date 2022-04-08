import ipywidgets as widgets
from IPython.display import display, clear_output
from transformers import RobertaForSequenceClassification, RobertaConfig
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
from vncorenlp import VnCoreNLP
import pandas as pd
import argparse
import torch
import tensorflow
import re


path_config  = 'config.json'
path_model = 'pytorch_model.bin'
path_bpe = 'bpe.codes'
path_vocab = 'dict.txt'


def get_model(path_model= None, path_config = None, path_bpe = None, path_vocab = None):
  config = RobertaConfig.from_pretrained(
      path_config, from_tf=False, num_labels = 2, output_hidden_states=False,
  )
  BERT_SA_NEW = RobertaForSequenceClassification.from_pretrained(
      path_model,
      config=config
  )
  BERT_SA_NEW.cpu()
  BERT_SA_NEW.eval()


  try:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bpe-codes', 
        default=path_bpe,
        required=False,
        type=str,
        help='path to fastBPE BPE'
    )
    args, unknown = parser.parse_known_args()
    bpe = fastBPE(args)
  except:
    bpe = None
    print("load bpe fail")

  try:
    vocab = Dictionary()
    vocab.add_from_file(path_vocab)
  except:
    vocab=None
    print('load vocab fail')
  return BERT_SA_NEW, bpe, vocab

model, bpe, vocab = get_model(path_model, path_config, path_bpe, path_vocab)


rdrsegmenter = VnCoreNLP('./vncorenlp/VnCoreNLP-1.1.1.jar', annotators="wseg,pos,ner", 
                         max_heap_size='-Xmx2g')


macp = pd.read_excel('Macp.xlsx')
macp = macp.dropna()
tenct = macp['Tên Công ty'].tolist()
for i in range(len(tenct)):
  tenct[i] = str(tenct[i]).lower()
tenct[:5]
tenma = macp['Mã '].tolist()

def del_test(text):
  year = ['năm 2021', 'năm 2020', 'năm 2019', 'năm 2018', 'năm 2017', 'năm 2016', 'năm 2015', 'năm 2014', 'năm 2013', 'năm 2012', 'năm 2011', 'năm 2010', 'Năm 2021', 'Năm 2020', 'Năm 2019', 'Năm 2018', 'Năm 2017', 'Năm 2016', 'Năm 2015', 'Năm 2014', 'Năm 2013', 'Năm 2012', 'Năm 2011', 'Năm 2010', '2021', '2020', '2019', '2018', '2017', '2016', '2015', '2014', '2013', '2012', '2011', '2010']
  month = ['tháng 1', 'tháng 2', 'tháng 3', 'tháng 4', 'tháng 5', 'tháng 6', 'tháng 7', 'tháng 8', 'tháng 9', 'tháng 10', 'tháng 11', 'tháng 12', 'Tháng 1', 'Tháng 2', 'Tháng 3', 'Tháng 4', 'Tháng 5', 'Tháng 6', 'Tháng 7', 'Tháng 8', 'Tháng 9', 'Tháng 10', 'Tháng 11', 'tháng 12']
  quy = ['quý 1', 'quý 2', 'quý 3', 'quý 4', 'Quý 1', 'Quý 2', 'Quý 3', 'Quý 4']
  text = text.replace('Covid-19', 'Covid')
  word_segmented_text = rdrsegmenter.ner(text)[0]
  for char, typ in word_segmented_text:
    if typ == 'B-ORG' or typ == 'I-ORG' or typ == 'B-PER' or typ == 'I-PER':
      char = char.replace('_', ' ')
      text = text.replace(char, 'name')
    if typ == "B-LOC" or typ == "I-LOC":
      if char != 'VN':
        char = char.replace('_', ' ')
        text = text.replace(char,'loc')
    if typ == 'O':
      if len(re.findall('\d*\.?\,?\d+\%', char)) > 0:
        text = text.replace(char, 'percent')
      if len(re.findall('\s?\(?[A-Z]{3,4}\)?\s?', char)) > 0 and char != 'USD':
          text = text.replace(char, 'name')
      if char in tenma:
        text = text.replace(char, 'name')
      char = char.replace('_', ' ')
      char_lower = char.lower()
      if char_lower in tenct:
        text = text.replace(char, 'name')
  text = text.replace('"', '')
  text = text.replace('”', '')
  text = text.replace('“', '')
  text = text.replace('.', '')
  text = text.replace(',', '')
  text = text.replace('(', '')
  text = text.replace(')', '')
  text = text.replace(':', '')
  text = text.replace('[', '')
  text = text.replace(']', '')
  text = text.replace('-', ' ')
  text = re.sub('\d{0,2}-?\d{0,2}\/\d{1,4}', 'date', text)
  for i in quy:
    text = text.replace(i, 'date')
  for i in year:
    text = text.replace(i, 'date')
  for i in month:
    text = text.replace(i, 'date')
  text = re.sub('\d+ năm ', 'date ', text)
  text = re.sub('\d+ tháng ', 'date ', text)
  text = re.sub(' \-?\d+\w?', ' number', text)
  text = text.split()
  for i in range(len(text)):
    if text[i].isdigit():
      text[i] = 'number'
  text = ' '.join(text)
  text1 = text.split()
  for i in range(len(text1)+1):
    try:
      if text1[i][0].isupper() and text1[i+1][0].isupper():
        text = text.replace(text1[i], 'name')
        text = text.replace(text1[i+1], 'name')
    except:
      pass
  text = rdrsegmenter.tokenize(text)
  text = ' '.join([' '.join(x) for x in text])
  text = text.lower()
  return text


def predict_message(model, bpe, sense, vocab):
  subwords = '<s> ' + bpe.encode(sense) + ' </s>'
  encoded_sent = vocab.encode_line(subwords, append_eos=True, add_if_not_exist=False).long().tolist()
  encoded_sent = pad_sequences([encoded_sent], maxlen=100, dtype="long", value=0, truncating="post", padding="post")
  mask = [int(token_id > 0) for token_id in encoded_sent[0]]


  encoded_sent = torch.tensor(encoded_sent).cpu()
  mask = torch.tensor(mask).cpu()
  encoded_sent = torch.reshape(encoded_sent, (1, 100))
  mask = torch.reshape(mask, (1, 100))

  with torch.no_grad():
    outputs = model(encoded_sent, 
      token_type_ids=None, 
      attention_mask=mask)
    logits = outputs[0]
  return int(torch.argmax(logits))


from os.path import dirname, join, realpath
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
import uvicorn

app = FastAPI(title="Spam detection", description="Check spam message", version="1.0")


origins = [
    "http://localhost:9090",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/")
async def predict(data: str):
    prediction = predict_message(model, bpe, del_test(data), vocab)

    return {
        "prediction": prediction
    }


if __name__ == "__main__":
    uvicorn.run("demo:app", host="0.0.0.0", port=5000, log_level="info")
