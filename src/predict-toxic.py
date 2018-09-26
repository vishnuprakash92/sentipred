# -*- coding: utf-8 -*-

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import model_from_json
from keras.preprocessing import sequence
import pickle,os

maxlen = 100


with open('./data/model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

json_file = open('./data/model/toxic_predcit_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json) 
file_path="./data/model/weights_base.best.hdf5"
loaded_model.load_weights(file_path)

test = pd.read_csv("./data/input/test.csv")
list_sentences_test = test["comment_text"].fillna("RT-Rakesh").values
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)


y_test = loaded_model.predict(X_te)

sample_submission = pd.read_csv("./data/input/sample_submission.csv")
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
sample_submission[list_classes] = y_test

sample_submission.to_csv("./output/baseline.csv", index=False)



