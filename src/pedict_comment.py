# -*- coding: utf-8 -*-
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import model_from_json
from keras.preprocessing import sequence
import pickle,os

#print(os.path.dirname(__file__))

maxlen = 100
# =============================================================================
# loading model,weights and tokeniser
# =============================================================================
with open('./data/model/sentiment_analyser_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

json_file = open('./data/model/sentiment_analyser_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json) 
loaded_model.load_weights("./data/model/sentiment_analyser_weights_base.best.hdf5")

# =============================================================================
# 
# =============================================================================
test = input("Enter your comment for comment analysis=")
#list_sentences_test = test["comment_text"].fillna("NA").values
list_tokenized_test = tokenizer.texts_to_sequences([test])
X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)


y_test = loaded_model.predict(X_te)

#print(y_test)
output = {"Comment":test,"positive":y_test[0][0], "negative":y_test[0][1]}
output=pd.DataFrame([output])
##sample_submission.to_csv("./output/baseline.csv", index=False)
print(output)
