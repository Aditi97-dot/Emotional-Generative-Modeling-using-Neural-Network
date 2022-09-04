
import numpy as np
import pandas as pd
import nltk
#nltk.download('punkt')
from nltk.tokenize import word_tokenize
#!pip install emoji
import emoji as emoji
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from keras.layers import *
from keras.models import Sequential



def run():
  train_X = pd.read_csv('emojify_train_x.csv',header=None)
  test_X = pd.read_csv('emojiy_test_x.csv',header=None)

  train_Y = pd.read_csv('Emojify_Y_train.csv',header=None)

  test_Y = pd.read_csv('emojiy_y_test.csv',header=None)

  frames = [train_X , test_X]
  X = pd.concat(frames)
  frames = [train_Y , test_Y]
  Y = pd.concat(frames)
  #X.shape
  #Y.shape

  import re as s
  def clean(train):
    List = train.iloc[:,0].tolist()
    Regex = str.maketrans("","","'")
    word = [s.translate(Regex) for s in List]
    tokenized_sent = []
    for s in word:
      tokenized_sent.append(word_tokenize(s.lower())) 
    return word , tokenized_sent

  trained_list , trained_and_tokened_list = clean(X)

  emoji_dictionary = {"0": "\u2764\uFE0F",    # :heart: prints a black instead of red heart depending on the font
                      "1": ":baseball:",
                      "2": ":beaming_face_with_smiling_eyes:",
                      "3": ":downcast_face_with_sweat:",
                      "4": ":fork_and_knife:",
                    }


  module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 
  model_e = hub.load(module_url)
  #print ("module %s loaded" % module_url)

  sentence_embeddings = model_e(trained_list)
  category = Y.iloc[:,0].tolist()
  category_ohe = pd.get_dummies(category)

  X_train, X_test, y_train, y_test = train_test_split(sentence_embeddings.numpy(), category_ohe, test_size=0.33, random_state=42)

  model = Sequential()
  model.add(Dense(64, input_dim=512, activation='relu'))
  model.add(Dense(128, activation='relu'))
  model.add(Dense(5, activation='sigmoid'))
  model.compile(
      optimizer='adam',
      loss='categorical_crossentropy',
      metrics=['acc'])
  model.fit(X_train,y_train,epochs=40,shuffle=True,validation_split=0.1)


  def getEmoji(sentence):
    query = []
    query.append(sentence)
    inp = model_e(query)
    out =  model.predict_classes(inp)
    print('Statement is :  '+query[0])
    print('Classification Result is :  '+emoji.emojize(emoji_dictionary[str(out[0])]))

#emoji.EMOJI_ALIAS_UNICODE
#model.evaluate(X_test,y_test)
# for e in emoji_dictionary.values():
#     print(emoji.emojize(e))

run()

sentence = input()
getEmoji(sentence)
