import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM
from keras.layers.embeddings import Embedding
# from sklearn.metrics import roc_auc_score,confusion_matrix, accuracy_score, make_scorer, f1_score,precision_score,recall_score, plot_confusion_matrix
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from argparse import Namespace
import tensorflow as tf

# tf.debugging.set_log_device_placement(True)

# from tensorflow.keras.preprocessing.text import OurTokenizer

from  componetes_preprocessamento import RemoveStopWords, Cleaner, Tokenizador, Stemmer, Joiner, Padding, Convert, OurTokenizer
from sklearn.pipeline import Pipeline
nltk.download('rslp')
nltk.download('stopwords')



# import seaborn as sns
# import matplotlib.pyplot as plt
# %matplotlib inline


args = Namespace(
    test_size = 0.3,
    random_state = 199,
    vocab_size = 7000,
    embedding_dim = 8,
    max_length = 5,
    batch_size=32,
    epochs=5,
    early_stopping_criteria=2,
    dropout_p=0.2,
    learning_rate = 0.0001,
    model_storage="model_storage/lstm",
    oov_tok = "<OOV>",
)

dataset = pd.read_csv("datasets/reviews2.csv")
X = dataset["review_comment_message"].copy()
y = dataset["review_score"].copy()
y = np.array(y)
del dataset

for i in range(0,len(y)):
    if y[i] == -1:
        y[i] = 2
X = X[:100]
y = y[:100]
y_dummy = np_utils.to_categorical(y)
print(y_dummy)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state = args.random_state)

def createLSTM(activation, neurons):
    model = Sequential()
    model.add(Embedding(args.vocab_size, args.embedding_dim, input_length=args.max_length))
    model.add(Dropout(0.2))
    model.add(LSTM(units=neurons,activation=activation))
    model.add(Dropout(0.2))
    # model.add(LSTM(units=neurons,activation=activation))
    # model.add(Dropout(0.2))
    # model.add(Dense(units=neurons,activation=activation))
    # model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.fit(X_train_new, y_train, batch_size=128, epochs=5)

    return model

def createLSTM_tf(activation, neurons):
    input = tf.keras.Input(shape=(args.max_length))
    x = tf.keras.layers.Embedding(args.vocab_size, args.embedding_dim, input_length=args.max_length)(input)
    x = tf.keras.layers.Dropout(args.dropout_p)(x)
    x = tf.keras.layers.LSTM(units=neurons,activation=activation, return_sequences =True)(x)
    # x = tf.contrib.rnn.LSTMCell.StandardLSTM(units=neurons,activation=activation, return_sequences =True)(x)
    x = tf.keras.layers.Dropout(args.dropout_p)(x)
    output = tf.keras.layers.Dense(1, activation='softmax')(x)
    model = tf.keras.Model(input, output)
    
    model.compile(
        loss = tf.keras.losses.BinaryCrossentropy(),
        optimizer= tf.keras.optimizers.Adam(
            learning_rate=args.learning_rate),
        metrics=['accuracy']
        )
    return model

parameters = dict()
# parameters["epochs"] = [5,10]
parameters["activation"] = ["relu"]
parameters["neurons"] = [8]
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
model_lstm = KerasClassifier(build_fn=createLSTM_tf, batch_size=args.batch_size, epochs=args.epochs)

pipeline = Pipeline([("Cleaner", Cleaner()), 
                    ("Tokenizador", Tokenizador("portuguese")), 
                    ("RemoveStopWords", RemoveStopWords("portuguese")), 
                    ("Stemmer", Stemmer()), 
                    ("Joiner", Joiner()),
                    ("Count", CountVectorizer()),
                    # ("OurTokenizer", OurTokenizer(args.vocab_size, args.oov_tok)),   
                    ("Padding", Padding(args.max_length, "post")),
                    ("Convert", Convert()),       
                    ("lstm", RandomizedSearchCV(model_lstm, parameters, scoring="f1_macro", cv=kfold, verbose=1, refit=True, n_jobs=-1)),
                    ])

            
pipeline.fit(X_train, y_train)
print("terminou")