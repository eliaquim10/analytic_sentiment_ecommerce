import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold
from argparse import Namespace
import tensorflow as tf


from  componetes_preprocessamento import RemoveStopWords, Cleaner, Tokenizador, Stemmer, Joiner, Padding, Convert, MyModel, pega_resultados
from sklearn.pipeline import Pipeline
nltk.download('rslp')
nltk.download('stopwords')
tf.debugging.set_log_device_placement(True)

args = Namespace(
    test_size = 0.3,
    random_state = 199,
    vocab_size = 10000,
    embedding_dim = 8,
    max_length = 10,
    batch_size=32,
    epochs=10,
    early_stopping_criteria=2,
    dropout_p=0.6,
    learning_rate = 0.0001,
    model_storage="model_storage/lstm",
    oov_tok = "<OOV>",
)

dataset = pd.read_csv("datasets/reviews2.csv")
X = dataset["review_comment_message"].copy()
y = dataset["review_score"].copy()
del dataset

y = y.apply(lambda x: x + 1)
filtro = y == 1
X = X[filtro]
y = y[filtro]

y = np.array(y)

y_dummy = np_utils.to_categorical(y)

def createLSTM_tf(activation="relu", neurons=8):
    input = tf.keras.Input(shape=(args.max_length))
    x = tf.keras.layers.Embedding(args.vocab_size, args.embedding_dim, input_length=args.max_length)(input)
    x = tf.keras.layers.LSTM(
        units = neurons,
        activation = activation, 
        return_sequences = True,
        recurrent_activation = "sigmoid",
        recurrent_dropout = 0,
        unroll = False,
        use_bias = True,
    )(x)
    x = tf.keras.layers.LSTM(units=neurons,activation=activation)(x)
    # x = tf.contrib.rnn.LSTMCell.StandardLSTM(units=neurons,activation=activation, return_sequences =True)(x)
    x = tf.keras.layers.Dropout(args.dropout_p)(x)
    output = tf.keras.layers.Dense(3, activation='sigmoid')(x)
    model = tf.keras.Model(input, output)
    
    model.compile(
        loss = tf.keras.losses.CategoricalCrossentropy(),
        optimizer= tf.keras.optimizers.Adam(
            learning_rate=args.learning_rate),
        metrics=['accuracy']
        )
    return model

X_train, X_test, y_train, y_test = train_test_split(X, y_dummy, test_size=args.test_size, random_state = args.random_state)


parameters = dict()
# parameters["epochs"] = [5,10]
parameters["activation"] = ["relu"]
parameters["neurons"] = [8]
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
model = MyModel(
    createLSTM_tf(neurons=32, activation='tanh'), 
    args.max_length,
    batch_size=args.batch_size, 
    epochs=args.epochs,
    return_sequences = False,
)

pipeline = Pipeline([("Cleaner", Cleaner()), 
                    ("Tokenizador", Tokenizador("portuguese")), 
                    ("RemoveStopWords", RemoveStopWords("portuguese")), 
                    ("Stemmer", Stemmer()), 
                    ("Joiner", Joiner()),
                    # ("Count", CountVectorizer()),
                    ("Count", TfidfVectorizer()),
                    ("Convert", Convert()),       
                    ("Padding", Padding(args.max_length, "post")),
                    ])

final_pipeline = Pipeline([
    ("init_pipe", pipeline),
    ("lstm", model),

])
print(final_pipeline)
            
final_pipeline.fit(X_train, y_train)

y_pred = final_pipeline.predict(X_test)

y_test_simples = model.transforma(y_test)
print(y_test_simples)


print("y_pred", y_pred[0])
print("y_test", y_test_simples[0])
resultado = pega_resultados("lstm", " - ", y_test_simples, y_pred, "acuracia", " - ")
# resultados.append(resultado)
print(resultado)

# ['lstm', 'Randomized Search', 0.5004485036288021, 0.2223550724637681, 0.16681616787626738, 0.3333333333333333, 'acuracia', ' - ']

