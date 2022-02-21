from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
import nltk
import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
import tensorflow as tf

# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
nltk.download('rslp')
nltk.download('stopwords')

class Cleaner(BaseEstimator, TransformerMixin):
  def __init__(self):
    super()
  def clear(self, review):
    review = review.lower()
    # remove pula de linha 
    review = re.sub('\n', ' ', review)        
    review = re.sub('\r', ' ', review)

    # remove numero 
    review = re.sub(r'\d+(?:\.\d*(?:[eE]\d+))?', ' #numero ', review)

    # remove caracters especiais 
    review = re.sub(r'R\$', ' ', review)
    review = re.sub(r'\W', ' ', review)
    review = re.sub(r'\s+', ' ', review)

    # remove links 
    urls = re.findall('(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', review)
    if len(urls) > 0:
      for url in urls:
        for link in url:
          review = review.replace(link, '')
      review = review.replace(':', '')
      review = review.replace('/', '')
    return review
  def fit(self, X, y=None):
    return self
  def transform(self, X, y=None):
    review_transformando = []
    for review in X:
      review_transformando.append(self.clear(review))
    return review_transformando
  
  
  

class RemoveStopWords(BaseEstimator, TransformerMixin):
  def __init__(self, lingua):
    super()
    self.lingua = lingua
    self.stopword = stopwords.words(self.lingua)
  def removeStopwords(self, words_review):
    return [word for word in words_review if word not in self.stopword]
  def fit(self, X, y=None):
      return self
  def transform(self, X, y=None):
    review_transformando = []
    for review in X:
      review_transformando.append(self.removeStopwords(review))
    return review_transformando
  

class Tokenizador(BaseEstimator, TransformerMixin):
  def __init__(self, lingua):
    super()
    self.lingua = lingua
  def fit(self, X, y=None):
    return self
  def transform(self, X, y=None):    
    review_transformando = []
    for review in X:
      review_transformando.append(word_tokenize(review, language=self.lingua))
    return review_transformando
  

class Stemmer(BaseEstimator, TransformerMixin):
  def __init__(self):
    super()
    self.stem = RSLPStemmer()
  def useStem(self, words_review):
    return [self.stem.stem(word) for word in words_review ]
  def fit(self, X, y=None):
    return self
  def transform(self, X, y=None):
    review_transformando = []
    for review in X:
      review_transformando.append(self.useStem(review))
    return review_transformando

class Joiner(BaseEstimator, TransformerMixin):
  def __init__(self):
    super()
  def juntar(self, words_review):
    return " ".join(words_review)
  def fit(self, X, y=None):
      return self
  def transform(self, X, y=None):
    review_transformando = []
    for review in X:
      review_transformando.append(self.juntar(review))
    return review_transformando

def to_dense(array):
  indices = []
  values = []
  dense_shape=[len(array), len(array[0])]
  for i, j in zip(range(len(array)), range(len(array[0]))):
    if(array[i][j] != 0):
      indices.append([i, j])
      values.append(array[i][j])

  return tf.SparseTensor(indices=indices,
                      values=values,
                      dense_shape=dense_shape)

class Padding(BaseEstimator, TransformerMixin):
  def __init__(self, max_length, trunc_type):
    super()
    self.max_length = max_length
    self.trunc_type = trunc_type
  def fit(self, X, y=None):
      return self
  def transform(self, X, y=None):
    new_X = pad_sequences(X, dtype="float16", maxlen=self.max_length)
    # return tf.sparse.from_dense(new_X)
    return new_X
  
  def fit_transform(self, X, y=None):
    new_X = pad_sequences(X, dtype="float16", maxlen=self.max_length, truncating=self.trunc_type)
    return new_X

class Convert(BaseEstimator, TransformerMixin):
  def __init__(self):
    super()
  def fit(self, X, y=None):
      return self
  def transform(self, X, y=None):
    
    # return tf.sparse.reorder(to_dense(X), name=None)
    print(type(X.todense()))
    return np.array(X.todense())

class OurTokenizer():
  def __init__(self, vocab_size, oov_tok):
    super()
    self.tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
  def fit(self, X, y=None):
    self.tokenizer.fit_on_texts(X)
    return self
  def transform(self, X, y=None):
    return self.tokenizer.texts_to_sequences(X)


class MyModel(ClassifierMixin, BaseEstimator):
  def __init__(self, model, heigth, batch_size, epochs, return_sequences =True, **kargs ):
    super(**kargs)
    self.model = model
    self.heigth = heigth
    self.batch_size = batch_size
    self.epochs = epochs
    self.return_sequences = return_sequences

  def fit(self, X, y=None):
    if (self.return_sequences):
      y = np.array([yi * np.ones((self.heigth, len(yi) )) for yi in y])
      
    self.model.fit(X, y, self.batch_size, self.epochs)
    return self
  
  def _get_params(self, deep):
    return self.model.get_params(deep)
  def transforma(self, y):
    if (not self.return_sequences):
      uniques = np.arange(len(y))
      return uniques[y.argmax(1)]   

    uniques = np.arange(len(y[0]))    
    return uniques[y.argmax(1)]

  def predict(self, X):
    y = self.model.predict(X)
    print(y[0])
    if (self.return_sequences):
      y = np.array([yi[0] for yi in y])
    return self.transforma(y)

def pega_resultados(modelo, otimizador, y_test, y_predict, metrica_base, hiper):
  acc = accuracy_score(y_test, y_predict)
  f1 = f1_score(y_test, y_predict, average=None).mean()
  p = precision_score(y_test, y_predict, average=None).mean()
  r = recall_score(y_test, y_predict, average=None).mean()
  return [modelo, otimizador, acc , f1, p, r, metrica_base, hiper]


def salvando_em_arquivo(nome_do_arquivo, resultado):
  resultado_pd = pd.DataFrame(resultado)
  resultado_pd.columns = ["modelo", "otimizador",  "acuracia", "f1","precisao", "revocacao",  "metricas", "hiper paramentros" ]
  for coluna in ["acuracia", "f1","precisao", "revocacao"]:
    resultado_pd[coluna] = resultado_pd[coluna].apply(lambda x:  "%.2f" % (float(x)*100))  
  resultado_pd.to_csv(nome_do_arquivo) 