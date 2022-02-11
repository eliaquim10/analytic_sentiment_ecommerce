from sklearn.base import BaseEstimator, TransformerMixin
import nltk
import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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
  
  
  
  def fit_transform(self, X, y=None):
      return self.fit(X, y).transform(X, y)

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
  
  def fit_transform(self, X, y=None):
      return self.fit(X, y).transform(X, y)

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
  
  def fit_transform(self, X, y=None):
      return self.fit(X, y).transform(X, y)

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
  
  def fit_transform(self, X, y=None):
      return self.fit(X, y).transform(X, y)

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
  
  def fit_transform(self, X, y=None):
      return self.fit(X, y).transform(X, y)

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