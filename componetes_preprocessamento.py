from sklearn.base import BaseEstimator, TransformerMixin
import nltk
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.tokenize import word_tokenize
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

# cleaner = Cleaner()
# X = np.arange(0, 100, dtype=int)
# X = X.reshape((10,10))
# y = np.arange(1, 10, dtype=int)
# print(X.T, "\n", y)
# print(cleaner.fit(X, y))