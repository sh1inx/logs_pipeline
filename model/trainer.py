from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import pickle

df = pd.read_csv('data/dataset.csv')
expressions = df['expression']
results = df['result']

vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3))
X = vectorizer.fit_transform(expressions)
y = results

model = LinearRegression()
model.fit(X, y)

with open('model/model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('model/vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
