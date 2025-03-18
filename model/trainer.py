from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import pickle
import re

def process_expression(expression):
    numbers = list(map(int, re.findall(r'-?\d+', expression)))
    operators = re.findall(r'[+\-*/]', expression)
    
    features = []
    if len(numbers) == 2:
        features.extend(numbers)
        
        features.append(0 if operators[0] == '+' else
                        1 if operators[0] == '-' else
                        2 if operators[0] == '*' else 3)
        
        features.append(abs(numbers[0] - numbers[1]))
        
        features.extend([abs(numbers[0]), abs(numbers[1])])
        
        return features
    return [0, 0, 0, 0]

df = pd.read_csv('data/dataset.csv')
expressions = df['expression']
results = df['result']

X = []
y = results

for expression in expressions:
    features = process_expression(expression)
    X.append(features)

model = DecisionTreeRegressor()
model.fit(X, y)

with open('model/model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
