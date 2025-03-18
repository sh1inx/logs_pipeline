import pandas as pd
import pickle
import re
from sklearn.tree import DecisionTreeRegressor
import logging

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

def load_model():
    with open('model/model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

def load_feedback():
    expressions = []
    correct_results = []
    
    with open('feedback_logs.log', 'r') as file:
        for line in file.readlines():
            parts = line.split(", ")
            
            expression = parts[0].split(": ")[1]
            correct_result = float(parts[2].split(": ")[1].strip())
            
            expressions.append(expression)
            correct_results.append(correct_result)
    
    return expressions, correct_results


def retrain_model():
    expressions, correct_results = load_feedback()
    
    X = []
    y = correct_results
    
    for expression in expressions:
        features = process_expression(expression)
        X.append(features)
    
    model = load_model()
    
    model.fit(X, y)
    
    with open('model/model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    
    print("Modelo re-treinado com sucesso!")

retrain_model()
