import pickle
import re

def load_model():
    with open('model/model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

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

def predict_expression(expression):
    model = load_model()
    features = process_expression(expression)
    result = model.predict([features])
    return result[0]
