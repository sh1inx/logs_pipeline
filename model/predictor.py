import pickle

def load_model_and_vectorizer():
    with open('model/model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    
    with open('model/vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    
    return model, vectorizer

def predict_expression(expression):
    model, vectorizer = load_model_and_vectorizer()
    
    expression_vector = vectorizer.transform([expression])
    
    result = model.predict(expression_vector)
    
    return result[0]
