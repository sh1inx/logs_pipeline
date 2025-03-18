import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.predictor import predict_expression
from model.feedback_trainer import retrain_model

logging.basicConfig(filename='feedback_logs.log', level=logging.INFO)

def save_feedback(expression, model_result, python_result, is_correct):
    logging.info(f"Expressão: {expression}, Resultado do modelo: {model_result}, Resultado correto: {python_result}, Correto: {is_correct}")

def evaluate_expression(expression):
    try:
        return eval(expression)
    except Exception as e:
        print(f"Erro ao calcular a expressão: {e}")
        return None

def main():
    print("Bem-vindo à calculadora IA!")
    feedback_counter = 0

    while True:
        expression = input("Digite uma expressão ou 'sair' para terminar: ")
        
        if expression.lower() == 'sair':
            break
        
        result = predict_expression(expression)
        print(f"O resultado de {expression} é: {result}")
        
        python_result = evaluate_expression(expression)
        
        if python_result is None:
            print("A expressão é inválida.")
            continue
        
        is_correct = abs(result - python_result) < 1e-6

        save_feedback(expression, result, python_result, is_correct)
        
        if is_correct:
            print("O cálculo está correto!")
        else:
            print(f"O cálculo está incorreto. O valor correto é: {python_result}")
        
        feedback_counter += 1
        if feedback_counter >= 10:
            print("Re-treinando o modelo com os feedbacks mais recentes...")
            retrain_model()
            feedback_counter = 0 

if __name__ == '__main__':
    main()
