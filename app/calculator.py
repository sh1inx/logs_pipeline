import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.predictor import predict_expression

def main():
    print("Bem-vindo à calculadora IA!")
    while True:
        expression = input("Digite uma expressão ou 'sair' para terminar: ")
        if expression.lower() == 'sair':
            break
        result = predict_expression(expression)
        print(f"O resultado de {expression} é: {result}")

if __name__ == '__main__':
    main()
