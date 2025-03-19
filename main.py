import random
import numpy as np
import pandas as pd
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.exceptions import NotFittedError

# Histórico de treino e rótulos para a IA
historico_treino = []
rotulos_treino = []

# Modelo de IA
modelo = DecisionTreeClassifier()
modelo_treinado = False

# Função para carregar o CSV e determinar a dificuldade inicial
def carregar_dificuldade_inicial():
    try:
        # Lê o arquivo de log (se existir)
        df = pd.read_csv("log_respostas.csv")
        if not df.empty:
            # Verifica as últimas dificuldades registradas
            dificuldades = df['dificuldade'].tolist()
            # Se a última dificuldade foi "médio", inicia no "médio"
            if "médio" in dificuldades[-5:]:
                return "médio"  # Considera as últimas 5 respostas
            return dificuldades[-1]  # Caso contrário, começa na última dificuldade registrada
        else:
            return "fácil"  # Se o CSV estiver vazio, começamos com "fácil"
    except FileNotFoundError:
        return "fácil"  # Se o arquivo não existir, começamos com "fácil"

# Função para treinar o modelo
def treinar_modelo():
    global modelo_treinado
    if len(historico_treino) > 5:
        X = np.array(historico_treino)
        y = np.array(rotulos_treino)
        modelo.fit(X, y)
        modelo_treinado = True

# Função para prever a dificuldade com base no modelo treinado
def prever_dificuldade():
    if not modelo_treinado or len(historico_treino) < 5:
        return "fácil"
    
    ultimo_desempenho = np.array([historico_treino[-1]])
    try:
        dificuldade_predita = modelo.predict(ultimo_desempenho)[0]
        return dificuldade_predita if dificuldade_predita in ["fácil", "médio", "difícil"] else "fácil"
    except NotFittedError:
        return "fácil"

# Função para gerar perguntas com base na dificuldade
def gerar_pergunta(dificuldade):
    if dificuldade == "fácil":
        a, b = random.randint(1, 10), random.randint(1, 10)
        operador = random.choice(["+", "-"])
    elif dificuldade == "médio":
        a, b = random.randint(10, 50), random.randint(10, 50)
        operador = random.choice(["+", "-", "*"])
    else:
        a, b = random.randint(50, 100), random.randint(50, 100)
        operador = random.choice(["+", "-", "*", "/"])
    
    expressao = f"{a} {operador} {b}"
    resposta_correta = eval(expressao)
    return expressao, round(resposta_correta, 2)

# Log de respostas estruturado
log_respostas = []

# Função para salvar logs em um arquivo CSV
def salvar_log():
    # Carrega os dados existentes do CSV
    try:
        df_existente = pd.read_csv("log_respostas.csv")
    except FileNotFoundError:
        df_existente = pd.DataFrame()

    # Adiciona os novos logs ao DataFrame existente
    df_novos_logs = pd.DataFrame(log_respostas)
    df_atualizado = pd.concat([df_existente, df_novos_logs], ignore_index=True)

    # Salva o DataFrame atualizado no CSV
    df_atualizado.to_csv("log_respostas.csv", index=False)

# Função para avaliar a resposta e ajustar o modelo
def avaliar_resposta(pergunta, resposta_usuario, resposta_correta, dificuldade, tempo_resposta):
    # Tolerância baseada no valor da operação (ex: 5% de tolerância para operações grandes)
    tolerancia_percentual = 0.05
    margem_erro = abs(resposta_correta) * tolerancia_percentual

    # Verifica se a diferença entre a resposta do usuário e a resposta correta está dentro da margem de erro
    correta = abs(resposta_usuario - resposta_correta) <= margem_erro
    
    log_respostas.append({
        "pergunta": pergunta, 
        "resposta": resposta_usuario, 
        "correta": correta, 
        "dificuldade": dificuldade,
        "tempo_resposta": tempo_resposta  # Tempo de resposta adicionado no log
    })
    
    # Calculando a taxa de acerto
    acertos = sum(1 for r in log_respostas if r["correta"])
    taxa_acerto = acertos / len(log_respostas)
    historico_treino.append([len(log_respostas), taxa_acerto])
    
    # Ajuste nos rótulos com base na taxa de acerto
    if taxa_acerto > 0.7:
        rotulos_treino.append("médio" if dificuldade == "fácil" else "difícil")
    elif taxa_acerto < 0.4:
        rotulos_treino.append("fácil" if dificuldade == "médio" else "médio")
    else:
        rotulos_treino.append(dificuldade)
    
    # Salvar log e treinar modelo
    salvar_log()
    treinar_modelo()
    return correta

# Função para iniciar o jogo
def iniciar_jogo():
    dificuldade = carregar_dificuldade_inicial()  # Determina a dificuldade inicial baseada no CSV
    print(f"Iniciando o jogo na dificuldade: {dificuldade}")
    
    while True:
        dificuldade = prever_dificuldade()
        pergunta, resposta_correta = gerar_pergunta(dificuldade)
        print(f"Resolva: {pergunta}")
        
        # Verifica se o usuário quer sair
        resposta_usuario = input("Sua resposta (ou digite 'sair' para sair): ")
        
        if resposta_usuario.lower() == "sair":
            print("Saindo do jogo... até logo!")
            break
        
        try:
            resposta_usuario = float(resposta_usuario)
        except ValueError:
            print("Por favor, insira um número válido ou digite 'sair' para sair.")
            continue
        
        # Começando o temporizador antes de o usuário digitar a resposta
        tempo_inicio = time.time()  # Inicia o tempo
        tempo_fim = time.time()  # Finaliza o tempo
        
        # Calcula o tempo de resposta
        tempo_resposta = tempo_fim - tempo_inicio
        
        # Avalia a resposta e ajusta o modelo
        correta = avaliar_resposta(pergunta, resposta_usuario, resposta_correta, dificuldade, tempo_resposta)
        print("Correto!" if correta else f"Errado! A resposta era {resposta_correta}")

if __name__ == "__main__":
    iniciar_jogo()
