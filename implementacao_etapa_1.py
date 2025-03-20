import random
import numpy as np
import pandas as pd
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.exceptions import NotFittedError
import matplotlib.pyplot as plt

# 1. Coleta de Logs - Simulamos um conjunto de logs ficticios

def gerar_logs_ficticios(n=1000):
    """Gera logs simulados com tempo de resposta, sucesso de acesso e faixa etaria."""
    logs = []
    for _ in range(n):
        ip = f"192.168.{random.randint(0, 255)}.{random.randint(0, 255)}"
        tempo_resposta = random.uniform(0.1, 5.0)
        acesso_sucesso = random.choice([True, False])
        faixa_etaria = random.choice(["<18", "18-30", "31-50", "51+"])
        logs.append([ip, tempo_resposta, acesso_sucesso, faixa_etaria])
    return logs

# 2. Ingestao - Estruturamos os dados do log
df_logs = pd.DataFrame(gerar_logs_ficticios(), columns=["IP", "Tempo_Resposta", "Acesso_Sucesso", "Faixa_Etaria"])

# 3. Pre-processamento - Normalizamos os dados e extraimos informacoes relevantes
historico_treino = []
rotulos_treino = []
modelo = DecisionTreeClassifier()
modelo_treinado = False

def processar_logs(df):
    """Transforma os dados para analise."""
    df["Tempo_Normalizado"] = (df["Tempo_Resposta"] - df["Tempo_Resposta"].mean()) / df["Tempo_Resposta"].std()
    return df

df_logs = processar_logs(df_logs)

# 4. Analise com IA - Treinamento do modelo de decisao
def treinar_modelo():
    global modelo_treinado
    if len(historico_treino) > 5:
        X = np.array(historico_treino)
        y = np.array(rotulos_treino)
        modelo.fit(X, y)
        modelo_treinado = True

def prever_classificacao():
    if not modelo_treinado or len(historico_treino) < 5:
        return "Normal"
    try:
        return modelo.predict([historico_treino[-1]])[0]
    except NotFittedError:
        return "Normal"

# 5. Classificacao automatica e resposta
log_respostas = []

def avaliar_acesso(ip, tempo_resposta, sucesso, faixa_etaria):
    dificuldade = prever_classificacao()
    categoria = "Crítico" if not sucesso else ("Suspeito" if tempo_resposta > 2 else "Normal")
    log_respostas.append({"IP": ip, "Tempo": tempo_resposta, "Classificacao": categoria, "Faixa_Etaria": faixa_etaria})
    historico_treino.append([tempo_resposta])
    rotulos_treino.append(categoria)
    treinar_modelo()
    
    return categoria

df_logs["Classificacao"] = df_logs.apply(lambda row: avaliar_acesso(row["IP"], row["Tempo_Resposta"], row["Acesso_Sucesso"], row["Faixa_Etaria"]), axis=1)

# 6. Exibicao de estatisticas adicionais
def mostrar_estatisticas(df):
    print("\nEstatisticas de Tempo de Resposta por Dificuldade:")
    print(df.groupby("Classificacao")["Tempo_Resposta"].mean())
    
    print("\nTaxa de Acertos Ficticia:")
    taxa_acerto = random.uniform(0.5, 0.95)
    print(f"Taxa de Acertos Simulada: {taxa_acerto * 100:.2f}%")
    
    print("\nDiferencas por Faixa Etaria:")
    print(df.groupby("Faixa_Etaria")["Tempo_Resposta"].mean())

    # Salvar os logs no arquivo txt
    with open("logs_acessos.txt", "w") as file:
        file.write("IP | Tempo de Resposta | Classificacao | Faixa Etaria\n")
        file.write("-" * 50 + "\n")
        for log in log_respostas:
            file.write(f"{log['IP']} | {log['Tempo']:.2f}s | {log['Classificacao']} | {log['Faixa_Etaria']}\n")
    print("\nLogs salvos em logs_acessos.txt")

# 7. Visualizacao
def exibir_grafico(df):
    plt.figure(figsize=(10, 5))
    cores = {"Normal": "green", "Suspeito": "orange", "Crítico": "red"}
    plt.scatter(df.index, df["Tempo_Resposta"], c=df["Classificacao"].map(cores), label="Acessos")
    plt.xlabel("Registro de Acesso")
    plt.ylabel("Tempo de Resposta (s)")
    plt.title("Classificacao de Acessos")
    plt.legend()
    plt.show()

def menu():
    while True:
        print("\nMENU:")
        print("1 - Ver estatisticas")
        print("2 - Exibir grafico")
        print("3 - Ver logs salvos")
        print("4 - Sair")
        opcao = input("Escolha uma opcao: ")
        
        if opcao == "1":
            mostrar_estatisticas(df_logs)
        elif opcao == "2":
            exibir_grafico(df_logs)
        elif opcao == "3":
            with open("logs_acessos.txt", "r") as file:
                print(file.read())
        elif opcao == "4":
            print("Saindo...")
            break
        else:
            print("Opcao invalida! Tente novamente.")

menu()
