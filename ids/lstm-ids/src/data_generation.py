import pandas as pd
import numpy as np
import os
import re

try:
    import observableToSink as obs
except ImportError:
    print("ERRO: Arquivo 'observableToSink.py' não encontrado.")
    print("Por favor, coloque-o na mesma pasta deste script.")
    exit()

print("Iniciando Etapa 1: Processamento dos logs brutos do Cooja...")


pastas_cenarios_treino = [
    "data/Volumes/T7/Vinnova-2021/Amika/DATASETS/data-traces1-NOMS-2023",
    "data/Volumes/T7/Vinnova-2021/Amika/DATASETS/data-traces2-MASS-2024"
]

pastas_cenarios_teste = [
    "data/Volumes/T7/Vinnova-2021/Amika/DATASETS/data-traces3-NOMS-2025"
]

# Processa os cenários de treino
for pasta in pastas_cenarios_treino:
    if not os.path.exists(pasta):
        print(f"AVISO: O caminho de TREINO não existe, pulando: {pasta}")
        continue
    print(f"Processando pasta de treino: {pasta}")
    try:
        obs.MyDataSet(dataAdd = pasta, binSize = 60)
    except Exception as e:
        print(f"ERRO ao processar a pasta {pasta}: {e}")
        print("Verifique se os logs (mote-output.log) estão presentes.")

# Processa os cenários de teste
for pasta in pastas_cenarios_teste:
    if not os.path.exists(pasta):
        print(f"AVISO: O caminho de TESTE não existe, pulando: {pasta}")
        continue
    print(f"Processando pasta de teste: {pasta}")
    try:
        obs.MyDataSet(dataAdd = pasta, binSize = 60)
    except Exception as e:
        print(f"ERRO ao processar a pasta {pasta}: {e}")
        print("Verifique se os logs (mote-output.log) estão presentes.")

print("Etapa 1 concluída. Os arquivos 'features_timeseries_60_sec.csv' foram gerados.")

print("\nIniciando Etapa 2: Agregação dos dados...")

trainDataList = []
testDataList = []
csv_filename = "features_timeseries_60_sec.csv"

# Carrega os CSVs de treino
for pasta in pastas_cenarios_treino:
    caminho_csv = os.path.join(pasta, csv_filename)
    if not os.path.exists(camin