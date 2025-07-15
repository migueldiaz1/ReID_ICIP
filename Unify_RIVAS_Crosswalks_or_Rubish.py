#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 22:46:19 2025

@author: mdb
"""


import numpy as np
import os
from utils.re_ranking import re_ranking
import shutil

# === CONFIGURACIÓN GENERAL ===
CLASSES = ["Crosswalks"]
BASE_PATH = "/home/mdb/reid-strong-baseline/TRAINING_RIVAS/"
DATASET_PATH_BASE = "/home/mdb/DL_Lab3/RIVAS_DATASET/DATASET"
OUTPUT_PATH = "/home/mdb/reid-strong-baseline/TRAINING_RIVAS/Crosswalks/COMBINED/RESULTS"

for CLASE in CLASSES:
    MODEL1_DIR = os.path.join(BASE_PATH, CLASE, "IBN")
    MODEL2_DIR = os.path.join(BASE_PATH, CLASE, "Resnet")
    OUTPUT_DIR = os.path.join(BASE_PATH, CLASE, "COMBINED")
    RESULTS_DIR = os.path.join(OUTPUT_DIR, "RESULTS")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # === CARGAR Y NORMALIZAR FEATURES ===
    qf1 = np.load(os.path.join(MODEL1_DIR, "qf.npy"))
    gf1 = np.load(os.path.join(MODEL1_DIR, "gf.npy"))
    qf2 = np.load(os.path.join(MODEL2_DIR, "qf.npy"))
    gf2 = np.load(os.path.join(MODEL2_DIR, "gf.npy"))

    # Normalización L2
    def l2norm(x):
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    qf1 = l2norm(qf1)
    gf1 = l2norm(gf1)
    qf2 = l2norm(qf2)
    gf2 = l2norm(gf2)

    # Concatenar embeddings
    qf = np.concatenate([qf1, qf2], axis=1)
    gf = np.concatenate([gf1, gf2], axis=1)
    
    # Precalcular dot products
    q_g = np.dot(qf, gf.T)
    q_q = np.dot(qf, qf.T)
    g_g = np.dot(gf, gf.T)
    
    ########### ATTEntion changed
    qg1, qg2 = np.dot(qf1, gf1.T), np.dot(qf2, gf2.T)
    qq1, qq2 = np.dot(qf1, qf1.T), np.dot(qf2, qf2.T)
    gg1, gg2 = np.dot(gf1, gf1.T), np.dot(gf2, gf2.T)
    alpha = 0.5
    beta = 0.5
    q_g = alpha * qg1 + beta * qg2
    q_q = alpha * qq1 + beta * qq2
    g_g = alpha * gg1 + beta * gg2
    ##################################

    
    # Valores fijos asumidos como los mejores
    k1, k2, lmbd = 10, 6, 0.1
    
    track_path = os.path.join(OUTPUT_PATH, f"track_k1{k1}_k2{k2}_l{int(lmbd * 100)}.txt")
    
    # q_g, q_q, g_g: matrices de distancia que ya deben estar cargadas
    dist = re_ranking(q_g, q_q, g_g, k1=k1, k2=k2, lambda_value=lmbd)
    indices = np.argsort(dist, axis=1)
    
    with open(track_path, 'w') as f:
        for row in indices:
            adjusted = [str(i + 1) for i in row[:100]]  # suma 1 a cada índice (para 1-based)
            f.write(" ".join(adjusted) + "\n")
    
    # Copiar directamente como TRACK.txt al directorio de resultados
    shutil.copyfile(track_path, os.path.join(OUTPUT_PATH, "TRACK.txt"))
    
    # Guardar la configuración asumida como "mejor"
    with open(os.path.join(OUTPUT_PATH, "best_config.txt"), 'w') as f:
        f.write(f"k1={k1}\nk2={k2}\nlambda={lmbd:.2f}\n")
    
    print(f"TRACK.txt guardado en: {RESULTS_DIR}")
    
    
# Verificación lectura posterior (opcional)
with open(track_path, "r") as f:
    lines = [line.strip().split() for line in f.readlines()]

preds_por_linea = [len(line) for line in lines]
if all(n == 100 for n in preds_por_linea):
    print("Todas las queries tienen 100 predicciones.")
else:
    print("⚠️ Alerta: algunas queries no tienen 100 predicciones.")
    for i, n in enumerate(preds_por_linea):
        if n != 100:
            print(f"  - Query {i+1}: {n} predicciones")
