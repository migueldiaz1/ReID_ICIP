#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 09:15:19 2025

@author: mdb
"""

import os
import yaml
import subprocess
import copy
import time
import math
import numpy as np
import torch
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET
import csv

# === Configuración general ===
CLASSES = ["Rubish", "Crosswalks", "Containers"]
MODEL_NAMES = ['resnet50', 'resnet50_ibn_a']
TRANSFER_ROOT = "/home/mdb/reid-strong-baseline/TRANSFER_Learning_SR_ensembling/"

VEHICLE_REID_WEIGHTS = {
    'resnet50': '/home/mdb/reid-strong-baseline/configs/resnet50-19c8e357 (1).pth',
    'resnet50_ibn_a': '/home/mdb/reid-strong-baseline/configs/r50_ibn_a.pth'
}

DATASET_PAIRS = {
    "base": "/home/mdb/DL_Lab3/UAM_DATASET/stratified/{}/",
    "sr": "/home/mdb/DL_Lab3/ST_DATASETS/UAM_SR/{}/",
    "sr_st": "/home/mdb/DL_Lab3/ST_DATASETS/UAM_SR_ST/{}/"
}

# Tamaños específicos de imagen por clase
CLASS_IMAGE_SIZES = {
    "Rubish": [160, 224],
    "Crosswalks": [320, 96],
    "Containers": [128, 160]
}

BASE_YAML_TEMPLATE = {
    'MODEL': {
        'PRETRAIN_CHOICE': 'imagenet',
        'NAME': None,
        'PRETRAIN_PATH': None,
        'METRIC_LOSS_TYPE': 'quadruplet',
        'IF_LABELSMOOTH': 'on',
        'IF_WITH_CENTER': 'no',
    },
    'INPUT': {
        'SIZE_TRAIN': [256, 256],
        'SIZE_TEST': [256, 256],
        'PROB': 0.5,
        'RE_PROB': 0.5,
        'PADDING': 10,
    },
    'DATASETS': {
        'NAMES': ['UAM'],
        'ROOT_DIR': None
    },
    'DATALOADER': {
        'SAMPLER': 'softmax_triplet',
        'NUM_INSTANCE': 4,
        'NUM_WORKERS': 0,
    },
    'SOLVER': {
        'OPTIMIZER_NAME': 'Adam',
        'MAX_EPOCHS': 50,
        'BASE_LR': 0.00035,
        'CENTER_LR': 0.5,
        'CENTER_LOSS_WEIGHT': 0.0005,
        'IMS_PER_BATCH': 64,
        'STEPS': [30, 45],
        'GAMMA': 0.1,
        'WEIGHT_DECAY': 0.0005,
        'WARMUP_FACTOR': 0.01,
        'WARMUP_ITERS': 10,
        'WARMUP_METHOD': 'linear',
        'QUADRUPLET_MARGIN1': 0.3,
        'QUADRUPLET_MARGIN2': 0.3,

    },
    'TEST': {
        'IMS_PER_BATCH': 128,
        'RE_RANKING': 'no',
        'NECK_FEAT': 'after',
        'FEAT_NORM': 'yes'
    }
}

# === Funciones auxiliares ===
def run_training(yaml_path):
    try:
        subprocess.run(["python", "tools/train.py", "--config_file", yaml_path], check=True)
                       #, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        print(f"[WARNING] Error en entrenamiento con {yaml_path}, continuando...")

def run_update(yaml_path, track_path):
    try:
        subprocess.run([
            "python", "tools/update.py",
            "--config_file", yaml_path,
            "--track", track_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        print(f"[WARNING] Error en update con {yaml_path}, continuando...")

def evaluate(track_path, base_path, name="Model"):
    try:
        print(f"Evaluando {name}...")
        subprocess.run([
            "python", "tools/Evaluate_UrbAM-ReID.py",
            "--track", track_path,
            "--path", base_path
        ], check=True)
    except subprocess.CalledProcessError:
        print(f"[WARNING] Error en evaluación: {name}")


def l2norm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def fusionar_tracks(track_paths, output_path, topk=100, decay=0.075):
    track_preds = []
    for path in track_paths:
        with open(path, 'r') as f:
            lines = [line.strip().split() for line in f.readlines()]
            track_preds.append(lines)

    num_queries = len(track_preds[0])

    with open(output_path, 'w') as fout:
        for q_idx in range(num_queries):
            votes = defaultdict(float)
            for track in track_preds:
                preds = track[q_idx]
                for rank, pred in enumerate(preds):
                    try:
                        pred_idx = int(pred)
                        weight = math.exp(-decay * rank)
                        votes[pred_idx] += weight
                    except ValueError:
                        continue

            sorted_preds = sorted(votes.items(), key=lambda x: -x[1])
            top_preds = [str(idx) for idx, _ in sorted_preds[:topk]]
            fout.write(" ".join(top_preds) + "\n")

# === ENTRENAMIENTO UAM + EVALUACIÓN MODELOS INDIVIDUALES ===
os.makedirs(TRANSFER_ROOT, exist_ok=True)

for class_name in CLASSES:
    for model_name in MODEL_NAMES:
        for dataset_key, uam_path_template in DATASET_PAIRS.items():
            print(f"\n=== Entrenando {class_name} | {model_name} | {dataset_key} ===")
            class_model_root = os.path.join(TRANSFER_ROOT, f"{class_name}_{model_name}_{dataset_key}")
            uam_output_dir = os.path.join(class_model_root, "UAM")
            os.makedirs(uam_output_dir, exist_ok=True)

            # === Configuración entrenamiento ===
            uam_yaml = copy.deepcopy(BASE_YAML_TEMPLATE)
            uam_yaml['MODEL']['NAME'] = model_name
            uam_yaml['MODEL']['PRETRAIN_PATH'] = VEHICLE_REID_WEIGHTS[model_name]
            uam_yaml['DATASETS']['ROOT_DIR'] = uam_path_template.format(class_name)
            uam_yaml['OUTPUT_DIR'] = uam_output_dir

            # Ajuste de tamaño de imagen específico
            size = CLASS_IMAGE_SIZES[class_name]
            uam_yaml['INPUT']['SIZE_TRAIN'] = size
            uam_yaml['INPUT']['SIZE_TEST'] = size

            uam_yaml_path = os.path.join(uam_output_dir, "train_uam.yaml")
            with open(uam_yaml_path, 'w') as f:
                yaml.dump(uam_yaml, f)

            run_training(uam_yaml_path)

            # === Update ===
            try:
                final_ckpt = sorted([f for f in os.listdir(uam_output_dir) if f.endswith('.pt')], key=lambda x: os.path.getmtime(os.path.join(uam_output_dir, x)), reverse=True)[0]
                final_ckpt_path = os.path.join(uam_output_dir, final_ckpt)
            except IndexError:
                continue

            update_yaml = copy.deepcopy(uam_yaml)
            update_yaml['DATASETS']['NAMES'] = ['UAM_test']
            update_yaml['TEST']['WEIGHT'] = final_ckpt_path
            
            update_yaml_path = os.path.join(uam_output_dir, "update_uam.yaml")
            with open(update_yaml_path, 'w') as f:
                yaml.dump(update_yaml, f)

            track_txt_path = os.path.join(uam_output_dir, "track.txt")
            run_update(update_yaml_path, track_txt_path)

            # === Evaluar modelo individual ===
            uam_base_path = uam_path_template.format(class_name)

            evaluate(
                track_txt_path,
                uam_base_path,
                name=f"{class_name}_{model_name}_{dataset_key}"
            )


print("\n=== BLOQUE 1 FINALIZADO === ")

print("\n=== BLOQUE 2: FUSIONANDO RESNET+IBN POR DATASET ===")

for class_name in CLASSES:
    for dataset_key in DATASET_PAIRS.keys():
        print(f"\n--> Fusionando {class_name} | {dataset_key}")

        model1_dir = os.path.join(TRANSFER_ROOT, f"{class_name}_resnet50_base", "UAM")
        model2_dir = os.path.join(TRANSFER_ROOT, f"{class_name}_resnet50_ibn_a_base", "UAM")

        if dataset_key == "sr":
            model1_dir = os.path.join(TRANSFER_ROOT, f"{class_name}_resnet50_sr", "UAM")
            model2_dir = os.path.join(TRANSFER_ROOT, f"{class_name}_resnet50_ibn_a_sr", "UAM")
        elif dataset_key == "sr_st":
            model1_dir = os.path.join(TRANSFER_ROOT, f"{class_name}_resnet50_sr_st", "UAM")
            model2_dir = os.path.join(TRANSFER_ROOT, f"{class_name}_resnet50_ibn_a_sr_st", "UAM")

        # Output fusionado
        output_dir = os.path.join(TRANSFER_ROOT, f"{class_name}_Combined_{dataset_key}", "UAM", "RESULTS")
        os.makedirs(output_dir, exist_ok=True)

        # Cargar features
        qf1 = np.load(os.path.join(model1_dir, "qf.npy"))
        gf1 = np.load(os.path.join(model1_dir, "gf.npy"))
        qf2 = np.load(os.path.join(model2_dir, "qf.npy"))
        gf2 = np.load(os.path.join(model2_dir, "gf.npy"))

        # Normalizar
        qf1 = l2norm(qf1)
        gf1 = l2norm(gf1)
        qf2 = l2norm(qf2)
        gf2 = l2norm(gf2)

        # === Fusión por suma ===
        qf_fused = l2norm((qf1 + qf2) / 2)
        gf_fused = l2norm((gf1 + gf2) / 2)

        # Productos punto
        q_g = np.dot(qf_fused, gf_fused.T)
        q_q = np.dot(qf_fused, qf_fused.T)
        g_g = np.dot(gf_fused, gf_fused.T)

        # Re-ranking
        from utils.re_ranking import re_ranking
        dist = re_ranking(q_g, q_q, g_g, k1=10, k2=6, lambda_value=0.1)
        indices = np.argsort(dist, axis=1)

        # Guardar track fusionado
        track_fusion_path = os.path.join(output_dir, "track.txt")
        with open(track_fusion_path, 'w') as f:
            for row in indices:
                adjusted = [str(i+1) for i in row[:100]]
                f.write(" ".join(adjusted) + "\n")

        print(f"✅ Guardado {track_fusion_path}")

        # Evaluar ensemble fusionado
        uam_base_path = DATASET_PAIRS[dataset_key].format(class_name)

        evaluate(
            track_fusion_path,
            uam_base_path,
            name=f"{class_name}_Combined_{dataset_key}"
        )


print("\n=== BLOQUE 2 FINALIZADO === 🚀")

print("\n=== BLOQUE 3: VOTACIÓN PONDERADA ENTRE BASE, SR Y SR_ST ===")

for class_name in CLASSES:
    print(f"\n--> Realizando votación ponderada para {class_name}")

    # Paths de los tracks fusionados resnet+ibn para cada dataset
    track_base = os.path.join(TRANSFER_ROOT, f"{class_name}_Combined_base", "UAM", "RESULTS", "track.txt")
    track_sr = os.path.join(TRANSFER_ROOT, f"{class_name}_Combined_sr", "UAM", "RESULTS", "track.txt")
    track_sr_st = os.path.join(TRANSFER_ROOT, f"{class_name}_Combined_sr_st", "UAM", "RESULTS", "track.txt")

    # Output votado
    output_dir = os.path.join(TRANSFER_ROOT, f"{class_name}_FINAL", "UAM", "RESULTS")
    os.makedirs(output_dir, exist_ok=True)

    fused_track_path = os.path.join(output_dir, "track.txt")

    # === Fusionar por votación weighted (exp decay) ===
    fusionar_tracks(
        [track_base, track_sr, track_sr_st],
        fused_track_path,
        topk=100,
        decay=0.075
    )

    print(f"✅ Guardado track fusionado por votación en: {fused_track_path}")

    # Evaluar la votación final por clase
    uam_base_path = DATASET_PAIRS["base"].format(class_name)

    evaluate(
        fused_track_path,
        uam_base_path,
        name=f"{class_name}_FINAL_VOTED"
    )

print("\n=== VOTACIÓN POR CLASE FINALIZADA === 🚀")

# === Generar el SUBMISSION GLOBAL ===
print("\n=== Generando SUBMISSION FINAL ===")

# Rutas globales
global_query_xml = Path("/home/mdb/DL_Lab3/Kaggle_Dataset/query_label.xml")
results_dir = Path(TRANSFER_ROOT)
submission_file = results_dir / "RESULT_SUBMISSION.txt"
output_csv_path = results_dir / "kaggle_submission_converted_1based.csv"

# Funciones auxiliares
def parse_items(xml_path):
    tree = ET.parse(xml_path)
    return [item.attrib["imageName"] for item in tree.getroot().findall("Item")]

def extract_number(filename):
    return str(int(filename.replace(".jpg", "").lstrip("0") or "0"))

# === Generar todos los nombres predichos ===
final_tracks = {}

for class_name in CLASSES:
    track_path = os.path.join(TRANSFER_ROOT, f"{class_name}_FINAL", "UAM", "RESULTS", "track.txt")
    with open(track_path, 'r') as f:
        lines = [line.strip().split() for line in f.readlines()]
    final_tracks[class_name] = lines

# Cargar queries globales
global_query_names = parse_items(global_query_xml)

# === Armar submission global ===
submission_lines = []
missing = []

idx_offset = 0  # Controlador del índice (acumulativo)

for i, qname in enumerate(global_query_names):
    # Decidir a qué clase pertenece este query
    if "Rubish" in qname:
        class_preds = final_tracks["Rubish"]
    elif "Crosswalks" in qname:
        class_preds = final_tracks["Crosswalks"]
    elif "Containers" in qname:
        class_preds = final_tracks["Containers"]
    else:
        class_preds = ["0"] * 100
        missing.append(qname)
        continue

    preds = class_preds[i]
    submission_lines.append(preds)

# === Guardar RESULT_SUBMISSION.txt ===
with open(submission_file, "w") as f:
    for preds in submission_lines:
        f.write(" ".join(preds) + "\n")

print(f"✅ Guardado RESULT_SUBMISSION.txt en {submission_file}")

# === Crear CSV Kaggle ===
num_queries = len(global_query_names)
image_name_format = "{:06d}.jpg"

converted_lines = []
all_indices = []

for i, preds in enumerate(submission_lines):
    image_name = image_name_format.format(i + 1)  # 1-based
    converted = [str(int(x)) for x in preds if x.isdigit()]
    converted_lines.append([image_name, " ".join(converted)])
    all_indices.extend(map(int, converted))

with open(output_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["imageName", "Corresponding Indexes"])
    writer.writerows(converted_lines)

min_idx = min(all_indices)
max_idx = max(all_indices)

print(f"✅ CSV para Kaggle guardado en {output_csv_path}")
print(f"✅ Índices predichos entre {min_idx} y {max_idx}")

if missing:
    print(f"⚠️ Queries faltantes: {len(missing)} → {missing[:5]}")

print("\n=== BLOQUE 3 FINALIZADO === 🚀🚀🚀")

