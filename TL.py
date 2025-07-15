#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 14:23:15 2025

@author: mdb
"""

import os
import yaml
import subprocess
import copy
import numpy as np
import shutil
import xml.etree.ElementTree as ET
import csv
from pathlib import Path

# === Configuración ===
CLASSES = ["Rubish", "Crosswalks", "Containers"]
MODEL_NAMES = ['resnet50', 'resnet50_ibn_a']
VEHICLE_REID_WEIGHTS = {
    'resnet50': '/home/mdb/reid-strong-baseline/configs/resnet50-19c8e357 (1).pth',
    'resnet50_ibn_a': '/home/mdb/reid-strong-baseline/configs/r50_ibn_a.pth'
}

UAM_BASE_PATH = "/home/mdb/DL_Lab3/UAM_DATASET/stratified/{}/"
RIVAS_BASE_PATH = "/home/mdb/DL_Lab3/RIVAS_DATASET/DATASET/{}/"
TRANSFER_ROOT = Path("/home/mdb/reid-strong-baseline/TRANSFER_Learning/")

BASE_YAML_TEMPLATE = {
    'MODEL': {
        'PRETRAIN_CHOICE': 'imagenet',
        'NAME': None,
        'PRETRAIN_PATH': None,
        'METRIC_LOSS_TYPE': 'triplet_center',
        'IF_LABELSMOOTH': 'on',
        'IF_WITH_CENTER': 'yes',
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
        'NUM_WORKERS': 6,
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
    },
    'TEST': {
        'IMS_PER_BATCH': 128,
        'RE_RANKING': 'no',
        'NECK_FEAT': 'after',
        'FEAT_NORM': 'yes'
    }
}

TRANSFER_ROOT.mkdir(parents=True, exist_ok=True)

def run_training(yaml_path):
    try:
        subprocess.run(["python", "tools/train.py", "--config_file", yaml_path], check=True)
    except subprocess.CalledProcessError:
        print(f"[WARNING] Error en entrenamiento con {yaml_path}, continuando...")

def run_update(yaml_path, track_path):
    try:
        subprocess.run(["python", "tools/update.py", "--config_file", yaml_path, "--track", track_path], check=True)
    except subprocess.CalledProcessError:
        print(f"[WARNING] Error en update con {yaml_path}, continuando...")

# === ENTRENAR UAM y FINE-TUNE RIVAS ===
for class_name in CLASSES:
    for model_name in MODEL_NAMES:
        print(f"\n=== Procesando clase: {class_name} | Modelo: {model_name} ===")

        class_model_root = TRANSFER_ROOT / f"{class_name}_{model_name}"
        uam_output_dir = class_model_root / "UAM"
        rivas_output_dir = class_model_root / "RIVAS"
        uam_output_dir.mkdir(parents=True, exist_ok=True)
        rivas_output_dir.mkdir(parents=True, exist_ok=True)

        # === ENTRENAR UAM ===
        uam_yaml = copy.deepcopy(BASE_YAML_TEMPLATE)
        uam_yaml['MODEL']['NAME'] = model_name
        uam_yaml['MODEL']['PRETRAIN_PATH'] = VEHICLE_REID_WEIGHTS[model_name]
        uam_yaml['DATASETS']['ROOT_DIR'] = UAM_BASE_PATH.format(class_name)
        uam_yaml['OUTPUT_DIR'] = str(uam_output_dir)

        if class_name == "Rubish":
            uam_yaml['SOLVER']['MAX_EPOCHS'] = 30
            uam_yaml['SOLVER']['STEPS'] = [15, 25]
            uam_yaml['DATALOADER']['NUM_INSTANCE'] = 2
            uam_yaml['SOLVER']['IMS_PER_BATCH'] = 32

        uam_yaml_path = uam_output_dir / "train_uam.yaml"
        with open(uam_yaml_path, 'w') as f:
            yaml.dump(uam_yaml, f)

        print(f"[UAM] Entrenando {model_name} en {class_name}...")
        run_training(str(uam_yaml_path))

        # === ENTRENAR RIVAS ===
        try:
            latest_ckpt = sorted(uam_output_dir.glob("*.pth"), key=os.path.getmtime)[-1]
            pretrained_uam_ckpt = str(latest_ckpt)
        except IndexError:
            print(f"[ERROR] No checkpoint encontrado en {uam_output_dir}. Saltando {class_name}-{model_name}.")
            continue

        rivas_yaml = copy.deepcopy(BASE_YAML_TEMPLATE)
        rivas_yaml['MODEL']['NAME'] = model_name
        rivas_yaml['MODEL']['PRETRAIN_PATH'] = pretrained_uam_ckpt
        rivas_yaml['MODEL']['PRETRAIN_CHOICE'] = 'self'
        rivas_yaml['DATASETS']['ROOT_DIR'] = RIVAS_BASE_PATH.format(class_name)
        rivas_yaml['SOLVER']['BASE_LR'] = 0.000035
        rivas_yaml['SOLVER']['MAX_EPOCHS'] = 30
        rivas_yaml['SOLVER']['STEPS'] = [15, 25]
        rivas_yaml['OUTPUT_DIR'] = str(rivas_output_dir)

        if class_name == "Rubish":
            rivas_yaml['SOLVER']['MAX_EPOCHS'] = 30
            rivas_yaml['SOLVER']['STEPS'] = [15, 25]
            rivas_yaml['DATALOADER']['NUM_INSTANCE'] = 2
            rivas_yaml['SOLVER']['IMS_PER_BATCH'] = 32

        rivas_yaml_path = rivas_output_dir / "train_rivas.yaml"
        with open(rivas_yaml_path, 'w') as f:
            yaml.dump(rivas_yaml, f)

        print(f"[RIVAS] Fine-tuning {model_name} en {class_name}...")
        run_training(str(rivas_yaml_path))

        # === UPDATE ===
        try:
            final_ckpt = sorted(rivas_output_dir.glob("*.pth"), key=os.path.getmtime)[-1]
        except IndexError:
            print(f"[ERROR] No checkpoint final en {rivas_output_dir}. Saltando update.")
            continue

        update_yaml = copy.deepcopy(rivas_yaml)
        update_yaml['TEST']['WEIGHT'] = str(final_ckpt)
        update_yaml_path = rivas_output_dir / "update_rivas.yaml"
        with open(update_yaml_path, 'w') as f:
            yaml.dump(update_yaml, f)

        track_txt_path = rivas_output_dir / "track.txt"
        print(f"[RIVAS] Generando track.txt para {model_name} en {class_name}...")
        run_update(str(update_yaml_path), str(track_txt_path))

print("\n=== ENTRENAMIENTO Y UPDATE COMPLETADO ===")


# === FUSIONAR RESULTADOS IBN + RESNET POR CLASE ===
import numpy as np
import shutil
from utils.re_ranking import re_ranking

print("\n=== FUSIONANDO RESULTADOS DE IBN + RESNET ===")

for CLASE in CLASSES:
    print(f"\n--> Procesando clase: {CLASE}")

    MODEL1_DIR = os.path.join(TRANSFER_ROOT, f"{CLASE}_resnet50_ibn_a", "RIVAS")
    MODEL2_DIR = os.path.join(TRANSFER_ROOT, f"{CLASE}_resnet50", "RIVAS")
    OUTPUT_DIR = os.path.join(TRANSFER_ROOT, f"{CLASE}_Combined", "RIVAS")
    RESULTS_DIR = os.path.join(OUTPUT_DIR, "RESULTS")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    try:
        # === Cargar y normalizar features ===
        qf1 = np.load(os.path.join(MODEL1_DIR, "qf.npy"))
        gf1 = np.load(os.path.join(MODEL1_DIR, "gf.npy"))
        qf2 = np.load(os.path.join(MODEL2_DIR, "qf.npy"))
        gf2 = np.load(os.path.join(MODEL2_DIR, "gf.npy"))

        def l2norm(x):
            return x / np.linalg.norm(x, axis=1, keepdims=True)

        # Normalizar embeddings
        qf1 = l2norm(qf1)
        gf1 = l2norm(gf1)
        qf2 = l2norm(qf2)
        gf2 = l2norm(gf2)

        # Concatenar embeddings
        qf_concat = np.concatenate([qf1, qf2], axis=1)
        gf_concat = np.concatenate([gf1, gf2], axis=1)

        # Calcular matrices de similitud
        q_g = np.dot(qf_concat, gf_concat.T)
        q_q = np.dot(qf_concat, qf_concat.T)
        g_g = np.dot(gf_concat, gf_concat.T)

        # Re-ranking
        k1, k2, lmbd = 10, 6, 0.1
        dist = re_ranking(q_g, q_q, g_g, k1=k1, k2=k2, lambda_value=lmbd)
        indices = np.argsort(dist, axis=1)

        # Guardar track.txt
        track_path = os.path.join(RESULTS_DIR, "track.txt")
        with open(track_path, 'w') as f:
            for row in indices:
                adjusted = [str(i + 1) for i in row[:100]]
                f.write(" ".join(adjusted) + "\n")

        # Guardar configuración
        with open(os.path.join(RESULTS_DIR, "best_config.txt"), 'w') as f:
            f.write(f"k1={k1}\nk2={k2}\nlambda={lmbd:.2f}\n")

        print(f"TRACK.txt fusionado guardado en: {track_path}")

        # Verificar predicciones
        with open(track_path, "r") as f:
            lines = [line.strip().split() for line in f.readlines()]

        preds_por_linea = [len(line) for line in lines]
        if all(n == 100 for n in preds_por_linea):
            print("✅ Todas las queries tienen 100 predicciones.")
        else:
            print("⚠️ Alerta: algunas queries no tienen 100 predicciones.")

    except Exception as e:
        print(f"[ERROR] Fallo en la fusión para {CLASE}: {e}")

print("\n=== FUSIÓN COMPLETADA ===")


# === UNIFICAR TRACKS POR CLASE EN UN SOLO ARCHIVO GLOBAL ===
import xml.etree.ElementTree as ET
from pathlib import Path
import csv

print("\n=== GENERANDO RESULTADOS GLOBALES PARA SUBMISSION ===")

# CONFIGURACIÓN GENERAL
final_clases = ["Containers", "Crosswalks", "Rubish"]
base_dir = Path("/home/mdb/DL_Lab3/RIVAS_DATASET/DATASET")
results_dir = Path("/home/mdb/reid-strong-baseline/TRANSFER_Learning/")
global_query_xml = Path("/home/mdb/DL_Lab3/Kaggle_Dataset/query_label.xml")

# SALIDAS
names_file = results_dir / "Results_RIVAS_ALL_names.txt"
indices_file = results_dir / "Results_RIVAS_ALL_indices.txt"
submission_file = results_dir / "RESULT_SUBMISSION.txt"
output_csv_path = results_dir / "kaggle_submission_converted_1based.csv"

# FUNCIONES

def parse_items(xml_path):
    tree = ET.parse(xml_path)
    return [item.attrib["imageName"] for item in tree.getroot().findall("Item")]

def extract_number(filename):
    return str(int(filename.replace(".jpg", "").lstrip("0") or "0"))

# PASO 1: Combinar nombres de queries y sus predicciones
all_entries = []

for cls in final_clases:
    query_xml = base_dir / cls / "query_label.xml"
    test_xml = base_dir / cls / "test_label.xml"
    track_file = results_dir / f"{cls}_Combined/RIVAS/RESULTS/track.txt"

    query_names = parse_items(query_xml)
    gallery_names = parse_items(test_xml)

    with open(track_file, "r") as f:
        lines = [line.strip().split() for line in f.readlines()]

    assert len(query_names) == len(lines), f"⚠️ Mismatch en {cls}"

    for qname, pred_indices in zip(query_names, lines):
        gallery_preds = []
        for idx in pred_indices:
            try:
                g_idx = int(idx) - 1
                if 0 <= g_idx < len(gallery_names):
                    gallery_preds.append(gallery_names[g_idx])
                else:
                    gallery_preds.append("unknown.jpg")
            except:
                gallery_preds.append("error.jpg")
        all_entries.append((qname, gallery_preds))

# PASO 2: Reordenar según el global query

global_query_names = parse_items(global_query_xml)
query_to_predictions = {q: preds for q, preds in all_entries}

final_lines = []
missing = []

for qname in global_query_names:
    preds = query_to_predictions.get(qname)
    if preds:
        final_lines.append(" ".join([qname] + preds))
    else:
        final_lines.append(qname + " " + "missing.jpg "*100)
        missing.append(qname)

with open(names_file, "w") as f:
    f.write("\n".join(final_lines))

print(f" Guardado names file: {names_file}")

# PASO 3: Convertir a índices
with open(names_file, "r") as fin:
    lines = fin.readlines()

with open(indices_file, "w") as fout:
    for line in lines:
        parts = line.strip().split()
        numbers = [extract_number(name) for name in parts]
        fout.write(" ".join(numbers) + "\n")

print(f" Guardado indices file: {indices_file}")

# PASO 4: Crear submission.txt
with open(indices_file, "r") as fin, open(submission_file, "w") as fout:
    for line in fin:
        parts = line.strip().split()
        fout.write(" ".join(parts[1:]) + "\n")

print(f" Guardado submission file: {submission_file}")

# PASO 5: Crear CSV para Kaggle
num_queries = 346
image_name_format = "{:06d}.jpg"

with open(submission_file, "r") as f:
    lines = [line.strip().split() for line in f.readlines()]

assert len(lines) == num_queries, f"⚠️ Se esperaban {num_queries} queries, pero hay {len(lines)}"

converted_lines = []
all_indices = []

for i, row in enumerate(lines):
    image_name = image_name_format.format(i + 1)
    converted = [str(int(x)) for x in row if x.isdigit()]
    converted_lines.append([image_name, " ".join(converted)])
    all_indices.extend(map(int, converted))

with open(output_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["imageName", "Corresponding Indexes"])
    writer.writerows(converted_lines)

min_idx = min(all_indices)
max_idx = max(all_indices)

print(f" CSV para Kaggle generado en: {output_csv_path}")
print(f"Mínimo índice en predicciones: {min_idx}")
print(f"Máximo índice en predicciones: {max_idx}")
if missing:
    print(f" Queries faltantes: {len(missing)} → {missing[:5]}")
