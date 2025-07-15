#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 08:56:53 2025

@author: mdb
"""

import os
import yaml
import subprocess
import copy
import math
import numpy as np
import shutil
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET
import csv
from utils.re_ranking import re_ranking

###########################################################################################################################
# ================== ELIJO EL BLOQUE DE MODELOS Y DATASETS A USAR ==================
USE_MODEL_PAIR = 'seresnet50_IBN'         # opciones: 'resnet50', 'seresnet50', 'resnet101', 'seresnet50_IBN'. 'resnet_50_101 NOTA: AÑADE MODELOS FACIL ABAJO
USE_DATASET_SET = 'augmv1_augmv2'   # opciones: 'original_sr_st', 'augmv1_augmv2', 'original_3x'
METRIC_LOSS_TYPE = 'quadruplet'  # opciones: 'triplet_center', 'quadruplet'
USE_UNIFIED_MODEL = False  # True es para unified, False para por clases NOTA: No está refinado, falta terminar  -> Meanwhile por False
###########################################################################################################################


# ================== CONFIGURACIÓN GENERAL ==================
CLASSES = ["Rubish", "Crosswalks", "Containers"]
TRANSFER_ROOT = "/home/mdb/reid-strong-baseline/TRANSFER_Learning_SR_ensembling/"
KAGGLE_ROOT = "/home/mdb/DL_Lab3/Kaggle_Dataset"
RIVAS_ROOT = "/home/mdb/DL_Lab3/RIVAS_DATASET/DATASET"

base_dir = Path(TRANSFER_ROOT)
global_query_xml = Path(KAGGLE_ROOT) / "query_label.xml"
dataset_dir = Path(RIVAS_ROOT)
results_dir = base_dir / "FINAL"


###################################### MODELOS
MODEL_CONFIGS = {
    'resnet50': {
        'models': ['resnet50', 'resnet50_ibn_a'],
        'weights': {
            'resnet50': '/home/mdb/reid-strong-baseline/configs/resnet50-19c8e357 (1).pth',
            'resnet50_ibn_a': '/home/mdb/reid-strong-baseline/configs/r50_ibn_a.pth'
        }
    },
    'seresnet50': {
        'models': ['se_resnext50', 'se_resnet50'],
        'weights': {
            'se_resnext50': '/home/mdb/reid-strong-baseline/configs/se_resnext50_32x4d-a260b3a4.pth',
            'se_resnet50': '/home/mdb/reid-strong-baseline/configs/se_resnet50-ce0d4300.pth'
        }
    },
    'seresnet50_IBN': {
        'models': ['se_resnext50', 'resnet50_ibn_a'],
        'weights': {
            'se_resnext50': '/home/mdb/reid-strong-baseline/configs/se_resnext50_32x4d-a260b3a4.pth',
            'resnet50_ibn_a': '/home/mdb/reid-strong-baseline/configs/r50_ibn_a.pth'
        }
    },
    'resnet101': {
        'models': ['se_resnet101', 'se_resnext101'],
        'weights': {
            'se_resnet101': '/home/mdb/reid-strong-baseline/configs/se_resnet101-7e38fcc6.pth',
            'se_resnext101': '/home/mdb/reid-strong-baseline/configs/se_resnext101_32x4d-3b2fe3d8.pth'
        }
    },
    'resnet_50_101': {
        'models': ['se_resnext50', 'se_resnext101'],
        'weights': {
            'se_resnext50': '/home/mdb/reid-strong-baseline/configs/se_resnext50_32x4d-a260b3a4.pth',
            'se_resnext101': '/home/mdb/reid-strong-baseline/configs/se_resnext101_32x4d-3b2fe3d8.pth'
        }
    },
    'resnet34': {
        'models': ['resnet34', 'resnet34'],
        'weights': {
            'resnet34': '/home/mdb/reid-strong-baseline/configs/resnet34-333f7ec4.pth',
            'resnet34': '/home/mdb/reid-strong-baseline/configs/resnet34-333f7ec4.pth'
        }
    },
    'resnet18': {
        'models': ['resnet18', 'resnet18'],
        'weights': {
            'resnet18': '/home/mdb/reid-strong-baseline/configs/resnet18-5c106cde.pth',
            'resnet18': '/home/mdb/reid-strong-baseline/configs/resnet18-5c106cde.pth'
        }
    }
}

#################################### DATASETS
DATASET_CONFIGS = {
    'original_sr_st': {
        "base": {
            "UAM": "/home/mdb/DL_Lab3/UAM_DATASET/stratified/{}/",
            "RIVAS": "/home/mdb/DL_Lab3/RIVAS_DATASET/DATASET/{}/",
        },
        "sr": {
            "UAM": "/home/mdb/DL_Lab3/ST_DATASETS/UAM_SR/{}/",
            "RIVAS": "/home/mdb/DL_Lab3/ST_DATASETS/RIVAS_SR/{}/",
        },
        "sr_st": {
            "UAM": "/home/mdb/DL_Lab3/ST_DATASETS/UAM_SR_ST/{}/",
            "RIVAS": "/home/mdb/DL_Lab3/ST_DATASETS/RIVAS_SR_ST/{}/",
        }
    },
    'augmv1_augmv2': {
        "base": {
            "UAM": "/home/mdb/DL_Lab3/UAM_DATASET/stratified/{}/",
            "RIVAS": "/home/mdb/DL_Lab3/RIVAS_DATASET/DATASET/{}/",
        },
        "augm1": {
            "UAM": "/home/mdb/DL_Lab3/ST_DATASETS/UAM_AUGM/{}/",
            "RIVAS": "/home/mdb/DL_Lab3/ST_DATASETS/RIVAS_AUGM/{}/",
        },
        "augm2": {
            "UAM": "/home/mdb/DL_Lab3/ST_DATASETS/UAM_AUGM2/{}/",
            "RIVAS": "/home/mdb/DL_Lab3/ST_DATASETS/RIVAS_AUGM2/{}/",
        }
    },
    'original_3x': {
        "base": {
            "UAM": "/home/mdb/DL_Lab3/UAM_DATASET/stratified/{}/",
            "RIVAS": "/home/mdb/DL_Lab3/RIVAS_DATASET/DATASET/{}/",
        },
        "base2": {
            "UAM": "/home/mdb/DL_Lab3/UAM_DATASET/stratified/{}/",
            "RIVAS": "/home/mdb/DL_Lab3/RIVAS_DATASET/DATASET/{}/",
        },
        "base3": {
            "UAM": "/home/mdb/DL_Lab3/UAM_DATASET/stratified/{}/",
            "RIVAS": "/home/mdb/DL_Lab3/RIVAS_DATASET/DATASET/{}/",
        }
    }
}



# ================== Rutas unificadas ==================
UNIFIED_DATASET_PATHS = {
    'original_3x': {
        "base": {
            "UAM": "/home/mdb/DL_Lab3/ST_DATASETS/UAM/Unified/",
            "RIVAS": "/home/mdb/DL_Lab3/ST_DATASETS/RIVAS/Unified/",
        },
        "base2": {
            "UAM": "/home/mdb/DL_Lab3/ST_DATASETS/UAM/Unified/",
            "RIVAS": "/home/mdb/DL_Lab3/ST_DATASETS/RIVAS/Unified/",
        },
        "base3": {
            "UAM": "/home/mdb/DL_Lab3/ST_DATASETS/UAM/Unified/",
            "RIVAS": "/home/mdb/DL_Lab3/ST_DATASETS/RIVAS/Unified/",
        }
    }
}

# ================== Modificación de DATASET_CONFIGS si se usa el modo unificado ==================
if USE_UNIFIED_MODEL:
    DATASET_CONFIGS = UNIFIED_DATASET_PATHS
    CLASSES = ["Unified"]


MODEL_NAMES = MODEL_CONFIGS[USE_MODEL_PAIR]['models']
PREWEIGHTS = MODEL_CONFIGS[USE_MODEL_PAIR]['weights']
DATASET_PAIRS = DATASET_CONFIGS[USE_DATASET_SET]
BASE_YAML_TEMPLATE = {
    'MODEL': {
        'PRETRAIN_CHOICE': 'imagenet',
        'NAME': None,
        'PRETRAIN_PATH': None,
        'METRIC_LOSS_TYPE': 'triplet_center',  # Cambiar de 'triplet_center' a 'quadruplet'
        'IF_LABELSMOOTH': 'on',
        'IF_WITH_CENTER': 'yes',            # Center loss NO compatible con quadruplet (habria que implementarlo)
    },
    'INPUT': {
        'SIZE_TRAIN': [256, 256],
        'SIZE_TEST': [256, 256],
        'PROB': 0.65,                 # Mod el 4 de mayo de 0.5 a 0.65
        'RE_PROB': 0.65,              # Mod el 4 de mayo de 0.5 a 0.65
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
        'MAX_EPOCHS': 30,
        'BASE_LR': 0.00035,
        'CENTER_LR': 0.5,
        'CENTER_LOSS_WEIGHT': 0.0005,      
        'IMS_PER_BATCH': 64,
        'STEPS': [20, 40],          # Mod el 4 de mayo de 30 45 a 20 40
        'GAMMA': 0.5,               # Mod el 4 de mayo de 0.1 a 0.5
        'WEIGHT_DECAY': 0.0001,     # Mod el 4 de mayo de 0.0005 a 0.0001
        'WARMUP_FACTOR': 0.01,
        'WARMUP_ITERS': 10,
        'WARMUP_METHOD': 'linear'
        #'QUADRUPLET_MARGIN1': 0.3,         # Añadir márgenes específicos de quadruplet
        #'QUADRUPLET_MARGIN2': 0.3
    },
    
    'TEST': {
        'IMS_PER_BATCH': 128,
        'RE_RANKING': 'yes',          # Mod el 4 de mayo de no a yes
        'NECK_FEAT': 'after',
        'FEAT_NORM': 'yes'
    }
}

# ============================================================================================================ Funciones auxiliares ==================

def run_training(yaml_path):
    try:
        subprocess.run(["python", "tools/train.py", "--config_file", yaml_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        print(f"[WARNING] Error en entrenamiento con {yaml_path}, continuando...")

def run_update(yaml_path, track_path):
    try:
        subprocess.run(["python", "tools/update.py", "--config_file", yaml_path, "--track", track_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        print(f"[WARNING] Error en update con {yaml_path}, continuando...")

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

# ================== Entrenamiento y Update ==================
os.makedirs(TRANSFER_ROOT, exist_ok=True)

for class_name in CLASSES:
    for model_name in MODEL_NAMES:
        for pair_name, paths in DATASET_PAIRS.items():
            print(f"\n================== Entrenando: {class_name} | {model_name} | {pair_name} ==================")

            class_model_root = os.path.join(TRANSFER_ROOT, f"{class_name}_{model_name}_{pair_name}")
            uam_output_dir = os.path.join(class_model_root, "UAM")
            rivas_output_dir = os.path.join(class_model_root, "RIVAS")
            os.makedirs(uam_output_dir, exist_ok=True)
            os.makedirs(rivas_output_dir, exist_ok=True)

            uam_yaml = copy.deepcopy(BASE_YAML_TEMPLATE)
            uam_yaml['MODEL']['NAME'] = model_name
            uam_yaml['MODEL']['PRETRAIN_PATH'] = PREWEIGHTS[model_name]
            uam_yaml['DATASETS']['ROOT_DIR'] = paths["UAM"].format(class_name)
            uam_yaml['OUTPUT_DIR'] = uam_output_dir
            uam_yaml['SOLVER']['STEPS'] = [20, 40]   # Mod el 4 de mayo : añadido
            
            if METRIC_LOSS_TYPE == 'quadruplet':
                uam_yaml['MODEL']['IF_WITH_CENTER'] = 'no'  # No usar center
                uam_yaml['MODEL']['METRIC_LOSS_TYPE'] = 'quadruplet'
                uam_yaml['SOLVER']['QUADRUPLET_MARGIN1'] = 0.3
                uam_yaml['SOLVER']['QUADRUPLET_MARGIN2'] = 0.3
            
            # Ajustar tamaño de imagen y parámetros específicos por clase
            if class_name == "Rubish":
                uam_yaml['INPUT']['SIZE_TRAIN'] = [160, 224]
                uam_yaml['INPUT']['SIZE_TEST'] = [160, 224]
                uam_yaml['SOLVER']['BASE_LR'] = 0.00035
                uam_yaml['SOLVER']['IMS_PER_BATCH'] = 64                    # Mod el 4 de mayo, de 64 a 128
                uam_yaml['SOLVER']['CENTER_LOSS_WEIGHT'] = 0.0005
                uam_yaml['SOLVER']['WARMUP_ITERS'] = 20
            elif class_name == "Crosswalks":
                uam_yaml['INPUT']['SIZE_TRAIN'] = [320, 96]
                uam_yaml['INPUT']['SIZE_TEST'] = [320, 96]
                uam_yaml['SOLVER']['BASE_LR'] = 0.00035
                uam_yaml['SOLVER']['IMS_PER_BATCH'] = 64                    # Mod el 4 de mayo, de 64 a 128
                uam_yaml['SOLVER']['CENTER_LOSS_WEIGHT'] = 0.0005
                uam_yaml['SOLVER']['WARMUP_ITERS'] = 20
            elif class_name == "Containers":
                uam_yaml['INPUT']['SIZE_TRAIN'] = [128, 160]
                uam_yaml['INPUT']['SIZE_TEST'] = [128, 160]
                uam_yaml['SOLVER']['BASE_LR'] = 0.00035
                uam_yaml['SOLVER']['IMS_PER_BATCH'] = 64                    # Mod el 4 de mayo, de 64 a 128
                uam_yaml['SOLVER']['CENTER_LOSS_WEIGHT'] = 0.0005
                uam_yaml['SOLVER']['WARMUP_ITERS'] = 20
            
            # Parámetros comunes mejorados para todas las clases
            uam_yaml['SOLVER']['MAX_EPOCHS'] = 20  #60       # Subimos un poco para datasets pequeños
            uam_yaml['INPUT']['RE_PROB'] = 0.75  # Mod el 6 de mayo de .5 a .75          # Más augmentation
            uam_yaml['TEST']['IMS_PER_BATCH'] = 64      # Mantener batch test alto para estabilidad
            


            uam_yaml_path = os.path.join(uam_output_dir, "train_uam.yaml")
            with open(uam_yaml_path, 'w') as f:
                yaml.dump(uam_yaml, f)

            run_training(uam_yaml_path)
            
            # ================== Evaluar UAM ==================
            try:
                latest_ckpt = sorted([f for f in os.listdir(uam_output_dir) if f.endswith('.pt')],
                                     key=lambda x: os.path.getmtime(os.path.join(uam_output_dir, x)), reverse=True)[0]
                pretrained_uam_ckpt = os.path.join(uam_output_dir, latest_ckpt)
            
                # Crear yaml de test UAM
                update_uam_yaml = copy.deepcopy(uam_yaml)
                update_uam_yaml['MODEL']['PRETRAIN_CHOICE'] = 'self'
                update_uam_yaml['DATASETS']['NAMES'] = ['UAM_test']
                update_uam_yaml['TEST']['WEIGHT'] = pretrained_uam_ckpt
            
                update_uam_path = os.path.join(uam_output_dir, "update_uam.yaml")
                with open(update_uam_path, "w") as f:
                    yaml.dump(update_uam_yaml, f)
            
                track_path_uam = os.path.join(uam_output_dir, "track.txt")
                # Lanzamos update sin salida
                run_update(update_uam_path, track_path_uam)
            
                # Evaluar y mostrar resultado
                eval_proc = subprocess.run([
                    "python", "Evaluate_UrbAM-ReID.py",
                    "--track", track_path_uam,
                    "--path", paths["UAM"].format(class_name)
                ], text=True, capture_output=True)
            
                print(f"================== EVALUACIÓN UAM | {class_name} | {model_name} | {pair_name} ==================")
                print(eval_proc.stdout)
            
            except Exception as e:
                print(f"[ERROR] No se pudo evaluar en UAM: {e}")

            
            try:
                latest_ckpt = sorted([f for f in os.listdir(uam_output_dir) if f.endswith('.pt')], key=lambda x: os.path.getmtime(os.path.join(uam_output_dir, x)), reverse=True)[0]
                pretrained_uam_ckpt = os.path.join(uam_output_dir, latest_ckpt)
            except IndexError:
                continue

            rivas_yaml = copy.deepcopy(BASE_YAML_TEMPLATE)
            rivas_yaml['MODEL']['NAME'] = model_name
            rivas_yaml['MODEL']['PRETRAIN_PATH'] = pretrained_uam_ckpt
            rivas_yaml['MODEL']['PRETRAIN_CHOICE'] = 'self'
            rivas_yaml['DATASETS']['ROOT_DIR'] = paths["RIVAS"].format(class_name)
            rivas_yaml['OUTPUT_DIR'] = rivas_output_dir
            rivas_yaml['SOLVER']['STEPS'] = [10, 25]   # Mod el 4 de mayo : añadido
            
            if METRIC_LOSS_TYPE == 'quadruplet':
                rivas_yaml['MODEL']['IF_WITH_CENTER'] = 'no'
                rivas_yaml['MODEL']['METRIC_LOSS_TYPE'] = 'quadruplet'
                rivas_yaml['SOLVER']['QUADRUPLET_MARGIN1'] = 0.3
                rivas_yaml['SOLVER']['QUADRUPLET_MARGIN2'] = 0.3
            
            # Ajustar tamaño de imagen según clase (también para RIVAS)
            # Cambiado el base_LR a 0.0005 en vez de 0.00035
            # Cambiado size de containers de [128, 160] a [256, 320]
            # Cambiado ims per batch de 64 a 128
            if class_name == "Rubish":
                rivas_yaml['INPUT']['SIZE_TRAIN'] = [160, 224]
                rivas_yaml['INPUT']['SIZE_TEST'] = [160, 224]
                rivas_yaml['SOLVER']['IMS_PER_BATCH'] = 64
                rivas_yaml['SOLVER']['BASE_LR'] = 0.0005  
                rivas_yaml['SOLVER']['MAX_EPOCHS'] = 30 #30
                rivas_yaml['SOLVER']['CENTER_LOSS_WEIGHT'] = 0.0005
            elif class_name == "Crosswalks":
                rivas_yaml['INPUT']['SIZE_TRAIN'] = [320, 96]
                rivas_yaml['INPUT']['SIZE_TEST'] = [320, 96]
                rivas_yaml['SOLVER']['IMS_PER_BATCH'] = 64
                rivas_yaml['SOLVER']['BASE_LR'] = 0.0005 
                rivas_yaml['SOLVER']['MAX_EPOCHS'] = 30 #30
                rivas_yaml['SOLVER']['CENTER_LOSS_WEIGHT'] = 0.0005
            elif class_name == "Containers":
                rivas_yaml['INPUT']['SIZE_TRAIN'] = [256, 320]
                rivas_yaml['INPUT']['SIZE_TEST'] = [256, 320]
                rivas_yaml['SOLVER']['IMS_PER_BATCH'] = 64
                rivas_yaml['SOLVER']['BASE_LR'] = 0.0005 
                rivas_yaml['SOLVER']['MAX_EPOCHS'] = 30
                rivas_yaml['SOLVER']['CENTER_LOSS_WEIGHT'] = 0.0005
            else:
                rivas_yaml['INPUT']['SIZE_TRAIN'] = [256, 128]
                rivas_yaml['INPUT']['SIZE_TEST'] = [256, 128]
                rivas_yaml['SOLVER']['IMS_PER_BATCH'] = 64
                rivas_yaml['SOLVER']['BASE_LR'] = 0.0005  
                rivas_yaml['SOLVER']['MAX_EPOCHS'] = 30
                rivas_yaml['SOLVER']['CENTER_LOSS_WEIGHT'] = 0.0005
            
            # Hiperparámetros específicos para fine-tuning RIVAS
            rivas_yaml['SOLVER']['WARMUP_ITERS'] = 20     
            rivas_yaml['TEST']['IMS_PER_BATCH'] = 128	
            

            rivas_yaml_path = os.path.join(rivas_output_dir, "train_rivas.yaml")
            with open(rivas_yaml_path, 'w') as f:
                yaml.dump(rivas_yaml, f)

            run_training(rivas_yaml_path)
            
            try:
                for item in os.listdir(uam_output_dir):
                    item_path = os.path.join(uam_output_dir, item)
                    if os.path.isfile(item_path) and item not in ["track.txt", "qf.npy", "gf.npy"]:
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
            except Exception as e:
                print(f"[WARNING] No se pudo limpiar {uam_output_dir} (excepto track.txt, qf.npy, gf.npy): {e}")

            try:
                final_ckpt = sorted([f for f in os.listdir(rivas_output_dir) if f.endswith('.pt')], key=lambda x: os.path.getmtime(os.path.join(rivas_output_dir, x)), reverse=True)[0]
                final_ckpt_path = os.path.join(rivas_output_dir, final_ckpt)
            except IndexError:
                continue

            update_yaml = copy.deepcopy(rivas_yaml)
            update_yaml['DATASETS']['NAMES'] = ['UAM_test']
            update_yaml['TEST']['WEIGHT'] = final_ckpt_path

            update_yaml_path = os.path.join(rivas_output_dir, "update_rivas.yaml")
            with open(update_yaml_path, 'w') as f:
                yaml.dump(update_yaml, f)

            track_txt_path = os.path.join(rivas_output_dir, "track.txt")
            run_update(update_yaml_path, track_txt_path)
            

# ================== Fusión ResNet+IBN para cada dataset ==================
for class_name in CLASSES:
    for pair_name in DATASET_PAIRS.keys():
        for domain in ["RIVAS"]:
            print(f"\nFusionando Model 1 + Model 2 para {class_name} | {pair_name} | {domain}")
            
            dir1 = os.path.join(TRANSFER_ROOT, f"{class_name}_{MODEL_NAMES[0]}_{pair_name}", domain)
            dir2 = os.path.join(TRANSFER_ROOT, f"{class_name}_{MODEL_NAMES[1]}_{pair_name}", domain)

            output_dir = os.path.join(TRANSFER_ROOT, f"{class_name}_Combined_{pair_name}", domain, "RESULTS")
            os.makedirs(output_dir, exist_ok=True)
            
            try:
                qf1 = np.load(os.path.join(dir1, "qf.npy"))
                gf1 = np.load(os.path.join(dir1, "gf.npy"))
                qf2 = np.load(os.path.join(dir2, "qf.npy"))
                gf2 = np.load(os.path.join(dir2, "gf.npy"))
            except Exception as e:
                print(f"[WARNING] No se encontraron embeddings para fusión en {domain}: {e}")
                continue

            # Normalizar
            qf1, gf1 = l2norm(qf1), l2norm(gf1)
            qf2, gf2 = l2norm(qf2), l2norm(gf2)

            # ================== FUSIÓN POR MEDIA ==================
            qf_mean = l2norm((qf1 + qf2) / 2)
            gf_mean = l2norm((gf1 + gf2) / 2)

            q_g = np.dot(qf_mean, gf_mean.T)
            q_q = np.dot(qf_mean, qf_mean.T)
            g_g = np.dot(gf_mean, gf_mean.T)

            dist = re_ranking(q_g, q_q, g_g, k1=10, k2=6, lambda_value=0.1)
            indices = np.argsort(dist, axis=1)

            track_mean_path = os.path.join(output_dir, "track_mean.txt")
            with open(track_mean_path, 'w') as f:
                for row in indices:
                    adjusted = [str(i+1) for i in row[:100]]
                    f.write(" ".join(adjusted) + "\n")

            # ================== FUSIÓN POR CONCATENACIÓN ==================
            qf_concat = l2norm(np.concatenate([qf1, qf2], axis=1))
            gf_concat = l2norm(np.concatenate([gf1, gf2], axis=1))

            q_g = np.dot(qf_concat, gf_concat.T)
            q_q = np.dot(qf_concat, qf_concat.T)
            g_g = np.dot(gf_concat, gf_concat.T)

            dist = re_ranking(q_g, q_q, g_g, k1=10, k2=6, lambda_value=0.1)
            indices = np.argsort(dist, axis=1)

            track_concat_path = os.path.join(output_dir, "track_concat.txt")
            with open(track_concat_path, 'w') as f:
                for row in indices:
                    adjusted = [str(i+1) for i in row[:100]]
                    f.write(" ".join(adjusted) + "\n")


            # ================== EVALUAR SOLO SI ES UAM ==================
            if domain == "UAM":
                dataset_path = DATASET_PAIRS[pair_name]["UAM"].format(class_name)
                
                for track_path in [track_mean_path, track_concat_path]:
                    try:
                        print(f"Evaluando {os.path.basename(track_path)} en {domain}")
                        eval_proc = subprocess.run([
                            "python", "Evaluate_UrbAM-ReID.py",
                            "--track", track_path,
                            "--path", dataset_path
                        ], text=True, capture_output=True)
                        print(eval_proc.stdout)
                    except Exception as e:
                        print(f"[WARNING] Falló evaluación para {track_path}: {e}")


# ================== Fusión ponderada final entre datasets (track_mean y track_concat) ==================
for class_name in CLASSES:
    for domain in ["RIVAS"]:
        for variant in ["track_mean.txt", "track_concat.txt"]:
            print(f"\nFusionando datasets para {class_name} | {domain} | {variant}")

            tracks = [
                os.path.join(TRANSFER_ROOT, f"{class_name}_Combined_{subname}", domain, "RESULTS", variant)
                for subname in DATASET_PAIRS.keys()
                if os.path.exists(os.path.join(TRANSFER_ROOT, f"{class_name}_Combined_{subname}", domain, "RESULTS", variant))
            ]

            if not tracks:
                print(f"[WARNING] No hay tracks ({variant}) para {class_name} | {domain}, saltando...")
                continue

            output_fused = os.path.join(TRANSFER_ROOT, f"{class_name}_FINAL", domain, "RESULTS")
            os.makedirs(output_fused, exist_ok=True)

            fused_track_path = os.path.join(output_fused, variant)
            fusionar_tracks(tracks, fused_track_path)

            # Evaluación solo si es UAM
            if domain == "UAM":
                try:
                    dataset_path = DATASET_PAIRS[list(DATASET_PAIRS.keys())[0]]["UAM"].format(class_name)
                    print(f"Evaluando track fusionado FINAL en UAM para {class_name} | {variant}")
                    eval_proc = subprocess.run([
                        "python", "Evaluate_UrbAM-ReID.py",
                        "--track", fused_track_path,
                        "--path", dataset_path
                    ], text=True, capture_output=True)
                    print(eval_proc.stdout)
                except Exception as e:
                    print(f"[WARNING] Error al evaluar en UAM ({variant}) para {class_name}: {e}")

print("\n================== Todo completado ==================")

# ================== Generación de archivo de SUBMISSION para Kaggle ==================
print("\n================== Generando submission ==================")

# ================== FUNCIONES AUXILIARES ==================
def parse_items(xml_path):
    tree = ET.parse(xml_path)
    return [item.attrib["imageName"] for item in tree.getroot().findall("Item")]

def extract_number(filename):
    return str(int(filename.replace(".jpg", "").lstrip("0") or "0"))

# ================== Generación de archivo de SUBMISSION para Kaggle (track_mean y track_concat) ==================
print("\n================== Generando submission (mean y concat) ==================")

for variant in ["track_mean.txt", "track_concat.txt"]:
    print(f"\n--- Procesando variante: {variant} ---")

    # ================== SALIDAS ==================
    names_file = results_dir / f"Results_RIVAS_ALL_names_{variant.replace('.txt','')}.txt"
    indices_file = results_dir / f"Results_RIVAS_ALL_indices_{variant.replace('.txt','')}.txt"
    submission_file = results_dir / f"RESULT_SUBMISSION_{variant.replace('.txt','')}.txt"
    output_csv_path = results_dir / f"kaggle_submission_converted_1based_{variant.replace('.txt','')}.csv"

    # ================== PASO 1: GENERAR NOMBRES POR CLASE ==================
    all_entries = []

    for cls in CLASSES:
        dataset_dir = Path(RIVAS_ROOT)
        query_xml = dataset_dir / cls / "query_label.xml"
        test_xml = dataset_dir / cls / "test_label.xml"
        track_file = base_dir / f"{cls}_FINAL" / "RIVAS" / "RESULTS" / variant

        query_names = parse_items(query_xml)
        gallery_names = parse_items(test_xml)

        with open(track_file, "r") as f:
            lines = [line.strip().split() for line in f.readlines()]

        assert len(query_names) == len(lines), f"Mismatch en {cls}: {len(query_names)} queries vs {len(lines)} predicciones"

        for qname, pred_indices in zip(query_names, lines):
            gallery_preds = []
            for idx in pred_indices:
                try:
                    g_idx = int(idx) - 1  # ¡Recordar que son 1-based!
                    if 0 <= g_idx < len(gallery_names):
                        gallery_preds.append(gallery_names[g_idx])
                    else:
                        gallery_preds.append("unknown.jpg")
                except:
                    gallery_preds.append("error.jpg")
            all_entries.append((qname, gallery_preds))

    # ================== PASO 2: ORDEN GLOBAL Y GUARDAR NOMBRES ==================
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

    results_dir.mkdir(parents=True, exist_ok=True)
    with open(names_file, "w") as f:
        f.write("\n".join(final_lines))

    print(f"Guardado NOMBRES en: {names_file}")

    # ================== PASO 3: CONVERTIR A ÍNDICES NUMÉRICOS ==================
    with open(names_file, "r") as fin:
        lines = fin.readlines()

    with open(indices_file, "w") as fout:
        for line in lines:
            parts = line.strip().split()
            numbers = [extract_number(name) for name in parts]
            fout.write(" ".join(numbers) + "\n")

    print(f"Guardado ÍNDICES en: {indices_file}")

    # ================== PASO 4: ELIMINAR ÍNDICE DE QUERY PARA SUBMISSION FINAL ==================
    with open(indices_file, "r") as fin, open(submission_file, "w") as fout:
        for line in fin:
            parts = line.strip().split()
            fout.write(" ".join(parts[1:]) + "\n")  # Skip query name

    print(f"Guardado RESULT_SUBMISSION en: {submission_file}")

    # ================== PASO 5: CREAR CSV PARA KAGGLE SUBMISSION ==================
    with open(submission_file, "r") as f:
        prediction_lines = [line.strip().split() for line in f.readlines()]

    num_queries = len(global_query_names)
    assert len(prediction_lines) == num_queries, f"Esperados {num_queries} queries, pero hay {len(prediction_lines)}"

    converted_lines = []
    all_indices = []

    for i, preds in enumerate(prediction_lines):
        image_name = "{:06d}.jpg".format(i + 1)  # 1-based naming
        valid_preds = [str(int(x)) for x in preds if x.isdigit()]
        converted_lines.append([image_name, " ".join(valid_preds)])
        all_indices.extend(map(int, valid_preds))

    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["imageName", "Corresponding Indexes"])
        writer.writerows(converted_lines)

    min_idx = min(all_indices)
    max_idx = max(all_indices)

    print(f"CSV convertido para Kaggle guardado en: {output_csv_path}")
    print(f"Índices predichos: mínimo={min_idx} | máximo={max_idx}")

    # ================== AVISO FINAL ==================
    if missing:
        print(f"WARNING: {len(missing)} queries faltantes (ejemplo: {missing[:5]})")

import xml.etree.ElementTree as ET
import pandas as pd

if USE_UNIFIED_MODEL:
    print("\n================== Reordenando CSV para priorizar clase del query ==================")
    
    from xml.etree import ElementTree as ET
    import pandas as pd

    # Path al XML con predictedClass
    classified_xml_path = Path(KAGGLE_ROOT) / "test_label_classified.xml"
    tree = ET.parse(classified_xml_path)
    root = tree.getroot()

    # Mapear imageName -> class
    gallery_classes = {
        item.attrib["imageName"]: item.attrib["predictedClass"]
        for item in root.findall("Item")
    }

    # Invertido: index (int) -> class
    index_to_class = {
        int(name.replace(".jpg", "").lstrip("0") or "0"): cls
        for name, cls in gallery_classes.items()
    }

    for variant in ["track_mean", "track_concat"]:
        csv_path = results_dir / f"kaggle_submission_converted_1based_{variant}.csv"
        if not csv_path.exists():
            print(f"[SKIP] No encontrado: {csv_path}")
            continue

        print(f"Procesando: {csv_path.name}")

        df = pd.read_csv(csv_path)
        reordered_rows = []

        for _, row in df.iterrows():
            query_name = row["imageName"]
            pred_indices = [int(x) for x in row["Corresponding Indexes"].split()]
            query_class = gallery_classes.get(query_name)

            if query_class:
                same_class = [idx for idx in pred_indices if index_to_class.get(idx) == query_class]
                other_class = [idx for idx in pred_indices if index_to_class.get(idx) != query_class]
                final_preds = same_class + other_class
            else:
                final_preds = pred_indices  # fallback sin cambio

            reordered_rows.append([query_name, " ".join(str(i) for i in final_preds[:100])])

        # Guardar CSV reordenado
        output_path = csv_path.with_name(csv_path.stem + ".csv")
        pd.DataFrame(reordered_rows, columns=["imageName", "Corresponding Indexes"]).to_csv(output_path, index=False)

    print("\n================== Finalizado ==================")

else:
    print("\n================== Finalizado ==================")
