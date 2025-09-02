#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 14:53:21 2025

@author: mdb
"""

###########################################################################################################################
# ================== Combination of model(s) and dataset(s)  ==================
USE_MODEL_PAIR = 'se_resnet50'         # Some options: 'resnet50', 'seresnet50', 'resnet101', 'seresnet50_IBN'. 'resnet_50_101 NOTE: ADD MODELS EASILY BELOW BY ADDING THE PATH
USE_DATASET_SET = 'All_combined_Augm'   # Some opction: 'original_sr_st', 'augmv1_augmv2', 'original_3x' NOTE: ADD MODELS EASILY BELOW BY ADDING THE PATH
METRIC_LOSS_TYPE = 'triplet_center'  # options: 'triplet_center', 'quadruplet'
USE_MODEL_TYPE = 'BoT'  # Opctions: 'BoT', 'PaT'
USE_UNIFIED_MODEL = True  # True is for unified, False for classes aware training
previous_fine_tuning = False  # True is for fine tuning before on "campus dataset", to avoid it set this to FALSE
training_on_UAM = False
specific_size = False
rerank_class = True
epochs = 85
verbosity = 1 # If you want to see the evolution of the process in the terminal set it to 1
###################################################


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
import pandas as pd
import torch
import random


# REPRODUCIBILITY
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ================== General config ==================
CLASSES = ["Containers", "Rubish", "Crosswalks"]
TRANSFER_ROOT = "/home/mdb/reid-strong-baseline/TRANSFER_Learning_SR_ensembling_BestEnsemb_Repeated/"
KAGGLE_ROOT = "/home/mdb/DL_Lab3/Kaggle_Dataset"
RIVAS_ROOT = "/home/mdb/DL_Lab3/RIVAS_DATASET/DATASET"

base_dir = Path(TRANSFER_ROOT)
global_query_xml = Path(KAGGLE_ROOT) / "query_label.xml"
dataset_dir = Path(RIVAS_ROOT)
results_dir = base_dir / "FINAL"


TRANSFER_ROOT = "/home/mdb/reid-strong-baseline/TRANSFER_Learning_SR_ensembling_BestEnsemb_Repeated/"
TRAIN_SCRIPT = "tools/train.py"
UPDATE_SCRIPT = "tools/update.py"



###################################### MODELS Selection
MODEL_CONFIGS = {
    'resnet50': {
        'models': ['resnet50', 'resnet50_ibn_a'],
        'weights': {
            'resnet50': '/home/mdb/reid-strong-baseline/configs/resnet50-19c8e357 (1).pth',
            'resnet50_ibn_a': '/home/mdb/reid-strong-baseline/configs/r50_ibn_a.pth'
        }
    },
    'best': {
        'models': ['resnet50', 'se_resnet50'],
        'weights': {
            'resnet50': '/home/mdb/reid-strong-baseline/configs/resnet50-19c8e357 (1).pth',
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
    }, 
    
    'senet154': {
        'models': ['senet154'],
        'weights': {
            'senet154': '/home/mdb/reid-strong-baseline/configs/legacy_senet154-e9eb9fe6.pth'
        }
    }, 
    'baseline': {
        'models': ['resnet50'],
        'weights': {
            'resnet50': '/home/mdb/reid-strong-baseline/configs/resnet50-19c8e357 (1).pth',
            }
    }, 
    
    'baselinex2': {
        'models': ['resnet50', 'resnet50_ibn_a'],
        'weights': {
            'resnet50': '/home/mdb/reid-strong-baseline/configs/resnet50-19c8e357 (1).pth',
            'resnet50_ibn_a': '/home/mdb/reid-strong-baseline/configs/r50_ibn_a.pth'
        }
    }, 
    
    'se_resnet50': {
        'models': ['se_resnet50'],
        'weights': {
            'se_resnet50': '/home/mdb/reid-strong-baseline/configs/se_resnet50-ce0d4300.pth'
            }
    },
    'Final': {
        'models': ['se_resnet101', 'se_resnet50'],
        'weights': {
            'se_resnet101': '/home/mdb/reid-strong-baseline/configs/se_resnet101-7e38fcc6.pth',
            'se_resnet50': '/home/mdb/reid-strong-baseline/configs/se_resnet50-ce0d4300.pth'
            }
    },
}

#################################### DATASETS Selection
DATASET_CONFIGS = {
    'original': {
        "base": {
            "UAM": "/home/mdb/DL_Lab3/UAM_DATASET/stratified/{}/",
            "RIVAS": "/home/mdb/DL_Lab3/RIVAS_DATASET/DATASET/{}/",
        }
    },
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
    },
    'Orig_SR': {
        "Mix": {
            "UAM": "/home/mdb/DL_Lab3/ST_DATASETS/UAM_SR_COMBINED/{}/",
            "RIVAS": "/home/mdb/DL_Lab3/ST_DATASETS/RIVAS_SR_COMBINED/{}/",
            }
        }
}



# ================== Unified paths (without class aware) ==================
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
    },
    'original': {
        "base": {
            "UAM": "/home/mdb/DL_Lab3/ST_DATASETS/UAM/Unified/",
            "RIVAS": "/home/mdb/DL_Lab3/ST_DATASETS/RIVAS/Unified/",
        }
    },
    'Super_Resolution': {
        "base": {
            "UAM": "/home/mdb/DL_Lab3/ST_DATASETS/UAM_SR/Unified/",
            "RIVAS": "/home/mdb/DL_Lab3/ST_DATASETS/RIVAS_SR/Unified/",
        }
    },
    'Style_Transfer': {
        "base": {
            "UAM": "/home/mdb/DL_Lab3/ST_DATASETS/UAM_SR_ST/Unified/",
            "RIVAS": "/home/mdb/DL_Lab3/ST_DATASETS/RIVAS_SR_ST/Unified/",
        }
    },
    'Augm': {
        "base": {
            "UAM": "/home/mdb/DL_Lab3/ST_DATASETS/UAM_AUGM/Unified/",
            "RIVAS": "/home/mdb/DL_Lab3/ST_DATASETS/RIVAS_AUGM/Unified/",
        }
    },
    'Combined_simple': {
        "base": {
            "UAM": "/home/mdb/DL_Lab3/ST_DATASETS/COMBINED_DATASET/Unified/",
            "RIVAS": "/home/mdb/DL_Lab3/ST_DATASETS/COMBINED_DATASET/Unified/",
        }
    },
    'Combined_tech': {
        "base": {
            "UAM": "/home/mdb/DL_Lab3/ST_DATASETS/ALL_COMBINED_UAM/Unified/",
            "RIVAS": "/home/mdb/DL_Lab3/ST_DATASETS/ALL_COMBINED_RIVAS/Unified/",
        }
    },
    'All_combined': {
        "base": {
            "UAM": "/home/mdb/DL_Lab3/ST_DATASETS/ALL_COMBINED_DATASET/Unified/",
            "RIVAS": "/home/mdb/DL_Lab3/ST_DATASETS/ALL_COMBINED_DATASET/Unified/",
        }
    },
    'All_combined_Augm': {
        "base": {
            "UAM": "/home/mdb/DL_Lab3/ST_DATASETS/ALL_COMBINED_DATASET_AUGM/Unified/",
            "RIVAS": "/home/mdb/DL_Lab3/ST_DATASETS/ALL_COMBINED_DATASET_AUGM/Unified/",
        }
    },
    'Best_DATA': {
        "All_combined_Augm": {
            "UAM": "/home/mdb/DL_Lab3/ST_DATASETS/ALL_COMBINED_DATASET_AUGM/Unified/",
            "RIVAS": "/home/mdb/DL_Lab3/ST_DATASETS/ALL_COMBINED_DATASET_AUGM/Unified/",
        },
        "Combined_tech": {
            "UAM": "/home/mdb/DL_Lab3/ST_DATASETS/ALL_COMBINED_UAM/Unified/",
            "RIVAS": "/home/mdb/DL_Lab3/ST_DATASETS/ALL_COMBINED_RIVAS/Unified/",
        }
    },
    
}



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
        'METRIC_LOSS_TYPE': 'triplet_center',  
        'IF_LABELSMOOTH': 'on',
        'IF_WITH_CENTER': 'yes',            # Center loss NOT compatible with quadruplet (needs implementation)
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
        'NUM_WORKERS': 8,
    },
    'SOLVER': {
        'OPTIMIZER_NAME': 'Adam',
        'MAX_EPOCHS': epochs,
        'BASE_LR': 0.00035,                
        'CENTER_LR': 0.5,
        'CLUSTER_MARGIN': 0.3,
        'CENTER_LOSS_WEIGHT': 0.0005,  
        
        'RANGE_K': 2,
        'RANGE_MARGIN': 0.3,
        'RANGE_ALPHA': 0,
        'RANGE_BETA': 1.0,
        'RANGE_LOSS_WEIGHT': 1,
        'BIAS_LR_FACTOR': 1,
        'WEIGHT_DECAY': 0.0005,
        'WEIGHT_DECAY_BIAS': 0.0005,
        'IMS_PER_BATCH': 64,                
        
        'STEPS': [40, 70],
        'GAMMA': 0.1,
        
        'WARMUP_FACTOR': 0.01,
        'WARMUP_ITERS': 10,
        'WARMUP_METHOD': 'linear',
        
        'CHECKPOINT_PERIOD': 10,
        'LOG_PERIOD': 5,
        'EVAL_PERIOD': 1005,
    },
    
    'TEST': {
        'IMS_PER_BATCH': 128,
        'RE_RANKING': 'no',        
        'NECK_FEAT': 'after'
    }
}

# ============================================================================================================ Aux functions ==================

def run_training(config_path):
    if verbosity == 0:
        try:
            subprocess.run(["python", TRAIN_SCRIPT, "--config_file", config_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            print(f"[WARNING] Error en entrenamiento con {config_path}, continuando...")
    else: 
        try:
            subprocess.run(["python", TRAIN_SCRIPT, "--config_file", config_path])
        except subprocess.CalledProcessError:
            print(f"[WARNING] Error en entrenamiento con {config_path}, continuando...")

def run_update(config_path, track_path):
    if verbosity == 0:
        try:
            subprocess.run(["python", UPDATE_SCRIPT, "--config_file", config_path, "--track", track_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            print(f"[WARNING] Error en update con {config_path}, continuando...")
    else: 
        try:
            subprocess.run(["python", UPDATE_SCRIPT, "--config_file", config_path, "--track", track_path])
        except subprocess.CalledProcessError:
            print(f"[WARNING] Error en update con {config_path}, continuando...")


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

# ================== Train and Update ==================
os.makedirs(TRANSFER_ROOT, exist_ok=True)

for class_name in CLASSES:
    for model_name in MODEL_NAMES:
        for pair_name, paths in DATASET_PAIRS.items():
            print(f"\n================== Entrenando: {class_name} | {model_name} | {pair_name} ==================")
            if USE_MODEL_TYPE == 'PaT':
                yaml_model_section = {
                    'NAME': 'part_attention_vit',
                    'TRANSFORMER_TYPE': model_name,
                    'STRIDE_SIZE': [12, 12],
                    'PRETRAIN_CHOICE': 'imagenet',
                    'METRIC_LOSS_TYPE': METRIC_LOSS_TYPE,
                    'IF_LABELSMOOTH': 'on',
                    'IF_WITH_CENTER': 'yes'
                }
            else:
                yaml_model_section = {
                    'NAME': model_name,
                    'PRETRAIN_PATH': PREWEIGHTS[model_name],
                    'PRETRAIN_CHOICE': 'imagenet',
                    'METRIC_LOSS_TYPE': METRIC_LOSS_TYPE,
                    'IF_LABELSMOOTH': 'on',
                    'IF_WITH_CENTER': 'yes'
                }

            class_model_root = os.path.join(TRANSFER_ROOT, f"{class_name}_{model_name}_{pair_name}")
            rivas_output_dir = os.path.join(class_model_root, "RIVAS")
            
            os.makedirs(rivas_output_dir, exist_ok=True)


            if training_on_UAM:
                uam_output_dir = os.path.join(class_model_root, "UAM")
                os.makedirs(uam_output_dir, exist_ok=True)
                uam_yaml = copy.deepcopy(BASE_YAML_TEMPLATE)
                uam_yaml['MODEL'] = yaml_model_section
                uam_yaml['MODEL']['PRETRAIN_PATH'] = PREWEIGHTS[model_name]
                uam_yaml['DATASETS']['ROOT_DIR'] = paths["UAM"].format(class_name)
                uam_yaml['OUTPUT_DIR'] = uam_output_dir
     
                if METRIC_LOSS_TYPE == 'quadruplet':
                    uam_yaml['MODEL']['IF_WITH_CENTER'] = 'no'  
                    uam_yaml['MODEL']['METRIC_LOSS_TYPE'] = 'quadruplet'
                    uam_yaml['SOLVER']['QUADRUPLET_MARGIN1'] = 0.3
                    uam_yaml['SOLVER']['QUADRUPLET_MARGIN2'] = 0.3
                
                if specific_size:
                    if class_name == "Rubish":
                        uam_yaml['INPUT']['SIZE_TRAIN'] = [160, 224]
                        uam_yaml['INPUT']['SIZE_TEST'] = [160, 224]

                    elif class_name == "Crosswalks":
                        uam_yaml['INPUT']['SIZE_TRAIN'] = [320, 96]
                        uam_yaml['INPUT']['SIZE_TEST'] = [320, 96]
                  
                    elif class_name == "Containers":
                        uam_yaml['INPUT']['SIZE_TRAIN'] = [128, 160]
                        uam_yaml['INPUT']['SIZE_TEST'] = [128, 160]
                    
                    
                # ================== Specific adjusts for PAT ==================
                if USE_MODEL_TYPE == 'PaT':
                    # Sustituye 'NAMES' por 'TRAIN' y define 'TEST'
                    uam_yaml['DATASETS']['TRAIN'] = uam_yaml['DATASETS'].pop('NAMES', ['UAM'])
                    uam_yaml['DATASETS']['TEST'] = ['UAM_test']
                    uam_yaml['OUTPUT_DIR'] = uam_output_dir
                    uam_yaml['LOG_ROOT'] = uam_output_dir  # obligatorio para PaT
                    uam_yaml['TB_LOG_ROOT'] = os.path.join(uam_output_dir, 'tb')  # o cualquier path válido
                    
                    if 'WARMUP_ITERS' in uam_yaml['SOLVER']:
                        uam_yaml['SOLVER']['WARMUP_EPOCHS'] = uam_yaml['SOLVER'].pop('WARMUP_ITERS')
                        
                    uam_yaml['TEST']['RE_RANKING'] = uam_yaml['TEST']['RE_RANKING'].lower() == 'yes'
                    #uam_yaml['TEST']['FEAT_NORM'] = uam_yaml['TEST']['FEAT_NORM'].lower() == 'True'
    
                uam_yaml_path = os.path.join(uam_output_dir, "train_uam.yaml")
                with open(uam_yaml_path, 'w') as f:
                    yaml.dump(uam_yaml, f)
    
                run_training(uam_yaml_path)
                
                # ================== Evaluate in UAM ==================
                try:
                    if USE_MODEL_TYPE == 'PaT':
                        latest_ckpt = sorted([f for f in os.listdir(uam_output_dir) if f.endswith('.pth')],
                                             key=lambda x: os.path.getmtime(os.path.join(uam_output_dir, x)), reverse=True)[0]
                        pretrained_uam_ckpt = os.path.join(uam_output_dir, latest_ckpt)
                    else:
                        latest_ckpt = sorted([f for f in os.listdir(uam_output_dir) if f.endswith('.pt')],
                                             key=lambda x: os.path.getmtime(os.path.join(uam_output_dir, x)), reverse=True)[0]
                        pretrained_uam_ckpt = os.path.join(uam_output_dir, latest_ckpt)
                
                    update_uam_yaml = copy.deepcopy(uam_yaml)
                    update_uam_yaml['MODEL']['PRETRAIN_CHOICE'] = 'self'
                    update_uam_yaml['TEST']['WEIGHT'] = pretrained_uam_ckpt
                    
                    if USE_MODEL_TYPE == 'PaT':
                        update_uam_yaml['DATASETS']['TEST'] = ['UAM_test']
                        if 'NAMES' in update_uam_yaml['DATASETS']:
                            del update_uam_yaml['DATASETS']['NAMES']
                        if isinstance(update_uam_yaml['TEST'].get('RE_RANKING', False), str):
                            update_uam_yaml['TEST']['RE_RANKING'] = update_uam_yaml['TEST']['RE_RANKING'].lower() == 'yes'
                    else:
                        update_uam_yaml['DATASETS']['NAMES'] = ['UAM_test']
    
                
                    update_uam_path = os.path.join(uam_output_dir, "update_uam.yaml")
                    with open(update_uam_path, "w") as f:
                        yaml.dump(update_uam_yaml, f)
                
                    track_path_uam = os.path.join(uam_output_dir, "track.txt")
                    # Lanzamos update sin salida
                    run_update(update_uam_path, track_path_uam)
                
                    eval_proc = subprocess.run([
                        "python", "Evaluate_UrbAM-ReID.py",
                        "--track", track_path_uam,
                        "--path", paths["UAM"].format(class_name)
                    ], text=True, capture_output=True)
                
                    print(f"================== EVALUACIÓN UAM | {class_name} | {model_name} | {pair_name} ==================")
                    print(eval_proc.stdout)
                
                except Exception as e:
                    print(f"[ERROR] No se pudo evaluar en UAM: {e}")

            
            if previous_fine_tuning:
                try:
                    if USE_MODEL_TYPE == 'PaT':
                        latest_ckpt = sorted(
                            [f for f in os.listdir(uam_output_dir) if f.endswith('.pth')],
                            key=lambda x: os.path.getmtime(os.path.join(uam_output_dir, f)),
                            reverse=True
                        )[0]
                    else:
                        latest_ckpt = sorted(
                            [f for f in os.listdir(uam_output_dir) if f.endswith('.pt')],
                            key=lambda x: os.path.getmtime(os.path.join(uam_output_dir, f)),
                            reverse=True
                        )[0]
                    pretrained_uam_ckpt = os.path.join(uam_output_dir, latest_ckpt)
                except IndexError:
                    print(f"[WARNING] No se encontró checkpoint en {uam_output_dir}, usando pesos por defecto.")
                    pretrained_uam_ckpt = PREWEIGHTS[model_name]
            else:
                pretrained_uam_ckpt = PREWEIGHTS[model_name]

            
            rivas_yaml = copy.deepcopy(BASE_YAML_TEMPLATE)
            rivas_yaml['MODEL'] = yaml_model_section
            # Establecer pesos pretrained correctamente
            rivas_yaml['MODEL']['PRETRAIN_PATH'] = pretrained_uam_ckpt  # <- necesario
            rivas_yaml['MODEL']['PRETRAIN_CHOICE'] = 'self' if previous_fine_tuning else 'imagenet'
            
            
            rivas_yaml['DATASETS']['ROOT_DIR'] = paths["RIVAS"].format(class_name)
            
            rivas_yaml['OUTPUT_DIR'] = rivas_output_dir
        
            if USE_MODEL_TYPE == 'PaT':
                rivas_yaml['LOG_NAME'] = rivas_output_dir
                rivas_yaml['DATASETS']['TRAIN'] = ['UAM',]
                rivas_yaml['DATASETS']['TEST'] = ['UAM',]
                rivas_yaml['LOG_ROOT'] = rivas_output_dir
                rivas_yaml['TB_LOG_ROOT'] = os.path.join(rivas_output_dir, 'tb')
            
                for key in ['RE_RANKING', 'FEAT_NORM']:
                    if isinstance(rivas_yaml['TEST'].get(key), str):
                        rivas_yaml['TEST'][key] = rivas_yaml['TEST'][key].lower() == 'True'
                            
            if METRIC_LOSS_TYPE == 'quadruplet':
                rivas_yaml['MODEL']['IF_WITH_CENTER'] = 'no'
                rivas_yaml['MODEL']['METRIC_LOSS_TYPE'] = 'quadruplet'
                rivas_yaml['SOLVER']['QUADRUPLET_MARGIN1'] = 0.3
                rivas_yaml['SOLVER']['QUADRUPLET_MARGIN2'] = 0.3
                
            rivas_yaml['MODEL'] = yaml_model_section.copy()
            rivas_yaml['MODEL']['PRETRAIN_PATH'] = pretrained_uam_ckpt            
            if specific_size:
                if class_name == "Rubish":
                    rivas_yaml['INPUT']['SIZE_TRAIN'] = [160, 224]
                    rivas_yaml['INPUT']['SIZE_TEST'] = [160, 224]
                elif class_name == "Crosswalks":
                    rivas_yaml['INPUT']['SIZE_TRAIN'] = [320, 96]
                    rivas_yaml['INPUT']['SIZE_TEST'] = [320, 96]
                elif class_name == "Containers":
                    rivas_yaml['INPUT']['SIZE_TRAIN'] = [128, 160]
                    rivas_yaml['INPUT']['SIZE_TEST'] = [128, 160]
                else:
                    rivas_yaml['INPUT']['SIZE_TRAIN'] = [256, 128]
                    rivas_yaml['INPUT']['SIZE_TEST'] = [256, 128]
            
            
            if USE_MODEL_TYPE == 'PaT':
                # Sustituye 'NAMES' por 'TRAIN' y define 'TEST'
                rivas_yaml['DATASETS']['TRAIN'] = rivas_yaml['DATASETS'].pop('NAMES', ['UAM'])
                rivas_yaml['DATASETS']['TEST'] = ['UAM_test']
            
                rivas_yaml['OUTPUT_DIR'] = rivas_output_dir
                rivas_yaml['LOG_ROOT'] = rivas_output_dir
                rivas_yaml['TB_LOG_ROOT'] = os.path.join(rivas_output_dir, 'tb')
                
                # Cambiar WARMUP_ITERS -> WARMUP_EPOCHS si existe
                if 'WARMUP_ITERS' in rivas_yaml['SOLVER']:
                    rivas_yaml['SOLVER']['WARMUP_EPOCHS'] = rivas_yaml['SOLVER'].pop('WARMUP_ITERS')

                rivas_yaml['TEST']['RE_RANKING'] = rivas_yaml['TEST']['RE_RANKING'].lower() == 'yes'
                rivas_yaml['TEST']['FEAT_NORM'] = rivas_yaml['TEST']['FEAT_NORM'].lower() == 'true'
            
            rivas_yaml_path = os.path.join(rivas_output_dir, "train_rivas.yaml")
            with open(rivas_yaml_path, 'w') as f:
                yaml.dump(rivas_yaml, f)
            
            run_training(rivas_yaml_path)
            
            if previous_fine_tuning:
                try:
                    for item in os.listdir(uam_output_dir):
                        item_path = os.path.join(uam_output_dir, item)
                        if os.path.isfile(item_path) and item not in ["track.txt", "qf.npy", "gf.npy"]:
                            os.remove(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                except Exception as e:
                    print(f"")
                
            if USE_MODEL_TYPE == 'PaT':
                try:
                    final_ckpt = sorted(
                        [f for f in os.listdir(rivas_output_dir) if f.endswith('.pth')],
                        key=lambda x: os.path.getmtime(os.path.join(rivas_output_dir, x)),
                        reverse=True
                    )[0]
                    print("Pesos cargados: ", final_ckpt)
                    final_ckpt_path = os.path.join(rivas_output_dir, final_ckpt)
                    
                except IndexError:
                    continue
            else:
                try:
                    final_ckpt = sorted(
                        [f for f in os.listdir(rivas_output_dir) if f.endswith('.pt')],
                        key=lambda x: os.path.getmtime(os.path.join(rivas_output_dir, x)),
                        reverse=True
                    )[0]
                    final_ckpt_path = os.path.join(rivas_output_dir, final_ckpt)
                except IndexError:
                    continue
            
            update_yaml = copy.deepcopy(rivas_yaml)
            print("yaml copiado")
            update_yaml['TEST']['WEIGHT'] = final_ckpt_path
            
            if USE_MODEL_TYPE == 'PaT':
                update_yaml['DATASETS']['TEST'] = ['UAM_test']
                if 'NAMES' in update_yaml['DATASETS']:
                    del update_yaml['DATASETS']['NAMES']
            else:
                update_yaml['DATASETS']['NAMES'] = ['UAM_test']
            

            update_yaml_path = os.path.join(rivas_output_dir, "update_rivas.yaml")
            with open(update_yaml_path, 'w') as f:
                yaml.dump(update_yaml, f)
                
            print("yaml hecho")
            track_txt_path = os.path.join(rivas_output_dir, "track.txt")
            run_update(update_yaml_path, track_txt_path)
            print("Update completado!!!!")
            

# ================== Mixing the results of different models ==================
for class_name in CLASSES:
    for pair_name in DATASET_PAIRS.keys():
        for domain in ["RIVAS"]:
            print(f"\nFusionando {len(MODEL_NAMES)} modelos para {class_name} | {pair_name} | {domain}")

            model_dirs = [
                os.path.join(TRANSFER_ROOT, f"{class_name}_{model_name}_{pair_name}", domain)
                for model_name in MODEL_NAMES
            ]

            output_dir = os.path.join(TRANSFER_ROOT, f"{class_name}_Combined_{pair_name}", domain, "RESULTS")
            os.makedirs(output_dir, exist_ok=True)

            qfs, gfs = [], []
            for dir_path in model_dirs:
                try:
                    qf = np.load(os.path.join(dir_path, "qf.npy"))
                    gf = np.load(os.path.join(dir_path, "gf.npy"))
                    qfs.append(l2norm(qf))
                    gfs.append(l2norm(gf))
                except Exception as e:
                    print(f"[WARNING] No se pudieron cargar embeddings de {dir_path}: {e}")
                    continue

            if not qfs or not gfs:
                print(f"[WARNING] No se pudo realizar fusión por falta de embeddings en {domain}")
                continue

            # ================== By avg ==================
            qf_mean = l2norm(np.mean(qfs, axis=0))
            gf_mean = l2norm(np.mean(gfs, axis=0))

            q_g = np.dot(qf_mean, gf_mean.T)
            q_q = np.dot(qf_mean, qf_mean.T)
            g_g = np.dot(gf_mean, gf_mean.T)

            dist = re_ranking(q_g, q_q, g_g, k1=10, k2=6, lambda_value=0.1)
            indices = np.argsort(dist, axis=1)

            track_mean_path = os.path.join(output_dir, "track_mean.txt")
            with open(track_mean_path, 'w') as f:
                for row in indices:
                    adjusted = [str(i + 1) for i in row[:100]]
                    f.write(" ".join(adjusted) + "\n")

            # ================== By concat ==================
            qf_concat = l2norm(np.concatenate(qfs, axis=1))
            gf_concat = l2norm(np.concatenate(gfs, axis=1))

            q_g = np.dot(qf_concat, gf_concat.T)
            q_q = np.dot(qf_concat, qf_concat.T)
            g_g = np.dot(gf_concat, gf_concat.T)

            dist = re_ranking(q_g, q_q, g_g, k1=10, k2=6, lambda_value=0.1)
            indices = np.argsort(dist, axis=1)

            track_concat_path = os.path.join(output_dir, "track_concat.txt")
            with open(track_concat_path, 'w') as f:
                for row in indices:
                    adjusted = [str(i + 1) for i in row[:100]]
                    f.write(" ".join(adjusted) + "\n")

            # ================== evaluate only if we are on UAM ==================
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


# ================== Final fussion between datasets ==================
for class_name in CLASSES:
    for domain in ["RIVAS"]:
        for variant in ["track_mean.txt", "track_concat.txt"]:
            print(f"\nFusionando datasets para {class_name} | {domain} | {variant}")

            tracks = []
            for pair_name in DATASET_PAIRS.keys():
                track_path = os.path.join(
                    TRANSFER_ROOT, f"{class_name}_Combined_{pair_name}", domain, "RESULTS", variant
                )
                if os.path.exists(track_path):
                    tracks.append(track_path)
                else:
                    print(f"[INFO] No encontrado: {track_path}")

            if not tracks:
                print(f"[WARNING] No hay tracks ({variant}) para {class_name} | {domain}, saltando...")
                continue

            output_fused = os.path.join(TRANSFER_ROOT, f"{class_name}_FINAL", domain, "RESULTS")
            os.makedirs(output_fused, exist_ok=True)

            fused_track_path = os.path.join(output_fused, variant)
            fusionar_tracks(tracks, fused_track_path)

            if domain == "UAM":
                try:
                    first_dataset_path = list(DATASET_PAIRS.values())[0]["UAM"].format(class_name)
                    print(f"Evaluando track fusionado FINAL en UAM para {class_name} | {variant}")
                    eval_proc = subprocess.run([
                        "python", "Evaluate_UrbAM-ReID.py",
                        "--track", fused_track_path,
                        "--path", first_dataset_path
                    ], text=True, capture_output=True)
                    print(eval_proc.stdout)
                except Exception as e:
                    print(f"[WARNING] Error al evaluar en UAM ({variant}) para {class_name}: {e}")

# ================== Generating the submission file for Kaggle ==================
print("\n================== Generando submission ==================")

# ================== Aux fct ==================
def parse_items(xml_path):
    tree = ET.parse(xml_path)
    return [item.attrib["imageName"] for item in tree.getroot().findall("Item")]

def extract_number(filename):
    return str(int(filename.replace(".jpg", "").lstrip("0") or "0"))

# ================== Generating the submission ==================
print("\n================== Generando submission (mean y concat) ==================")

for variant in ["track_mean.txt", "track_concat.txt"]:
    print(f"\n--- Procesando variante: {variant} ---")

    # ================== Outputs ==================
    names_file = results_dir / f"Results_RIVAS_ALL_names_{variant.replace('.txt','')}.txt"
    indices_file = results_dir / f"Results_RIVAS_ALL_indices_{variant.replace('.txt','')}.txt"
    submission_file = results_dir / f"RESULT_SUBMISSION_{variant.replace('.txt','')}.txt"
    output_csv_path = results_dir / f"kaggle_submission_converted_1based_{variant.replace('.txt','')}.csv"

    # ================== Step 1: Generate a name per class ==================
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

    # ================== Step 2: Global order and saving the names ==================
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

    # ================== Step 3: Convert to numerical order ==================
    with open(names_file, "r") as fin:
        lines = fin.readlines()

    with open(indices_file, "w") as fout:
        for line in lines:
            parts = line.strip().split()
            numbers = [extract_number(name) for name in parts]
            fout.write(" ".join(numbers) + "\n")

    print(f"Guardado ÍNDICES en: {indices_file}")

    # ================== Step 4: Delete de index of the query ==================
    with open(indices_file, "r") as fin, open(submission_file, "w") as fout:
        for line in fin:
            parts = line.strip().split()
            fout.write(" ".join(parts[1:]) + "\n")  # Skip query name

    print(f"Guardado RESULT_SUBMISSION en: {submission_file}")

    # ================== Step 5: Create the final CSV to upload it on Kaggle ==================
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

    # ================== Warning ==================
    if missing:
        print(f"WARNING: {len(missing)} queries faltantes (ejemplo: {missing[:5]})")


if USE_UNIFIED_MODEL:
    print("\n================== Procesando predicciones con modelo unificado ==================")

    # Cargar clases del QUERY desde el XML correcto
    query_classified_xml = Path(KAGGLE_ROOT) / "query_label_classified.xml"
    tree_q = ET.parse(query_classified_xml)
    root_q = tree_q.getroot()
    query_classes = {
        item.attrib["imageName"]: item.attrib["predictedClass"]
        for item in root_q.findall("Item")
    }

    # Cargar clases de GALLERY desde el XML de test
    test_classified_xml = Path(KAGGLE_ROOT) / "test_label_classified.xml"
    tree_g = ET.parse(test_classified_xml)
    root_g = tree_g.getroot()
    gallery_classes = {
        item.attrib["imageName"]: item.attrib["predictedClass"]
        for item in root_g.findall("Item")
    }

    # index (int) → class para la galería (para poder mapear índices a clases)
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
        processed_rows = []

        for _, row in df.iterrows():
            query_name = row["imageName"]
            pred_indices = [int(x) for x in row["Corresponding Indexes"].split()]
            final_preds = pred_indices

            if rerank_class:
                query_class = query_classes.get(query_name)
                if query_class:
                    same_class_preds = [idx for idx in pred_indices if index_to_class.get(idx) == query_class]
                    other_class_preds = [idx for idx in pred_indices if index_to_class.get(idx) != query_class]
                    final_preds = same_class_preds + other_class_preds

            processed_rows.append([query_name, " ".join(str(i) for i in final_preds[:100])])

        output_path = csv_path.with_name(csv_path.stem + "CLASSIFIED.csv")
        pd.DataFrame(processed_rows, columns=["imageName", "Corresponding Indexes"]).to_csv(output_path, index=False)
        print(f"Guardado: {output_path}")
        
    print("\n================== Finalizado ==================")

else:
    print("\n================== Finalizado ==================")
