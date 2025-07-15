#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 23:38:30 2025

@author: mdb
"""
"""
Hyperparameter tuning automático para BoT (Bag of Tricks)
Probará dos datasets por clase y guardará resultados y gráficas
"""

import os
import itertools
import yaml
import subprocess
from datetime import datetime
import json
import copy
import glob
import shutil

# === Configuración global ===
CLASSES = ["Containers", "Crosswalks", "Rubish"]
DATASET_PATHS = [
    "/home/mdb/DL_Lab3/UAM_DATASET/stratified_correct_noC004/{}",
    "/home/mdb/DL_Lab3/COMBINED_DATASET/{}"
]
ROOT_BASE = "/home/mdb/reid-strong-baseline/Hyperparameter_tuning"

# === Hiperparámetros a combinar ===
model_names = ['resnet50', 'resnet50_ibn_a']
model_paths = {
    'resnet50': '/home/mdb/reid-strong-baseline/configs/resnet50-19c8e357 (1).pth',
    'resnet50_ibn_a': '/home/mdb/reid-strong-baseline/configs/r50_ibn_a.pth'
}
loss_types = ['triplet_center']
reranking_options = ['yes', 'no']
base_lrs = [0.00035, 0.0001]
center_loss_weights = [0.0005, 0.001]
range_margins = [0.3]
range_betas = [1.0]
weight_decays = [0.0001]
max_epochs_list = [25, 50]

# === Plantilla YAML base ===
base_yaml_template = {
    'MODEL': {
        'PRETRAIN_CHOICE': 'imagenet',
        'METRIC_LOSS_TYPE': None,
        'IF_LABELSMOOTH': 'on',
        'IF_WITH_CENTER': 'yes',
        'NAME': None,
        'PRETRAIN_PATH': None,
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
        'MAX_EPOCHS': None,
        'BASE_LR': None,
        'CENTER_LR': 0.5,
        'CENTER_LOSS_WEIGHT': None,
        'CLUSTER_MARGIN': 0.3,
        'RANGE_K': 2,
        'RANGE_MARGIN': None,
        'RANGE_ALPHA': 0,
        'RANGE_BETA': None,
        'RANGE_LOSS_WEIGHT': 1,
        'BIAS_LR_FACTOR': 1,
        'WEIGHT_DECAY': None,
        'WEIGHT_DECAY_BIAS': 0.0005,
        'IMS_PER_BATCH': 64,
        'STEPS': [40, 70],
        'GAMMA': 0.1,
        'WARMUP_FACTOR': 0.01,
        'WARMUP_ITERS': 10,
        'WARMUP_METHOD': 'linear',
        'CHECKPOINT_PERIOD': 10,
        'LOG_PERIOD': 5,
        'EVAL_PERIOD': 105
    },
    'TEST': {
        'IMS_PER_BATCH': 128,
        'WEIGHT': "path",
        'RE_RANKING': None,
        'NECK_FEAT': 'after',
        'FEAT_NORM': 'yes'
    }
}

# === Loop por clase y dataset ===
best_map_by_group = {}  # clave = (class, dataset_tag, model) → mAP
best_model_path_by_group = {}

for CLASS_NAME in CLASSES:
    for dataset_path_template in DATASET_PATHS:
        dataset_path = dataset_path_template.format(CLASS_NAME)
        dataset_tag = os.path.basename(os.path.dirname(dataset_path))
        output_root = os.path.join(ROOT_BASE, CLASS_NAME, dataset_tag)
        config_dir = os.path.join(output_root, "configs")
        os.makedirs(config_dir, exist_ok=True)

        combinations = list(itertools.product(
            model_names, loss_types, reranking_options,
            base_lrs, center_loss_weights,
            range_margins, range_betas, weight_decays, max_epochs_list
        ))

        for i, (model, loss, rerank, lr, clw, r_margin, r_beta, wd, max_ep) in enumerate(combinations):
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                exp_name = f"{model}_{loss}_RR{rerank}_LR{lr}_CLW{clw}_RM{r_margin}_RB{r_beta}_WD{wd}_EP{max_ep}_{timestamp}"
                exp_dir = os.path.join(output_root, exp_name)
                os.makedirs(exp_dir, exist_ok=True)
                total_experiments = len(CLASSES) * len(DATASET_PATHS) * len(combinations)

                print(f"\n[{i+1}] Ejecutando experimento: {exp_name} ({i+1} de {len(combinations)} para {CLASS_NAME}, {((CLASSES.index(CLASS_NAME) * len(DATASET_PATHS) + DATASET_PATHS.index(dataset_path_template)) * len(combinations) + i + 1)} de {total_experiments} en total)")
                print(f"Parámetros: model={model}, loss={loss}, rerank={rerank}, lr={lr}, clw={clw}, rm={r_margin}, rb={r_beta}, wd={wd}, epochs={max_ep}")

                # YAML de entrenamiento
                config = copy.deepcopy(base_yaml_template)
                config['MODEL'].update({
                    'METRIC_LOSS_TYPE': loss,
                    'NAME': model,
                    'PRETRAIN_PATH': model_paths[model]
                })
                config['DATASETS']['ROOT_DIR'] = dataset_path
                config['SOLVER'].update({
                    'MAX_EPOCHS': max_ep,
                    'BASE_LR': lr,
                    'CENTER_LOSS_WEIGHT': clw,
                    'RANGE_MARGIN': r_margin,
                    'RANGE_BETA': r_beta,
                    'WEIGHT_DECAY': wd
                })
                config['TEST']['RE_RANKING'] = rerank
                config['OUTPUT_DIR'] = exp_dir

                yaml_path = os.path.join(config_dir, f"{exp_name}.yml")
                with open(yaml_path, "w") as f:
                    yaml.dump(config, f)

                subprocess.run(["python", "tools/train.py", "--config_file", yaml_path],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                model_files = sorted(glob.glob(os.path.join(exp_dir, "*.pt")), key=os.path.getmtime, reverse=True)
                if not model_files:
                    print(f"[{i+1}] No se encontró modelo .pt en {exp_dir}")
                    shutil.rmtree(exp_dir)
                    os.remove(yaml_path)
                    continue

                best_model_path = model_files[0]
                # YAML de test
                test_config = copy.deepcopy(config)
                test_config['DATASETS']['NAMES'] = ['UAM_test']
                test_config['TEST']['WEIGHT'] = best_model_path
                test_config['MODEL']['PRETRAIN_CHOICE'] = 'self'

                test_yaml_path = os.path.join(exp_dir, "test_config.yaml")
                with open(test_yaml_path, "w") as f:
                    yaml.dump(test_config, f)

                track_path = os.path.join(exp_dir, "track.txt")
                subprocess.run([
                    "python", "tools/update.py",
                    "--config_file", test_yaml_path,
                    "--track", track_path,
                    "TEST.RE_RANKING", rerank
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                eval_output = subprocess.run([
                    "python", "Evaluate_UrbAM-ReID.py",
                    "--track", track_path,
                    "--path", dataset_path
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                output_text = eval_output.stdout
                print(f"[{i+1}] Salida de evaluación:\n{output_text}")

                # === EXTRAER mAP ===
                try:
                    for line in output_text.splitlines():
                        if "mAP=" in line:
                            map_value = float(line.split("=")[1])
                            # Al final del bloque try donde se extrae el mAP:

                            # === REGISTRAR RESULTADOS EN JSON ===
                            exp_result = {
                                "exp_name": exp_name,
                                "class": CLASS_NAME,
                                "dataset": dataset_tag,
                                "model": model,
                                "loss": loss,
                                "rerank": rerank,
                                "lr": lr,
                                "clw": clw,
                                "rm": r_margin,
                                "rb": r_beta,
                                "wd": wd,
                                "epochs": max_ep,
                                "mAP": map_value
                            }
                            
                            # Ruta fija (crear solo una vez al principio del script si quieres)
                            json_result_path = os.path.join(ROOT_BASE, "hyperparameter_tuning_log.json")
                            
                            # Cargar resultados existentes
                            if os.path.exists(json_result_path):
                                with open(json_result_path, "r") as jf:
                                    resultados_totales = json.load(jf)
                            else:
                                resultados_totales = []
                            
                            # Añadir el nuevo resultado
                            resultados_totales.append(exp_result)
                            
                            # Guardar actualizados
                            with open(json_result_path, "w") as jf:
                                json.dump(resultados_totales, jf, indent=4)

                            break
                    else:
                        raise ValueError("No se encontró mAP en el output")

                    group_key = (CLASS_NAME, dataset_tag, model)
                    prev_best = best_map_by_group.get(group_key, -1)

                    if map_value > prev_best:
                        # eliminar modelo anterior si existía
                        old_path = best_model_path_by_group.get(group_key)
                        if old_path:
                            try:
                                shutil.rmtree(os.path.dirname(old_path))
                            except Exception:
                                pass
                        best_map_by_group[group_key] = map_value
                        best_model_path_by_group[group_key] = best_model_path
                        print(f"[{i+1}] Nuevo mejor modelo para {group_key}: mAP={map_value}")
                    else:
                        # borrar este modelo porque no es el mejor
                        shutil.rmtree(exp_dir)
                        os.remove(yaml_path)
                        print(f"[{i+1}] Eliminado modelo no óptimo (mAP={map_value})")

                except Exception as e:
                    print(f"[{i+1}] ⚠️ Error extrayendo mAP: {e}")
                    shutil.rmtree(exp_dir)
                    os.remove(yaml_path)
                    continue

            except Exception as e:
                print(f"[{i+1}] ⚠️ Error general en {exp_name}: {e}")
                continue

