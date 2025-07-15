#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 23:48:17 2025

@author: mdb
"""

import os
import subprocess
import yaml
import shutil

# Rutas base
base_dir = "/home/mdb/reid-strong-baseline/configs/HYP_TUNED"
output_base = "/home/mdb/reid-strong-baseline/OUTPUTS"

# Recorrer todas las clases
for class_name in os.listdir(base_dir):
    class_path = os.path.join(base_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    # Recorrer los modelos (IBN, Resnet)
    for model_dir in os.listdir(class_path):
        model_path = os.path.join(class_path, model_dir)
        if not os.path.isdir(model_path):
            continue

        # Crear carpeta de salida si no existe
        output_path = os.path.join(output_base, class_name, model_dir)
        os.makedirs(output_path, exist_ok=True)

        # Copiar archivos relevantes (.pt, .yaml, .txt) a OUTPUTS/CLASE/MODELO/
        for fname in os.listdir(model_path):
            if fname.endswith(".pt") or fname.endswith(".yaml") or fname.startswith("track_") or fname.startswith("Results_"):
                src = os.path.join(model_path, fname)
                dst = os.path.join(output_path, fname)
                shutil.copy(src, dst)

        # Buscar todos los archivos .yaml
        for file_name in os.listdir(model_path):
            if file_name.endswith(".yaml"):
                yaml_path = os.path.join(model_path, file_name)
                print(f"\n Ejecutando test para: {yaml_path}")

                with open(yaml_path, "r") as f:
                    config = yaml.safe_load(f)

                weight_path = config.get("TEST", {}).get("WEIGHT", None)
                if not weight_path or not os.path.exists(weight_path):
                    print(f"  Peso no encontrado o no definido: {weight_path}")
                    continue

                # Ejecutar update.py
                track_path = os.path.join(model_path, f"track_{file_name.replace('.yaml', '.txt')}")
                subprocess.run([
                    "python", "tools/update.py",
                    "--config_file", yaml_path,
                    "--track", track_path
                ])

                # Copiar el track generado también a OUTPUTS
                if os.path.exists(track_path):
                    shutil.copy(track_path, output_path)

                # Ejecutar Evaluate_UrbAM-ReID.py y mostrar salida
                dataset_path = config.get("DATASETS", {}).get("ROOT_DIR", "")
                if not dataset_path or not os.path.exists(dataset_path):
                    print(f" Dataset ROOT_DIR no encontrado o no válido: {dataset_path}")
                    continue

                eval_proc = subprocess.run([
                    "python", "Evaluate_UrbAM-ReID.py",
                    "--track", track_path,
                    "--path", dataset_path
                ], capture_output=True, text=True)

                print("Resultado de evaluación:")
                print(eval_proc.stdout)
                if eval_proc.stderr:
                    print("STDERR:")
                    print(eval_proc.stderr)
                    
import numpy as np
import os
import subprocess
import re
from utils.re_ranking import re_ranking
from itertools import product
import shutil

# === CONFIGURACIÓN GENERAL ===
CLASSES = ["Containers", "Rubish", "Crosswalks"]
BASE_PATH = "/home/mdb/reid-strong-baseline/OUTPUTS"
DATASET_PATH_BASE = "/home/mdb/DL_Lab3/UAM_DATASET/stratified_correct_noC004"
EVALUATE_SCRIPT = "Evaluate_UrbAM-ReID.py"

for CLASE in CLASSES:
    MODEL1_DIR = os.path.join(BASE_PATH, CLASE, "Resnet")
    MODEL2_DIR = os.path.join(BASE_PATH, CLASE, "IBN")
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

    # Parámetros para reranking
    k1s = [10, 50, 100]
    k2s = [3, 6, 9]
    lambdas = [0.1, 0.3, 0.5]
    track_base = os.path.join(OUTPUT_DIR, "track_k1{}_k2{}_l{}.txt")

    best_map = -1
    best_params = None

    for k1, k2, lmbd in product(k1s, k2s, lambdas):
        dist = re_ranking(q_g, q_q, g_g, k1=k1, k2=k2, lambda_value=lmbd)
        indices = np.argsort(dist, axis=1)

        track_path = track_base.format(k1, k2, int(lmbd * 100))
        with open(track_path, 'w') as f:
            for row in indices:
                f.write(" ".join(map(str, row[:99] + 1)) + "\n")

        result = subprocess.run([
            "python", EVALUATE_SCRIPT,
            "--track", track_path,
            "--path", os.path.join(DATASET_PATH_BASE, CLASE)
        ], capture_output=True, text=True)

        match = re.search(r"mAP= ([0-9.]+)", result.stdout)
        if match:
            mAP = float(match.group(1))
            #print(f" {CLASE} - k1={k1}, k2={k2}, λ={lmbd:.2f} → mAP={mAP:.5f}")
            if mAP > best_map:
                best_map = mAP
                best_params = (k1, k2, lmbd, track_path)
        else:
            print(f" {CLASE} - Error leyendo mAP:", result.stdout)

    # Guardar mejor resultado
    print(f"\nMejor configuración para {CLASE}:")
    print("k1={}, k2={}, lambda={:.2f}".format(*best_params[:3]))
    print("mAP={:.5f}".format(best_map))

    shutil.copyfile(best_params[3], os.path.join(RESULTS_DIR, "TRACK.txt"))

    with open(os.path.join(RESULTS_DIR, "best_config.txt"), 'w') as f:
        f.write("k1={}\nk2={}\nlambda={:.2f}\nmAP={:.5f}\n".format(*best_params[:3], best_map))

    print(f"Guardado en: {RESULTS_DIR}")


import os
import shutil

CLASSES = ["Containers", "Rubish", "Crosswalks"]
BASE_PATH = "/home/mdb/reid-strong-baseline/OUTPUTS"

for class_name in CLASSES:
    combined_path = os.path.join(BASE_PATH, class_name, "COMBINED")
    print(f"\n Limpiando COMBINED para clase: {class_name}")
    
    if not os.path.exists(combined_path):
        print(f"  Ruta no encontrada: {combined_path}")
        continue

    for item in os.listdir(combined_path):
        item_path = os.path.join(combined_path, item)

        if os.path.isdir(item_path) and item != "RESULTS":
            print(f" Borrando carpeta: {item_path}")
            shutil.rmtree(item_path)
        elif os.path.isfile(item_path):
            print(f" Borrando archivo: {item_path}")
            os.remove(item_path)

print("\n Limpieza completada.")


import xml.etree.ElementTree as ET
from pathlib import Path

# Rutas locales base
unified_query_xml = Path("/home/mdb/DL_Lab3/UAM_DATASET/unified/query_label.xml")
unified_test_xml = Path("/home/mdb/DL_Lab3/UAM_DATASET/unified/test_label.xml")
stratified_base = Path("/home/mdb/DL_Lab3/UAM_DATASET/stratified_correct_noC004")
results_base = Path("/home/mdb/reid-strong-baseline/OUTPUTS")
output_path = results_base / "Results_UAM_ALL"

# Leer queries globales
tree = ET.parse(unified_query_xml)
root = tree.getroot()
items_node = root.find("Items")
if items_node is None:
    raise ValueError(f"No se encontró el nodo <Items> en {unified_query_xml}")
query_items = items_node.findall("Item")
query_class_map = [(item.attrib["imageName"], item.attrib["predictedClass"]) for item in query_items]
print(f"Se leyeron {len(query_class_map)} queries del archivo global.")

# Leer test global y mapear imageName → índice global
tree = ET.parse(unified_test_xml)
items_node = tree.getroot().find("Items")
if items_node is None:
    raise ValueError(f"No se encontró el nodo <Items> en {unified_test_xml}")
global_test_items = items_node.findall("Item")
global_test_names = [item.attrib["imageName"] for item in global_test_items]
global_index_map = {name: idx for idx, name in enumerate(global_test_names)}

# Cachear queries por clase y sus índices locales
query_indices_by_class = {}
results_lines_by_class = {}
local_gallery_by_class = {}

class_name_map = {
    "containers": "Containers",
    "crosswalks": "Crosswalks",
    "rubish": "Rubish"
}

for lowercase_class, proper_class in class_name_map.items():
    # Leer queries de la clase
    class_query_xml = stratified_base / proper_class / "query_label.xml"
    tree = ET.parse(class_query_xml)
    items_node = tree.getroot().find("Items")
    if items_node is None:
        raise ValueError(f"No se encontró el nodo <Items> en {class_query_xml}")
    class_items = items_node.findall("Item")
    query_names = [item.attrib["imageName"] for item in class_items]
    query_indices_by_class[lowercase_class] = {name: idx for idx, name in enumerate(query_names)}
    print(f"{lowercase_class}: {len(query_names)} queries")

    # Leer resultados
    class_result_path = results_base / proper_class / "COMBINED" / "RESULTS" / "TRACK.txt"
    if not class_result_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de resultados: {class_result_path}")
    with open(class_result_path, "r") as f:
        results_lines_by_class[lowercase_class] = [line.strip().split() for line in f.readlines()]
    print(f"{lowercase_class}: {len(results_lines_by_class[lowercase_class])} líneas de resultados")

    # Leer galería local de la clase
    test_xml_path = stratified_base / proper_class / "test_label.xml"
    tree = ET.parse(test_xml_path)
    items_node = tree.getroot().find("Items")
    if items_node is None:
        raise ValueError(f"No se encontró el nodo <Items> en {test_xml_path}")
    gallery_items = items_node.findall("Item")
    local_gallery_names = [item.attrib["imageName"] for item in gallery_items]
    local_gallery_by_class[lowercase_class] = local_gallery_names
    print(f"{lowercase_class}: {len(local_gallery_names)} imágenes en galería")

# Generar Results_UAM_ALL traducido a índices globales
final_results = []
missing = []

# Encontrar longitud mínima común (por seguridad)
min_length = min(
    len(line)
    for result_lines in results_lines_by_class.values()
    for line in result_lines
)
print(f"Longitud mínima común entre resultados: {min_length}")

for image_name, class_name in query_class_map:
    query_idx_map = query_indices_by_class.get(class_name, {})
    results_lines = results_lines_by_class.get(class_name, [])
    local_gallery = local_gallery_by_class.get(class_name, [])

    query_idx = query_idx_map.get(image_name)
    if query_idx is not None and query_idx < len(results_lines):
        local_indices = results_lines[query_idx][:100]
        global_indices = []

        for local_idx in local_indices:
            try:
                gallery_idx = int(local_idx)
                if 0 <= gallery_idx < len(local_gallery):
                    img_name = local_gallery[gallery_idx]
                    global_idx = global_index_map.get(img_name, -1)
                    global_indices.append(str(global_idx))
                else:
                    global_indices.append("-1")
            except (ValueError, IndexError) as e:
                print(f"Error con índice local {local_idx}: {e}")
                global_indices.append("-1")

        if len(global_indices) < 100:
            global_indices += ["-1"] * (100 - len(global_indices))
        final_results.append(" ".join(global_indices))

    else:
        missing.append((image_name, class_name))

# Guardar archivo final
with open(output_path, "w") as f:
    f.write("\n".join(final_results))

# Reporte
print(f"\n Total queries en Results_UAM_ALL: {len(final_results)}")
print(f"Total queries esperadas: {len(query_class_map)}")
if missing:
    print(f"  Faltan {len(missing)} queries (primeros 10):")
    for m in missing[:10]:
        print("  ", m)

import subprocess
from pathlib import Path

# Define las clases y sus paths
clases = ["Containers", "Crosswalks", "Rubish"]
base_results = Path("/home/mdb/reid-strong-baseline/OUTPUTS")
base_dataset = Path("/home/mdb/DL_Lab3/UAM_DATASET/stratified_correct_noC004")

print("==== EVALUACIÓN POR CLASE ====\n")

for clase in clases:
    print(f"\n Evaluando clase: {clase}")
    track_path = base_results / clase / "COMBINED" / "RESULTS" / "TRACK.txt"
    dataset_path = base_dataset / clase

    proc = subprocess.run([
        "python", "Evaluate_UrbAM-ReID.py",
        "--track", str(track_path),
        "--path", str(dataset_path)
    ], capture_output=True, text=True)

    print(proc.stdout)
    if proc.stderr:
        print("⚠️ STDERR:")
        print(proc.stderr)

# Evaluación final con el conjunto unificado
print("\n==== EVALUACIÓN FINAL: UNIFICADO ====\n")
track_all = base_results / "Results_UAM_ALL"
unified_dataset = Path("/home/mdb/DL_Lab3/UAM_DATASET/unified")

proc = subprocess.run([
    "python", "Evaluate_UrbAM-ReID.py",
    "--track", str(track_all),
    "--path", str(unified_dataset)
], capture_output=True, text=True)

print(proc.stdout)
if proc.stderr:
    print(" STDERR:")
    print(proc.stderr)
