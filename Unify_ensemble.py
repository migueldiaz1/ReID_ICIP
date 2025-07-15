#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 17:53:50 2025

@author: mdb
"""

import xml.etree.ElementTree as ET
from pathlib import Path

# Rutas locales base
unified_query_xml = Path("/home/mdb/DL_Lab3/UAM_DATASET/unified/query_label.xml")
unified_test_xml = Path("/home/mdb/DL_Lab3/UAM_DATASET/unified/test_label.xml")
stratified_base = Path("/home/mdb/DL_Lab3/UAM_DATASET/stratified_correct_noC004")
results_base = Path("/home/mdb/DL_Lab3/Part-Aware-Transformer/FINAL_ENSEMBLING/")
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
    class_result_path = results_base / proper_class / "TRACK.txt"
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
base_results = Path("/home/mdb/DL_Lab3/Part-Aware-Transformer/FINAL_ENSEMBLING/")
base_dataset = Path("/home/mdb/DL_Lab3/UAM_DATASET/stratified_correct_noC004")


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
