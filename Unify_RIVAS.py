#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 23:19:50 2025

@author: mdb
"""
import xml.etree.ElementTree as ET
from pathlib import Path
import csv

# === CONFIGURACIÓN GENERAL ===
clases = ["Containers", "Crosswalks", "Rubish"]
base_dir = Path("/home/mdb/reid-strong-baseline/TRANSFER_Learning_SR_ensembling")
results_dir = base_dir / "FINAL"
global_query_xml = Path("/home/mdb/DL_Lab3/Kaggle_Dataset/query_label.xml")

# === SALIDAS ===
names_file = results_dir / "Results_RIVAS_ALL_names.txt"
indices_file = results_dir / "Results_RIVAS_ALL_indices.txt"
submission_file = results_dir / "RESULT_SUBMISSION.txt"
output_csv_path = results_dir / "kaggle_submission_converted_1based.csv"

# === FUNCIONES AUXILIARES ===
def parse_items(xml_path):
    tree = ET.parse(xml_path)
    return [item.attrib["imageName"] for item in tree.getroot().findall("Item")]

def extract_number(filename):
    return str(int(filename.replace(".jpg", "").lstrip("0") or "0"))

# === PASO 1: GENERAR NOMBRES POR CLASE ===
all_entries = []

for cls in clases:
    dataset_dir = Path("/home/mdb/DL_Lab3/RIVAS_DATASET/DATASET")
    query_xml = dataset_dir / cls / "query_label.xml"
    test_xml = dataset_dir / cls / "test_label.xml"
    track_file = base_dir / f"{cls}_FINAL" / "RIVAS" / "RESULTS" / "track.txt"

    query_names = parse_items(query_xml)
    gallery_names = parse_items(test_xml)

    with open(track_file, "r") as f:
        lines = [line.strip().split() for line in f.readlines()]

    assert len(query_names) == len(lines), f"⚠️ Mismatch en {cls}: {len(query_names)} queries vs {len(lines)} predicciones"

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

# === PASO 2: ORDEN GLOBAL Y GUARDAR NOMBRES ===
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

print(f"✅ Guardado NOMBRES en: {names_file}")

# === PASO 3: CONVERTIR A ÍNDICES NUMÉRICOS ===
with open(names_file, "r") as fin:
    lines = fin.readlines()

with open(indices_file, "w") as fout:
    for line in lines:
        parts = line.strip().split()
        numbers = [extract_number(name) for name in parts]
        fout.write(" ".join(numbers) + "\n")

print(f"✅ Guardado ÍNDICES en: {indices_file}")

# === PASO 4: ELIMINAR ÍNDICE DE QUERY PARA SUBMISSION FINAL ===
with open(indices_file, "r") as fin, open(submission_file, "w") as fout:
    for line in fin:
        parts = line.strip().split()
        fout.write(" ".join(parts[1:]) + "\n")  # Skip query name

print(f"✅ Guardado RESULT_SUBMISSION en: {submission_file}")

# === PASO 5: CREAR CSV PARA KAGGLE SUBMISSION ===
with open(submission_file, "r") as f:
    prediction_lines = [line.strip().split() for line in f.readlines()]

num_queries = len(global_query_names)
assert len(prediction_lines) == num_queries, f"⚠️ Esperados {num_queries} queries, pero hay {len(prediction_lines)}"

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

print(f"✅ CSV convertido para Kaggle guardado en: {output_csv_path}")
print(f"Índices predichos: mínimo={min_idx} | máximo={max_idx}")

# === AVISO FINAL ===
if missing:
    print(f"⚠️ WARNING: {len(missing)} queries faltantes (ejemplo: {missing[:5]})")

print("\n=== PROCESO COMPLETADO ===")
