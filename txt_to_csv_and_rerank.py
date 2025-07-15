#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 21:31:52 2025

@author: mdb
"""

def convert_track_to_csv_with_class_rerank(
    track_txt_path,
    test_xml_path,
    output_csv_path,
    query_classified_xml,
    test_classified_xml,
    one_based=True,
    top_k=100,
    rerank_class=True
):
    import xml.etree.ElementTree as ET
    import pandas as pd
    import csv
    from pathlib import Path

    def parse_items(xml_path):
        tree = ET.parse(xml_path)
        return [item.attrib["imageName"] for item in tree.getroot().findall("Item")]

    def extract_number(filename):
        return str(int(filename.replace(".jpg", "").lstrip("0") or "0"))

    # === Paso 1: cargar nombres e índices ===
    gallery_names = parse_items(test_xml_path)
    gallery_index_to_name = {idx: name for idx, name in enumerate(gallery_names)}
    name_to_index = {name: idx for idx, name in enumerate(gallery_names)}

    # === Paso 2: cargar clases ===
    query_class_dict = {}
    gallery_class_dict = {}

    tree_q = ET.parse(query_classified_xml)
    for item in tree_q.getroot().findall("Item"):
        query_class_dict[item.attrib["imageName"]] = item.attrib["predictedClass"]

    tree_g = ET.parse(test_classified_xml)
    for item in tree_g.getroot().findall("Item"):
        gallery_class_dict[item.attrib["imageName"]] = item.attrib["predictedClass"]

    # Crear mapping: índice (1-based) → clase
    index_to_class = {
        idx + 1: gallery_class_dict.get(name, "unknown") for idx, name in enumerate(gallery_names)
    }

    # === Paso 3: cargar track.txt ===
    with open(track_txt_path, 'r') as f:
        lines = [line.strip().split() for line in f.readlines()]

    all_rows = []
    for idx, line in enumerate(lines):
        query_name = "{:06d}.jpg".format(idx + 1)  # 1-based nombre de la query

        pred_indices = []
        for x in line[:top_k]:
            try:
                g_idx = int(x)
                if one_based:
                    g_idx -= 1
                if 0 <= g_idx < len(gallery_names):
                    gallery_name = gallery_names[g_idx]
                    gallery_index = int(extract_number(gallery_name))
                    pred_indices.append(gallery_index)
                else:
                    pred_indices.append(0)
            except:
                pred_indices.append(0)

        # === Paso 4: reranking por clase ===
        if rerank_class:
            query_cls = query_class_dict.get(query_name, None)
            if query_cls:
                same_class = [idx for idx in pred_indices if index_to_class.get(idx) == query_cls]
                other_class = [idx for idx in pred_indices if index_to_class.get(idx) != query_cls]
                pred_indices = same_class + other_class

        all_rows.append([query_name, " ".join(str(i) for i in pred_indices[:top_k])])

    # === Paso 5: guardar CSV ===
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["imageName", "Corresponding Indexes"])
        writer.writerows(all_rows)

    print(f"[OK] CSV con reranking por clase guardado en: {output_csv_path}")


convert_track_to_csv_with_class_rerank(
    track_txt_path="/home/mdb/reid-strong-baseline/TRANSFER_Learning_SR_ensembling_BestEnsemb_FINAL/Unified_se_resnet50_base/RIVAS/track.txt",
    test_xml_path="/home/mdb/DL_Lab3/RIVAS_DATASET/DATASET/Unified/test_label.xml",
    output_csv_path="/home/mdb/reid-strong-baseline/TRANSFER_Learning_SR_ensembling_BestEnsemb_FINAL/Unified_se_resnet50_base/RIVAS/Track_200ep_Best.csv",
    query_classified_xml="/home/mdb/DL_Lab3/Kaggle_Dataset/query_label_classified.xml",
    test_classified_xml="/home/mdb/DL_Lab3/Kaggle_Dataset/test_label_classified.xml",
    one_based=True,
    top_k=100,
    rerank_class=True
)
