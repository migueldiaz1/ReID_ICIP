#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 11:04:01 2025

@author: mdb
"""

def convert_track_to_csv(track_txt_path, test_xml_path, output_csv_path, one_based=True, top_k=100):
    import xml.etree.ElementTree as ET
    import pandas as pd
    import csv

    def parse_gallery_names(xml_path):
        tree = ET.parse(xml_path)
        return [item.attrib["imageName"] for item in tree.getroot().findall("Item")]

    def extract_number(filename):
        return str(int(filename.replace(".jpg", "").lstrip("0") or "0"))

    # Cargar nombres de galería desde el XML
    gallery_names = parse_gallery_names(test_xml_path)

    # Cargar el track.txt
    with open(track_txt_path, 'r') as f:
        lines = [line.strip().split() for line in f.readlines()]

    all_rows = []
    for idx, line in enumerate(lines):
        query_name = "{:06d}.jpg".format(idx + 1)  # nombre 1-based del query

        indices = []
        for x in line[:top_k]:
            try:
                g_idx = int(x)
                if one_based:
                    g_idx -= 1  # convertir a 0-based si es necesario
                if 0 <= g_idx < len(gallery_names):
                    gallery_image = gallery_names[g_idx]
                    gallery_index = extract_number(gallery_image)
                    indices.append(gallery_index)
                else:
                    indices.append("0")
            except:
                indices.append("0")

        all_rows.append([query_name, " ".join(indices)])

    # Guardar como CSV
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["imageName", "Corresponding Indexes"])
        writer.writerows(all_rows)

    print(f"[OK] CSV guardado en: {output_csv_path}")


convert_track_to_csv(
    track_txt_path="/home/mdb/DL_Lab3/track_bline_orig.txt",
    test_xml_path="/home/mdb/DL_Lab3/RIVAS_DATASET/DATASET/Unified/test_label.xml",
    output_csv_path="/home/mdb/DL_Lab3/track_bline_orig.csv"
    )
