#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 21:06:51 2025

@author: mdb
"""

import os
from PIL import Image
import sys

def check_images(root_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    total = 0
    bad = 0

    for root, _, files in os.walk(root_dir):
        for fname in files:
            if os.path.splitext(fname)[-1].lower() not in exts:
                continue
            path = os.path.join(root, fname)
            total += 1
            try:
                with Image.open(path) as img:
                    img.verify()
            except Exception as e:
                print(f"[ERROR] Imagen corrupta: {path} ({e})")
                bad += 1

    print(f"\nTotal imágenes revisadas: {total}")
    print(f"Imágenes corruptas o ilegibles: {bad}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python validate_images.py /ruta/al/dataset")
    else:
        check_images(sys.argv[1])
