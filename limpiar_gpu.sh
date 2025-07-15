#!/bin/bash
# Deja solo el proceso con PID=$1, mata los demás procesos GPU

TARGET_PID=$1

for pid in $(nvidia-smi | grep ' C ' | awk '{print $5}' | grep -v "$TARGET_PID"); do
    echo "Matando PID $pid"
    kill -9 $pid
done

echo "✅ Solo queda el proceso $TARGET_PID en GPU (si todo ha ido bien)"

