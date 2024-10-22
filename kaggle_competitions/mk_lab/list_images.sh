#!/bin/bash

# 定義要讀取的資料夾和輸出的 CSV 檔名
INPUT_DIR="/mnt/sda1/htchang/DL/HW3/kaggle_competitions/test"  # 替換成你的資料夾路徑
OUTPUT_FILE="output.csv"

# 讀取資料夾下所有的圖片檔案，排序後寫入 CSV
find "$INPUT_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.gif" -o -iname "*.bmp" \) | sort | while read -r file; do
  echo "$(basename "$file")" >> "$OUTPUT_FILE"
done

echo "所有圖片檔案的檔名已經按照字母順序寫入 $OUTPUT_FILE"
