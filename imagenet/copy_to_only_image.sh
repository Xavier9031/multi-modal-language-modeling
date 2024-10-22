#!/bin/bash

# 檢查是否提供了兩個參數
if [ $# -ne 2 ]; then
  echo "Usage: $0 <source_directory> <destination_directory>"
  exit 1
fi

# 設定源目錄和目標目錄
source_dir="$1"
dest_dir="$2"

# 確保目標目錄存在，如果不存在則創建
mkdir -p "$dest_dir"

# 遍歷源目錄下的所有子目錄中的JPEG檔案
find "$source_dir" -type d -exec find {} -maxdepth 1 -type f -name "*.JPEG" \; | while read -r file; do
  # 印出檔案名稱
  echo "Found: $file"
  
  # 複製檔案到目標目錄
  cp "$file" "$dest_dir/"
done

echo "All JPEG files have been copied to $dest_dir"