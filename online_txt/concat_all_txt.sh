#!/bin/bash

# 檢查是否提供了輸入目錄和輸出文件
if [ $# -ne 2 ]; then
  echo "Usage: $0 <input_directory> <output_file>"
  exit 1
fi

input_dir="$1"
output_file="$2"

# 找到所有.txt文件並按字母順序連接它們
find "$input_dir" -type f -name "*.txt" -print0 | sort -z | xargs -0 cat > "$output_file"

echo "已將 $input_dir 中的所有.txt文件連接到 $output_file"