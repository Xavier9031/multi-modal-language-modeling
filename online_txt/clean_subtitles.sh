#!/bin/bash

# 函數：顯示使用方法
show_usage() {
  echo "Usage: $0 -i <input_directory> -o <output_directory> [-m <min_chars>]"
  echo "  -i: 包含要處理的.txt文件的輸入目錄"
  echo "  -o: 保存處理後文件的輸出目錄"
  echo "  -m: 保留行的最小字符數（可選，默認為5）"
  exit 1
}

# 解析命令行參數
while getopts ":i:o:m:" opt; do
  case $opt in
    i) input_dir="$OPTARG" ;;
    o) output_dir="$OPTARG" ;;
    m) min_chars="$OPTARG" ;;
    \?) echo "無效選項：-$OPTARG" >&2; show_usage ;;
    :) echo "選項 -$OPTARG 需要參數" >&2; show_usage ;;
  esac
done

# 檢查必要的參數
if [ -z "$input_dir" ] || [ -z "$output_dir" ]; then
  show_usage
fi

# 設定默認的最小字符數
min_chars=${min_chars:-5}

# 確保輸出目錄存在
mkdir -p "$output_dir"

# 處理輸入目錄中的所有.txt文件
find "$input_dir" -type f -name "*.txt" -print0 | while IFS= read -r -d '' file; do
  base_name=$(basename "$file" .txt)
  output_file="$output_dir/${base_name}_clean.txt"
  
  # 使用sed和awk處理文件
  sed 's/^[[:space:]]*//' "$file" | awk -v min="$min_chars" 'length >= min' > "$output_file"

  # 如果輸出文件非空，報告成功
  if [ -s "$output_file" ]; then
    echo "已處理：$file -> $output_file"
  else
    # 如果輸出文件為空，刪除它並報告
    rm "$output_file"
    echo "已處理但為空（已刪除）：$file"
  fi
done

echo "所有文件處理完成。字幕文件位於：$output_dir"