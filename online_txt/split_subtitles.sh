#!/bin/bash

# 函數：顯示使用方法
show_usage() {
  echo "Usage: $0 -i <input_directory> -o <output_directory> [-m <max_chars>]"
  echo "  -i: 包含.txt文件的輸入目錄"
  echo "  -o: 輸出字幕文件的目錄"
  echo "  -m: 每行的最大字符數（可選，默認為15）"
  exit 1
}

# 函數：將文本分割成句子或子句
split_text() {
  local delimiter="$1"
  sed "s/$delimiter/$delimiter\n/g" | sed '/^$/d'
}

# 新函數：嘗試在常見的連詞或短語後分割
try_split_at_common_words() {
  local line="$1"
  local max_len="$2"
  local common_words=("但是" "不過" "然而" "雖然" "因為" "所以" "並且" "而且" "或者" "並" "而")
  
  for word in "${common_words[@]}"; do
    if [[ "$line" == *"$word"* ]]; then
      local parts=()
      IFS="$word" read -ra parts <<< "$line"
      for ((i=0; i<${#parts[@]}-1; i++)); do
        local part="${parts[i]}$word"
        if [ ${#part} -le $max_len ]; then
          echo "$part"
          line="${parts[*]:$((i+1))}"
          word=""
        fi
      done
      break
    fi
  done
  echo "$line"
}

# 新函數：將行強制分割成指定長度
force_split_line() {
  local line="$1"
  local max_len="$2"
  
  while [ ${#line} -gt $max_len ]; do
    local part="${line:0:$max_len}"
    echo "$part"
    line="${line:$max_len}"
  done
  echo "$line"
}

# 函數：處理單個文件
process_file() {
  local input="$1"
  local output="$2"
  local max_len="$3"
  
  # 首先嘗試使用句點分割
  split_text "。" < "$input" | while IFS= read -r line; do
    if [ ${#line} -le $max_len ]; then
      echo "$line" >> "$output"
    else
      # 如果句子太長，則嘗試使用逗號分割
      echo "$line" | split_text "，" | while IFS= read -r subline; do
        if [ ${#subline} -le $max_len ]; then
          echo "$subline" >> "$output"
        else
          # 如果子句仍然太長，嘗試在常見詞語後分割
          local temp_line="$subline"
          while [ ${#temp_line} -gt $max_len ]; do
            local new_lines=$(try_split_at_common_words "$temp_line" $max_len)
            if [ "$new_lines" == "$temp_line" ]; then
              # 如果無法在常見詞語處分割，則強制分割
              force_split_line "$temp_line" $max_len >> "$output"
              break
            else
              local first_line=$(echo "$new_lines" | head -n 1)
              echo "$first_line" >> "$output"
              temp_line=$(echo "$new_lines" | tail -n +2)
            fi
          done
          [ ${#temp_line} -gt 0 ] && echo "$temp_line" >> "$output"
        fi
      done
    fi
  done
}

# 解析命令行參數
while getopts ":i:o:m:" opt; do
  case $opt in
    i) input_dir="$OPTARG" ;;
    o) output_dir="$OPTARG" ;;
    m) max_chars="$OPTARG" ;;
    \?) echo "無效選項：-$OPTARG" >&2; show_usage ;;
    :) echo "選項 -$OPTARG 需要參數" >&2; show_usage ;;
  esac
done

# 檢查必要的參數
if [ -z "$input_dir" ] || [ -z "$output_dir" ]; then
  show_usage
fi

# 設定默認的最大字符數
max_chars=${max_chars:-15}

# 確保輸出目錄存在
mkdir -p "$output_dir"

# 處理輸入目錄中的所有.txt文件
find "$input_dir" -type f -name "*.txt" | while read -r file; do
  base_name=$(basename "$file" .txt)
  output_file="$output_dir/${base_name}_subtitles.txt"
  
  echo "處理文件：$file -> $output_file"
  process_file "$file" "$output_file" $max_chars
done

echo "所有文件處理完成。字幕文件位於：$output_dir"