import os
import json
import hashlib
from tqdm import tqdm

def process_file(file_path, output_dir):
    # 讀取文件內容
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 解析每行內容
    data = []
    for line in tqdm(lines):
        image_file, caption = line.strip().split(' ', 1)
        
        # 構建完整的圖像路徑
        full_path = os.path.join("/mnt/sda1/htchang/DL/HW3/train_data/images", image_file)
        
        # 使用SHA-256哈希算法對文件路徑進行哈希
        hash_object = hashlib.sha256(full_path.encode())
        hashed_id = hash_object.hexdigest()[:20]  # 取前20個字符作為ID
        if caption == "※※※":
            continue
        
        item = {
            "id": hashed_id,
            "image": full_path,
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\n請幫我截取出圖片中下方的語音字幕"
                },
                {
                    "from": "gpt",
                    "value": caption
                }
            ]
        }
        data.append(item)

    # 將數據寫入JSON文件
    output_file = os.path.join(output_dir, "train_output.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"數據已保存到 {output_file}")

# 使用示例
file_path = "/mnt/sda1/htchang/DL/HW3/train_data/out.txt"  # 你的輸入文件路徑
output_dir = "/mnt/sda1/htchang/DL/HW3/train_data"  # 輸出目錄

# 如果輸出目錄不存在，則創建它
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

process_file(file_path, output_dir)