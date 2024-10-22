import pandas as pd
import re
from tqdm import tqdm

# 读取CSV文件
def process_csv(file_path):
    df = pd.read_csv(file_path)
    return df['譯義']

# 读取TXT文件
def process_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return lines

# 提取「譯義」栏位中每个单独的意思
def extract_meanings(translation):
    # 去除前面的数字和点号
    cleaned_translation = re.sub(r'\d+\.\s*', '', translation)
    # 根据"。"和换行符号分割成多个意思
    meanings = re.split(r'[。，\n]', cleaned_translation)
    # 去除每个意思结尾的标点符号，并且过滤掉不符合长度要求或以标点符号开头的词义
    filtered_meanings = []
    for meaning in meanings:
        meaning = meaning.strip()
        meaning = re.sub(r'[，。！？；：,.!?;:]+$', '', meaning)
        # 替换所有标点符号为空格
        meaning = re.sub(r'[，。！？；：,.!?;:．、《》「」『』【】（）()〈〉❖─]+', ' ', meaning)
        # 把第一个空格替换成空
        meaning = re.sub(r'^\s+', '', meaning)
        if 5 <= len(meaning) <= 30 and not re.match(r'^[，。！？；：,.!?;:]', meaning):
            filtered_meanings.append(meaning)
    return filtered_meanings

def main(file_path, file_type):
    if file_type == 'csv':
        translations = process_csv(file_path)
    elif file_type == 'txt':
        translations = process_txt(file_path)
    else:
        raise ValueError("文件類型必須是 'csv' 或 'txt'")

    meanings = []
    for translation in tqdm(translations, desc="提取詞義進度"):
        meanings.extend(extract_meanings(translation))

    # 将提取的词义写入txt文件
    output_file = '/mnt/sda1/htchang/DL/HW3/online_txt/meanings_processed.txt'
    with open(output_file, 'w', encoding='utf-8') as file:
        for meaning in meanings:
            file.write(meaning + '\n')

    print(f'提取的詞義已寫入 {output_file}')

# 修改為你的文件路徑和文件類型
file_path = '/mnt/sda1/htchang/DL/HW3/online_txt/all_txt.txt'
file_type = 'txt'  # 'csv' 或 'txt'

main(file_path, file_type)