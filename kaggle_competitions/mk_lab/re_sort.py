import pandas as pd
import re

# 讀取 CSV 檔案
df = pd.read_csv('output.csv')

# 定義一個函數來提取排序的關鍵部分
def extract_key(filename):
    match = re.match(r"video(\d+)_(\d+)\.jpg", filename)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return (0, 0)

# 使用提取的關鍵部分進行排序
df['key'] = df['filename'].apply(extract_key)
df = df.sort_values('key').drop('key', axis=1)

# 將排序後的 DataFrame 存回新的 CSV 檔案
df.to_csv('sorted_output.csv', index=False)
