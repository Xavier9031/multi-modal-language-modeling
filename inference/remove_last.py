import pandas as pd

# 读取CSV文件
csv_file_path = "/mnt/sda1/htchang/DL/HW3/inference/results_h1.csv"
df = pd.read_csv(csv_file_path)

# 定义函数去除最后一个逗号或句号
def remove_last_comma_period(text):
    if text.endswith(',') or text.endswith('，'):
        return text[:-1]
    elif text.endswith('.') or text.endswith('。'):
        return text[:-1]
    elif text.endswith('�'):
        return text[:-1]
    return text

# 移除 .jpg 并应用函数到文本列
df['id'] = df['id'].str.replace('.jpg', '')
df['text'] = df['text'].apply(remove_last_comma_period)

# 保存处理后的结果到新的CSV文件
output_csv_file_path = "/mnt/sda1/htchang/DL/HW3/inference/results_h1_remove_last.csv"
df.to_csv(output_csv_file_path, index=False)

print("Processed CSV saved to:", output_csv_file_path)
