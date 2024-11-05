import os
import re

import pandas as pd

high_score = [10, 5, 3, 3, 3, 3, 21, 41]
mid_score = [5, 3, 2, 2, 2, 2, 10, 20]

# ASAP评分范围字典
asap_ranges = {
    1: (2.0, 12.0),
    2: (1.0, 6.0),
    3: (0.0, 3.0),
    4: (0.0, 3.0),
    5: (0.0, 4.0),
    6: (0.0, 4.0),
    7: (0.0, 30.0),
    8: (0.0, 60.0),
}

# 根据每个set_id的评分范围进行标准化并划分档次
def assign_grade_labels_by_set(data):
    # 处理每一行，根据set_id对应的评分范围进行归一化和分档
    for row in data:
        set_id = int(row[1])  # set_id
        score = float(row[3]) if row[3].strip() else None  # score列为第4列

        if set_id in asap_ranges and score is not None:
            min_score, max_score = asap_ranges[set_id]
            high_score_start = high_score[set_id - 1] # 高分开始的分数
            mid_score_start = mid_score[set_id - 1]

            # 将分数划分为三个档次
            if score >= high_score_start :
                row.append(3)  # 档次3
            elif score >= mid_score_start:
                row.append(2)  # 档次2
            else:
                row.append(1)  # 档次1
        else:
            row.append(None)  # 如果没有对应的范围，或无效分数，跳过
            print(f"Invalid score {score} for set_id {set_id}!")
    return data


# 处理每个文件（train.tsv, dev.tsv, test.tsv），并将其保存为xlsx格式
def process_file(file_path):
    if not os.path.exists(file_path):
        print(f"{file_path} does not exist.")
        return

    # 逐行读取文件，并只保留所需的列（id, set_id, text, score）
    with open(file_path, 'r', encoding='utf-8') as file:
        data = []
        for line in file.readlines():
            columns = re.split(r'\t+', line.strip())  # 分割列
            if len(columns) >= 6:  # 确保有足够的列
                # 选择需要的列 (id, set_id, text, score)
                selected_columns = [columns[0], columns[1], columns[2], columns[5]]  # 选择id, set_id, text, score
                # print(selected_columns)
                data.append(selected_columns)

    # 为每行数据分配档次标签
    data_with_labels = assign_grade_labels_by_set(data)

    # 将数据转换为 pandas DataFrame
    df = pd.DataFrame(data_with_labels, columns=['id', 'set_id', 'text', 'score', 'grade_label'])

    # 保存为Excel文件
    output_file = file_path.replace('.tsv', '_labeled.xlsx')
    df.to_excel(output_file, index=False)

    print(f"Processed and saved labeled data to {output_file}")


# 遍历fold_0到fold_4的文件夹，并处理其中的train.tsv, dev.tsv, test.tsv
def categorize_folds(base_dir):
    for fold_num in range(5):  # fold_0到fold_4
        fold_dir = os.path.join(base_dir, f'fold_{fold_num}')
        if os.path.exists(fold_dir):
            for file_type in ['train.tsv', 'dev.tsv', 'test.tsv']:
                file_path = os.path.join(fold_dir, file_type)
                process_file(file_path)
        else:
            print(f"Folder {fold_dir} does not exist")


if __name__ == "__main__":
    base_dir = './data'  # 假设fold_0到fold_4在data文件夹下
    categorize_folds(base_dir)
