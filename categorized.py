# -*- coding: utf-8 -*-
"""
@Time: 2024/10/18 14:34
@Auth: Zhang Hongxing
@File: categorized.py
@Note:   
"""
# 读取数据
import os

import pandas as pd


def process_file(file_path):
    if not os.path.exists(file_path):
        print(f"{file_path} does not exist.")
        return

    # 读取xlsx
    data = pd.read_excel(file_path)
    # 按照grade_label进行分组
    grouped = data.groupby('grade_label')
    a=0
    b=0
    c=0
    for grade_label, group in grouped:
        print(f"Grade {grade_label}: {len(group)} samples")
        if grade_label==1:
            a+=len(group)
        if grade_label==2:
            b+=len(group)
        if grade_label==3:
            c+=len(group)
    # 打印比例
    ratio_a = a / len(data)
    ratio_b = b / len(data)
    ratio_c = c / len(data)
    print(f"Grade 1: {ratio_a * 100:.2f}%")
    print(f"Grade 2: {ratio_b * 100:.2f}%")
    print(f"Grade 3: {ratio_c * 100:.2f}%")

    # 每个组保存为一个xlsx
    for grade_label, group in grouped:
        grade_file_path = file_path.replace('.xlsx', f'_grade_{grade_label}.xlsx')
        group.to_excel(grade_file_path, index=False)
        print(f"Saved to {grade_file_path}")
        # 保存为tsv
        grade_file_path = file_path.replace('.xlsx', f'_grade_{grade_label}.tsv')
        group.to_csv(grade_file_path, sep='\t', index=False, header=False)

def categorize_folds(base_dir):
    for fold_num in range(5):  # fold_0到fold_4
        fold_dir = os.path.join(base_dir, f'fold_{fold_num}')
        if os.path.exists(fold_dir):
            for file_type in ['train_labeled.xlsx', 'dev_labeled.xlsx', 'test_labeled.xlsx']:
                file_path = os.path.join(fold_dir, file_type)
                process_file(file_path)
        else:
            print(f"Folder {fold_dir} does not exist")


if __name__ == "__main__":
    base_dir = './data'  # 假设fold_0到fold_4在data文件夹下
    categorize_folds(base_dir)
