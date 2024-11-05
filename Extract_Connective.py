import os
import spacy
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

sns.set_style('whitegrid')

plt.rcParams['font.size'] = 23
plt.rcParams['figure.dpi'] = 512
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载NLP模型
#print("正在加载NLP模型...")
nlp = spacy.load('en_core_web_sm')

# 从JSON文件读取衔接词词典
# print("正在读取衔接词词典...")
with open("connector_dict.json", "r", encoding="utf-8") as f:
    connector_dict = json.load(f)

# 创建保存输出和图片的目录
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)


# 统计每篇文章中的衔接词
def extract_connectors(text, connector_dict):
    doc = nlp(text)
    connectors_count = {category: {subcategory: 0 for subcategory in subcategories} for category, subcategories in
                        connector_dict.items()}
    all_connectors = []

    for token in doc:
        for category, subcategories in connector_dict.items():
            for subcategory, connectors in subcategories.items():
                if token.text.lower() in connectors:
                    connectors_count[category][subcategory] += 1
                    all_connectors.append(token.text.lower())

    return connectors_count, all_connectors


# 可视化函数，柱状图显示各类作文的衔接词数量，横坐标分为低分、中分、高分
def plot_connectors(grade_counts, title, identifier):
    categories = list(grade_counts['低分'].keys())
    grades = ['低分', '中分', '高分']

    data = []
    for grade in grades:
        counts = [sum(grade_counts[grade][category].values()) for category in categories]
        data.append(counts)

    data = pd.DataFrame(data, index=grades, columns=categories)

    # 绘制柱状图
    ax = data.plot(kind='bar', figsize=(12, 8), width=0.7)
    plt.title(title, fontsize=23, pad=20)
    plt.xlabel('分数段', fontsize=18)
    plt.ylabel('数量', fontsize=18)

    # 设置图例在右上角
    ax.legend(title='衔接词类别', loc='upper right', fontsize=14, title_fontsize='16')

    # 添加水平网格线
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    ax.xaxis.grid(False)

    # 设置横坐标标签水平显示
    plt.xticks(rotation=0)

    # 保存图像文件
    file_name = f"{output_dir}/{title.replace(' ', '_')}_{identifier}.png"
    plt.tight_layout()
    #plt.savefig(file_name)
    print(f"已保存柱状图: {file_name}")
    plt.show()


# 绘制各类作文的衔接词比例柱状图
def plot_pie_chart(grade_counts, title, identifier):
    categories = list(grade_counts['低分'].keys())
    grades = ['低分', '中分', '高分']

    data = []
    for grade in grades:
        total_count = sum([sum(grade_counts[grade][category].values()) for category in categories])
        proportions = [sum(grade_counts[grade][category].values()) / total_count for category in categories]
        data.append(proportions)

    data = pd.DataFrame(data, index=grades, columns=categories)

    # 绘制柱状图
    ax = data.plot(kind='bar', figsize=(12, 8), width=0.7)
    plt.title(title, fontsize=23, pad=20)
    plt.xlabel('分数段', fontsize=18)
    plt.ylabel('比例', fontsize=18)

    # 设置图例在右上角
    ax.legend(title='衔接词类别', loc='upper right', fontsize=14, title_fontsize='16')

    # 添加水平网格线
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    ax.xaxis.grid(False)

    # 设置横坐标标签水平显示
    plt.xticks(rotation=0)

    # 保存图像文件
    file_name = f"{output_dir}/{title.replace(' ', '_')}_proportion_{identifier}.png"
    plt.tight_layout()
    #plt.savefig(file_name)
    print(f"已保存比例柱状图: {file_name}")
    plt.show()


# 处理单个dev_labeled文件，保存输出信息到文件
def process_dev_file(file_path, fold):
    print(f"正在处理 {file_path}...")
    data = pd.read_excel(file_path)
    output_file = os.path.join(output_dir, f"fold_{fold}_output.txt")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"正在处理 {file_path}...\n")

        category_counts = {category: {subcategory: 0 for subcategory in subcategories} for category, subcategories in
                           connector_dict.items()}

        for text in data['text']:
            connectors_count, _ = extract_connectors(text, connector_dict)
            for category, subcategories in connectors_count.items():
                for subcategory, count in subcategories.items():
                    category_counts[category][subcategory] += count

        # 打印并保存各个衔接词类别及其总个数
        f.write(f"{os.path.basename(file_path)} 中的衔接词数量:\n")
        for category, subcategories in category_counts.items():
            total_category_count = sum(subcategories.values())
            f.write(f"{category} (总计: {total_category_count}):\n")
            print(f"{category} (总计: {total_category_count}):")
            for subcategory, count in subcategories.items():
                f.write(f"  {subcategory}: {count}\n")
                print(f"  {subcategory}: {count}")


# 新增绘制平均衔接词数量柱状图的函数
def plot_avg_connectors(avg_counts_per_category, title, identifier):
    categories = list(avg_counts_per_category['低分'].keys())
    grades = ['低分', '中分', '高分']

    # 将数据转化为DataFrame格式
    data = pd.DataFrame([avg_counts_per_category[grade] for grade in grades], index=grades, columns=categories)

    # 绘制柱状图
    ax = data.plot(kind='bar', figsize=(12, 8), width=0.7)
    plt.title(title, fontsize=23, pad=20)
    plt.xlabel('分数段', fontsize=18)
    plt.ylabel('平均数量', fontsize=18)

    # 设置图例在右上角
    ax.legend(title='衔接词类别', loc='upper right', fontsize=14, title_fontsize='16')

    # 添加水平网格线
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    ax.xaxis.grid(False)

    # 设置横坐标标签水平显示
    plt.xticks(rotation=0)

    # 保存图像文件
    file_name = f"{output_dir}/{title.replace(' ', '_')}_{identifier}.png"
    plt.tight_layout()
    plt.savefig(file_name)
    print(f"已保存平均数量柱状图: {file_name}")
    plt.show()

# 处理三类评分文件，保存输出信息到文件
def process_by_grade(fold_path, fold):
    print(f"正在处理fold_{fold}中的评分文件...")
    low_file = os.path.join(fold_path, 'dev_labeled_grade_1.xlsx')
    mid_file = os.path.join(fold_path, 'dev_labeled_grade_2.xlsx')
    high_file = os.path.join(fold_path, 'dev_labeled_grade_3.xlsx')

    files = {'低分': low_file, '中分': mid_file, '高分': high_file}
    output_file = os.path.join(output_dir, f"fold_{fold}_grade_output.txt")

    grade_counts = {grade: {category: {subcategory: 0 for subcategory in subcategories} for category, subcategories in
                            connector_dict.items()} for grade in files.keys()}

    avg_counts_per_category = {'低分': {}, '中分': {}, '高分': {}}

    with open(output_file, "w", encoding="utf-8") as f:
        for grade, file_path in files.items():
            print(f"正在处理来自 {file_path} 的{grade} 作文")
            data = pd.read_excel(file_path)
            f.write(f"\n正在处理来自 {file_path} 的{grade} 作文\n")

            all_connectors = []
            essay_count = len(data)

            for text in data['text']:
                connectors_count, essay_connectors = extract_connectors(text, connector_dict)
                for category, subcategories in connectors_count.items():
                    for subcategory, count in subcategories.items():
                        grade_counts[grade][category][subcategory] += count
                all_connectors.extend(essay_connectors)

            # 保存每个类别的每个衔接词的总个数
            f.write(f"{grade}作文中的衔接词数量:\n")
            for category, subcategories in grade_counts[grade].items():
                total_category_count = sum(subcategories.values())
                f.write(f"{category} (总计: {total_category_count}):\n")
                print(f"{category} (总计: {total_category_count}):")
                for subcategory, count in subcategories.items():
                    f.write(f"  {subcategory}: {count}\n")
                    print(f"  {subcategory}: {count}")

            # 计算并保存平均衔接词数量
            avg_counts_per_category[grade] = {
            category: sum(subcategories.values()) / essay_count for category, subcategories in
            grade_counts[grade].items()
        }

            # 保存前10个最常见的衔接词
            connector_freq = Counter(all_connectors)
            f.write(f"{grade}作文中最常见的10个衔接词: {connector_freq.most_common(10)}\n")
            print(f"{grade}作文中最常见的10个衔接词: {connector_freq.most_common(10)}")

            # 打印平均衔接词数量
            print(f"\n{fold_path}中每篇文章的四大类衔接词平均数量：")
            for grade, counts in avg_counts_per_category.items():
                print(f"{grade}平均数量：", counts)
                f.write(f"\n{grade}平均数量：{counts}\n")

    # 可视化并保存图表
    plot_connectors(grade_counts, f"各类作文中的衔接词类别分布", f"fold_{fold}")
    plot_pie_chart(grade_counts, f"各类作文中的衔接词比例", f"fold_{fold}")
    plot_avg_connectors(avg_counts_per_category, f"各类作文中的平均衔接词数量", f"fold_{fold}")


# 处理所有fold下的文件
def process_all_folds(data_dir):
    print("开始处理所有fold下的文件...")
    for fold in range(5):  # 假设有5个fold
        fold_path = os.path.join(data_dir, f'fold_{fold}')
        process_by_grade(fold_path, fold)


# 主程序
if __name__ == "__main__":
    data_dir = "data"
    process_all_folds(data_dir)
    print("所有文件处理完毕！")
