# coding: UTF-8
"""
统计所有项目的-DFS-Selected60-filter数据集的样本总数
"""
import os
from collections import defaultdict

# 项目列表
dataset_list = ['Ant', 'ArgoUML', 'Columba', 'Hibernate', 'JEdit', 
                'JFreeChart', 'JMeter', 'JRuby', 'SQuirrel']

# 数据集后缀
dataset_suffix = '-DFS-Selected60-filter'

# 统计结果
results = defaultdict(lambda: {'train': 0, 'dev': 0, 'test': 0, 'total': 0})
grand_total = {'train': 0, 'dev': 0, 'test': 0, 'total': 0}

print("="*80)
print("统计所有项目的-DFS-Selected60-filter数据集样本数")
print("="*80)
print(f"{'项目':<15} {'训练集':<10} {'验证集':<10} {'测试集':<10} {'总计':<10}")
print("-"*80)

for dataset in dataset_list:
    dataset_dir = os.path.join('dataset', f"{dataset}{dataset_suffix}")
    data_dir = os.path.join(dataset_dir, 'data')
    
    # 统计每个文件的行数
    for split in ['train', 'dev', 'test']:
        file_path = os.path.join(data_dir, f"{split}.txt")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='UTF-8') as f:
                count = sum(1 for line in f if line.strip())
                results[dataset][split] = count
                grand_total[split] += count
        else:
            print(f"警告: {file_path} 不存在")
    
    results[dataset]['total'] = (results[dataset]['train'] + 
                                 results[dataset]['dev'] + 
                                 results[dataset]['test'])
    grand_total['total'] += results[dataset]['total']
    
    # 打印每个项目的结果
    print(f"{dataset:<15} {results[dataset]['train']:<10} "
          f"{results[dataset]['dev']:<10} {results[dataset]['test']:<10} "
          f"{results[dataset]['total']:<10}")

print("-"*80)
print(f"{'总计':<15} {grand_total['train']:<10} "
      f"{grand_total['dev']:<10} {grand_total['test']:<10} "
      f"{grand_total['total']:<10}")
print("="*80)

# 详细统计
print("\n详细统计:")
print(f"训练集总样本数: {grand_total['train']:,}")
print(f"验证集总样本数: {grand_total['dev']:,}")
print(f"测试集总样本数: {grand_total['test']:,}")
print(f"所有数据集总样本数: {grand_total['total']:,}")

