#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集质量检查脚本
检查图像和标签的匹配情况、标签准确性、词汇表完整性等
"""
import os
import glob
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

def load_vocabulary(classes_file: str) -> Dict[str, int]:
    """加载词汇表"""
    vocab = {}
    if os.path.exists(classes_file):
        with open(classes_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                char = line.strip()
                if char:
                    vocab[char] = idx
    # 确保blank token存在
    if "" not in vocab:
        vocab[""] = 0
    return vocab

def load_labels(labels_file: str) -> Dict[str, str]:
    """加载标签文件"""
    labels_dict = {}
    if os.path.exists(labels_file):
        with open(labels_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    img_name, formula = parts
                    img_name = os.path.splitext(img_name)[0]
                    labels_dict[img_name] = formula
                else:
                    print(f"警告: 第{line_num}行格式不正确: {line[:50]}")
    return labels_dict

def check_token_coverage(text: str, vocab: Dict[str, int]) -> Tuple[List[str], bool]:
    """检查文本中的字符是否都在词汇表中"""
    missing_chars = []
    all_covered = True
    
    # 先检查多字符token
    special_tokens = sorted([token for token in vocab.keys() if len(token) > 1], 
                           key=len, reverse=True)
    
    i = 0
    text_len = len(text)
    checked_positions = set()
    
    while i < text_len:
        matched = False
        
        # 尝试匹配多字符token
        for token in special_tokens:
            if i + len(token) <= text_len and text[i:i+len(token)] == token:
                for pos in range(i, i + len(token)):
                    checked_positions.add(pos)
                i += len(token)
                matched = True
                break
        
        # 如果没有匹配到多字符token，检查单字符
        if not matched:
            char = text[i]
            if char not in vocab:
                if char not in missing_chars:
                    missing_chars.append(char)
                all_covered = False
            checked_positions.add(i)
            i += 1
    
    return missing_chars, all_covered

def main():
    print("=" * 60)
    print("数据集质量检查报告")
    print("=" * 60)
    
    # 1. 检查图像文件
    image_dir = "images"
    image_files = glob.glob(os.path.join(image_dir, "*.jpg"))
    image_names = {os.path.splitext(os.path.basename(f))[0] for f in image_files}
    print(f"\n1. 图像文件统计:")
    print(f"   图像总数: {len(image_files)}")
    
    # 2. 检查标签文件
    labels_file = "labels.txt"
    labels_dict = load_labels(labels_file)
    print(f"\n2. 标签文件统计:")
    print(f"   标签总数: {len(labels_dict)}")
    
    # 3. 检查匹配情况
    matched_images = image_names & set(labels_dict.keys())
    unmatched_images = image_names - set(labels_dict.keys())
    unmatched_labels = set(labels_dict.keys()) - image_names
    
    print(f"\n3. 图像-标签匹配情况:")
    print(f"   匹配的图像: {len(matched_images)}")
    print(f"   无标签的图像: {len(unmatched_images)}")
    print(f"   无图像的标签: {len(unmatched_labels)}")
    
    if unmatched_images:
        print(f"\n   前10个无标签的图像示例:")
        for img in list(unmatched_images)[:10]:
            print(f"     - {img}")
    
    if unmatched_labels:
        print(f"\n   前10个无图像的标签示例:")
        for label in list(unmatched_labels)[:10]:
            print(f"     - {label}")
    
    # 4. 检查词汇表
    classes_file = "classes.txt"
    vocab = load_vocabulary(classes_file)
    print(f"\n4. 词汇表统计:")
    print(f"   词汇表大小: {len(vocab)}")
    print(f"   包含blank token: {'' in vocab}")
    
    # 检查classes.txt中的空行
    with open(classes_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        empty_lines = [i+1 for i, line in enumerate(lines) if not line.strip()]
        if empty_lines:
            print(f"   警告: classes.txt中有{len(empty_lines)}个空行 (行号: {empty_lines[:10]})")
    
    # 5. 检查标签中的字符覆盖情况
    print(f"\n5. 标签字符覆盖检查:")
    all_missing_chars = Counter()
    labels_with_missing = []
    label_lengths = []
    
    for img_name, formula in labels_dict.items():
        if img_name not in matched_images:
            continue
        
        missing_chars, all_covered = check_token_coverage(formula, vocab)
        label_lengths.append(len(formula))
        
        if not all_covered:
            labels_with_missing.append((img_name, formula, missing_chars))
            for char in missing_chars:
                all_missing_chars[char] += 1
    
    print(f"   检查的标签数: {len(matched_images)}")
    print(f"   有缺失字符的标签: {len(labels_with_missing)} ({len(labels_with_missing)/len(matched_images)*100:.2f}%)")
    
    if all_missing_chars:
        print(f"\n   缺失的字符 (出现次数):")
        for char, count in all_missing_chars.most_common(20):
            print(f"     '{char}': {count}次")
    
    if labels_with_missing:
        print(f"\n   前10个有缺失字符的标签示例:")
        for img_name, formula, missing in labels_with_missing[:10]:
            print(f"     {img_name}: {formula[:50]}... (缺失: {missing})")
    
    # 6. 标签长度统计
    if label_lengths:
        print(f"\n6. 标签长度统计:")
        print(f"   平均长度: {sum(label_lengths)/len(label_lengths):.2f}")
        print(f"   最短: {min(label_lengths)}")
        print(f"   最长: {max(label_lengths)}")
        print(f"   长度分布:")
        length_dist = Counter(label_lengths)
        for length in sorted(length_dist.keys())[:20]:
            print(f"     长度{length}: {length_dist[length]}个")
    
    # 7. 检查重复标签
    formula_counts = Counter(labels_dict.values())
    duplicates = {formula: count for formula, count in formula_counts.items() if count > 1}
    if duplicates:
        print(f"\n7. 重复标签统计:")
        print(f"   唯一标签数: {len(formula_counts)}")
        print(f"   重复标签数: {len(duplicates)}")
        print(f"   最常见的20个标签:")
        for formula, count in formula_counts.most_common(20):
            print(f"     {formula[:50]}: {count}次")
    
    # 8. 检查特殊字符的使用情况
    print(f"\n8. 特殊字符使用统计:")
    special_chars = ['_2', '_3', '_4', '_5', '_6', '_7', 
                    '\\~=', '\\$=', '\\@=', '\\&=', '\\*=',
                    '|+', '|2+', '|3+', '|4+', '|5+', '|6+', '|7+',
                    '|-', '|2-', '|3-', '|4-', '|5-', '|6-', '|7-']
    
    special_usage = Counter()
    for formula in labels_dict.values():
        for special in special_chars:
            if special in formula:
                special_usage[special] += 1
    
    for special, count in special_usage.most_common(20):
        print(f"   {special}: {count}次")

    suggestions = []
    
    if len(unmatched_images) > 0:
        suggestions.append(f"   - 有{len(unmatched_images)}张图像没有标签，建议补充标签或移除图像")
    
    if len(unmatched_labels) > 0:
        suggestions.append(f"   - 有{len(unmatched_labels)}个标签没有对应图像，建议检查标签文件")

if __name__ == "__main__":
    main()

