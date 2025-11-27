import os
import glob
import torch
import random
import cv2 as cv
import numpy as np
from typing import Tuple, Dict
from torch.utils.data import Dataset
from torchvision import transforms


# 化学数据集类
# 读取图片，读取YOLO标签，classes.txt，labels.txt
class ChemDataset(Dataset):
    def __init__(self,
                 root_dir: str = ".",
                 image_dir: str = "images",
                 label_dir: str = "labels",
                 classes_file: str = "classes.txt",
                 labels_file: str = "labels.txt",  # 化学方程式文本标签文件
                 target_size: Tuple[int, int] = (256, 256),  # 图像大小
                 train: bool = True,  # 是否为训练模式
                 augment: bool = False,  # 数据增强
                 random_graph: bool = True
                 ):
        # 初始化参数
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, image_dir)
        self.label_dir = os.path.join(root_dir, label_dir)
        self.target_size = target_size
        self.train = train
        self.augment = augment
        self.random_graph = random_graph

        # 检查目录是否存在
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"图像目录不存在: {self.image_dir}")
        if not os.path.exists(self.label_dir):
            raise FileNotFoundError(f"标签目录不存在: {self.label_dir}")

        # 加载类别
        self.classes = self._load_classes(os.path.join(root_dir, classes_file))
        # 加载词汇表，字符转为索引，词汇表的长度
        self.vocab_dict = {char: idx for idx, char in enumerate(self.classes)}
        # 加载化学方程式标签，标签字典
        self.labels_dict = self._load_labels(os.path.join(root_dir, labels_file))
        # 收集图像和标签文件
        self.image_paths, self.label_paths = self._collect_pairs()

        # 图像预处理变换
        self.transform = transforms.Compose([
            # 转换为Tensor类型
            transforms.ToTensor(),
            # 归一化到[-1,1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    # 加载词汇表类别
    def _load_classes(self, classes_file: str):
        # 判断classes_file是否存在
        if os.path.exists(classes_file):
            with open(classes_file, "r", encoding="utf-8") as f:
                classes = [line.strip() for line in f if line.strip()]
                if not classes or classes[0] != "":
                    classes.insert(0, "")
                return classes
        # 默认词汇表
        return ["", "H", "O", "C", "N", "+", "=", "2", "3", "4", "5", "6", "7", "8", "9"]

    # 加载化学方程式标签
    def _load_labels(self, labels_file: str) -> Dict[str, str]:
        labels_dict = {}
        if os.path.exists(labels_file):
            with open(labels_file, "r", encoding="utf-8") as f:
                # 遍历每一行
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t", 1)
                    if len(parts) == 2:
                        # 获取图像名和化学方程式
                        img_name, formula = parts
                        img_name = os.path.splitext(img_name)[0]
                        labels_dict[img_name] = formula
        return labels_dict

    # 将化学方程式文本转为序列
    def _text_to_tokens(self, text: str) -> torch.Tensor:
        # 如果为空
        if not text:
            return torch.tensor([0], dtype=torch.long)  # blank token

        tokens = []
        i = 0
        text_len = len(text)

        # 按长度从长到短排序的特殊token（避免短token匹配长token的前缀）
        special_tokens = sorted([token for token in self.vocab_dict.keys() if len(token) > 1],
                                key=len, reverse=True)

        while i < text_len:
            # 是否匹配
            matched = False

            # 先尝试匹配多字符token（最长匹配原则）
            for token in special_tokens:
                if i + len(token) <= text_len and text[i:i + len(token)] == token:
                    tokens.append(self.vocab_dict[token])
                    i += len(token)
                    matched = True
                    break

            # 如果没有匹配到多字符token
            if not matched:
                char = text[i]
                if char in self.vocab_dict:
                    tokens.append(self.vocab_dict[char])
                # 如果字符不在词汇表中，则跳过
                i += 1

        # 如果没有任何token，返回blank token
        if not tokens:
            # 返回空白的token序列
            return torch.tensor([0], dtype=torch.long)
        return torch.tensor(tokens, dtype=torch.long)

    # 收集图像和标签文本，保留一个同时存在图像和标签的label
    def _collect_pairs(self):
        image_files = []
        label_files = []
        for ext in ("*.jpg", "*.png", "*.jpeg", "*.bmp"):
            # 匹配图像文件
            image_files.extend(glob.glob(os.path.join(self.image_dir, ext)))
        image_files.sort()

        # 只保留有对应标签文件的图像
        valid_pairs = []
        for img in image_files:
            name = os.path.splitext(os.path.basename(img))[0]
            label = os.path.join(self.label_dir, name + ".txt")

            if os.path.exists(label):
                valid_pairs.append((img, label))

        if valid_pairs:
            # 解压
            image_files, label_files = zip(*valid_pairs)
            return list(image_files), list(label_files)
        else:
            # 返回空列表
            return [], []

    def __len__(self):
        return len(self.image_paths)

    # YOLO标签的解析
    def _parse_label(self, label_path, width, height):
        # 初始化边界框列表
        boxes = []
        if label_path and os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    # 空格分割行的内容
                    parts = line.strip().split()
                    # 标签中有5个内容(类，xc，yc，w，h)（边界框）
                    if len(parts) != 5:
                        continue
                    cls, xc, yc, w, h = parts
                    # 将归一化的坐标转换为绝对坐标
                    xc = float(xc) * width
                    yc = float(yc) * height
                    w = float(w) * width
                    h = float(h) * height
                    # 计算边界框坐标
                    xmin = max(0, xc - w / 2)  # 左上角x
                    ymin = max(0, yc - h / 2)  # 左上角y
                    xmax = min(width, xc + w / 2)  # 右下角x
                    ymax = min(height, yc + h / 2)  # 右下角y
                    boxes.append([int(cls), xmin, ymin, xmax, ymax])

        # 如果没有检测框，返回一个空的检测框
        if not boxes:
            boxes = [[0, 0, 0, 0, 0]]
        return torch.tensor(boxes, dtype=torch.float32)

    # 加载图像
    def _load_image(self, path):
        img = cv.imread(path)
        if img is None:
            raise ValueError(f"无法读取图像")
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        return img

    # 调整边界框
    def _resize_boxes(self, boxes, scale_x, scale_y):
        boxes[:, 1] *= scale_x
        boxes[:, 2] *= scale_y
        boxes[:, 3] *= scale_x
        boxes[:, 4] *= scale_y
        return boxes

    # 随机生成图数据
    def _random_graph(self, info):
        num_nodes = max(4, info.get("num_objects", 4))
        node_types = torch.randint(0, len(self.classes), (num_nodes,))
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        edge_types = torch.randint(0, 4, (edge_index.size(1),))
        return {
            "node_types": node_types,
            "edge_index": edge_index,
            "edge_types": edge_types,
        }

    # 获取样本 idx:索引
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        # 加载图像
        image = self._load_image(img_path)
        # 获取图片的宽高
        h, w = image.shape[:2]
        # 加载标签，获得边界框
        boxes = self._parse_label(label_path, w, h)

        # 数据增强，训练模式
        if self.augment and self.train:
            # 随机概率为0.5
            # 随机水平翻转
            if random.random() > 0.5:
                image = cv.flip(image, 1)
                # 调整坐标
                boxes[:, [1, 3]] = w - boxes[:, [3, 1]]

            # 随机亮度
            if random.random() > 0.5:
                alpha = random.uniform(0.7, 1.3)
                # 调整亮度，clip:裁剪
                image = np.clip(image * alpha, 0, 255).astype(np.uint8)

            # 随机对比度
            if random.random() > 0.5:
                alpha = random.uniform(0.8, 1.2)
                beta = random.uniform(-15, 15)
                # convertScaleAbs:对比度
                image = cv.convertScaleAbs(image, alpha=alpha, beta=beta)

            # 随机旋转
            if random.random() > 0.5:
                angle = random.uniform(-5, 5)
                # 中心点
                center = (w // 2, h // 2)
                # 旋转矩阵的计算
                M = cv.getRotationMatrix2D(center, angle, 1.0)
                image = cv.warpAffine(image, M, (w, h), borderMode=cv.BORDER_REPLICATE)

            # 随机缩放
            if random.random() > 0.5:
                scale = random.uniform(0.95, 1.05)
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv.resize(image, (new_w, new_h))
                image = cv.resize(image, (w, h))

            # 随机高斯噪声
            if random.random() > 0.5:
                noise = np.random.normal(0, 5, image.shape).astype(np.float32)
                image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            # 随机模糊
            if random.random() > 0.3:  # 随机概率为0.3
                kernel_size = random.choice([3, 5])
                # 高斯模糊
                image = cv.GaussianBlur(image, (kernel_size, kernel_size), 0)

        # 调整图像大小
        image_resized = cv.resize(image, self.target_size)
        scale_x = self.target_size[0] / w
        scale_y = self.target_size[1] / h
        boxes = self._resize_boxes(boxes.clone(), scale_x, scale_y)

        # 应用图像变换
        tensor_img = self.transform(image_resized)

        # 获取化学方程式文本标签
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        formula_text = self.labels_dict.get(img_name, "")

        # 转换为token序列
        if formula_text:
            formula_tokens = self._text_to_tokens(formula_text)
        else:
            # 如果没有文本标签，尝试从boxes的类别构建（fallback）
            if boxes.size(0) > 0 and boxes[0, 0] > 0:
                formula_tokens = boxes[:, 0].long()
            else:
                formula_tokens = torch.tensor([0], dtype=torch.long)  # blank token

        # 构建样本信息字典
        sample_info = {
            "image_path": img_path,
            "original_size": (w, h),
            "num_objects": (boxes.size(0) if boxes is not None else 0),
            "formula_text": formula_text,
            "formula_tokens": formula_tokens,
        }

        # 生成图数据
        graph_data = self._random_graph(sample_info) if self.random_graph else None

        return tensor_img, boxes, sample_info, graph_data

# 整理函数
def collate_fn(batch):
    images, targets, infos, graphs = zip(*batch)
    # 堆叠
    images = torch.stack(images, dim=0)
    # 转换为列表
    targets = list(targets)
    infos = list(infos)
    graphs = list(graphs)
    return images, targets, infos, graphs
