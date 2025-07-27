import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor  # Faster R-CNN模型类
import torchvision.transforms as T  # ToPILImage基础转换
import xml.etree.ElementTree as ET  # 解析XML
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from tqdm import tqdm  # 进度条
import time
import albumentations as A  # 数据增强
from albumentations.pytorch import ToTensorV2
import cv2
from torchmetrics.detection import MeanAveragePrecision  # 计算mAP

# --- 0. 配置参数 ---
PASCAL_VOC_ROOT = "./VOCdevkit/VOC2007"
SELECTED_CLASSES = [  # 检测类别列表
    'background',
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
]
NUM_CLASSES = len(SELECTED_CLASSES)  # 总类别数

CONFIG = {
    "learning_rate": 0.005,
    "optimizer_type": "SGD",  # 优化器类型
    "batch_size": 4,
    "num_epochs": 20,  # 训练总轮数
    "max_train_images": None,  # 训练集
    "max_val_images": None,  # 验证集
    "data_augmentation_level": "medium",  # 数据增强级别: "none", "light", "medium"
    "scheduler_step_size": 7,  # 每隔多少epoch衰减一次
    "scheduler_gamma": 0.1,  # 衰减因子
    "img_size": (600, 600)  # 统一尺寸
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 图像转换操作
def get_albumentations_transforms(is_train, img_size=(600, 600), augmentation_level="medium"):
    """
    定义并返回图像增强和预处理流程。
    """

    common_post_transforms = [
        # 将图像统一缩放到指定的 img_size
        A.Resize(height=img_size[0], width=img_size[1], interpolation=cv2.INTER_LINEAR),
    ]

    # 归一化和转换为PyTorch Tensor
    final_transforms = [
        # 对图像进行归一化
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # 将NumPy数组格式的图像和边界框转换为PyTorch Tensor格式
        ToTensorV2(),
    ]

    if is_train:  # 如果是训练阶段则应用数据增强
        train_specific_transforms = []
        if augmentation_level == "light":  # 轻量级增强
            train_specific_transforms = [
                A.HorizontalFlip(p=0.5),
            ]
        elif augmentation_level == "medium":  # 中等级别增强
            train_specific_transforms = [
                # 随机裁剪图像的一个区域并调整大小，同时确保边界框的有效性
                # erosion_rate 控制边界框被侵蚀的程度，p是应用此增强的概率
                A.RandomSizedBBoxSafeCrop(width=img_size[1], height=img_size[0], erosion_rate=0.1, p=0.5),
                A.HorizontalFlip(p=0.5),  # 水平翻转
                # 随机调整图像的亮度、对比度、饱和度、色调
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),  # 添加高斯噪声
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),  # 模拟相机ISO噪声
                A.RandomBrightnessContrast(p=0.3),  # 随机调整亮度和对比度
            ]
        # 如果 augmentation_level 是 "none" 或其他未定义的值，则不应用额外的训练时增强

        # 组合所有训练阶段的转换操作
        return A.Compose(
            train_specific_transforms + common_post_transforms + final_transforms,
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.1,
                                     min_area=100)
        )
    else:
        return A.Compose(
            common_post_transforms + final_transforms,
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])
        )


# --- 1. 数据集类  ---
class VOCDataset(torch.utils.data.Dataset):
    """
    类用于加载和处理PASCAL VOC 2007数据。
    继承自 torch.utils.data.Dataset。
    """

    def __init__(self, root_dir, image_set_file_name, selected_class_names, is_train,
                 augmentation_level="medium", img_size=(600, 600), max_num_images=None):

        self.root_dir = root_dir
        self.is_train = is_train
        # 根据是否为训练集和指定的增强级别，获取相应的Albumentations转换流程
        self.transforms = get_albumentations_transforms(is_train=self.is_train,
                                                        img_size=img_size,
                                                        augmentation_level=augmentation_level)
        self.selected_class_names = selected_class_names
        # 创建从类别名称到整数索引的映射
        self.class_to_idx_map = {cls_name: i for i, cls_name in enumerate(selected_class_names)}

        # 构建到图片和标注文件的完整路径
        self.image_files_dir = os.path.join(root_dir, "JPEGImages")
        self.annotation_files_dir = os.path.join(root_dir, "Annotations")

        # 读取指定的图片集合文件
        image_set_full_path = os.path.join(root_dir, "ImageSets", "Main", image_set_file_name)
        with open(image_set_full_path) as f:
            all_image_ids_in_set = [line.strip() for line in f.readlines()]  # 获取所有图片ID

        self.image_file_paths = []  # 存储最终筛选后用于加载的图片文件路径
        self.raw_annotations_list = []  # 存储原始解析的标注信息，用于传递给Albumentations
        loaded_image_counter = 0

        print(f"正在为 '{image_set_file_name}' (is_train={is_train}, aug_level={augmentation_level}) 加载和筛选图片...")
        # 遍历指定图片集中的每个图片ID
        for img_id in tqdm(all_image_ids_in_set, desc="处理图片ID"):
            # 如果已达到最大图片数量限制，则停止加载
            if max_num_images is not None and loaded_image_counter >= max_num_images:
                break

            annotation_file_path = os.path.join(self.annotation_files_dir, img_id + ".xml")
            try:
                xml_tree = ET.parse(annotation_file_path)  # 解析XML标注文件
            except FileNotFoundError:
                # print(f"警告: 图片 {img_id} 的标注文件未找到，已跳过.")
                continue  # 如果找不到标注文件，则跳过该图片

            xml_root_node = xml_tree.getroot()

            current_image_boxes_pascal_voc = []  # 存储当前图片中所有物体的边界框 (Pascal VOC格式)
            current_image_class_indices = []  # 存储当前图片中所有物体的类别索引
            image_contains_selected_class = False  # 标记当前图片是否包含我们感兴趣的类别

            # 遍历XML文件中的每个<object>标签
            for obj_node in xml_root_node.findall("object"):
                class_name_from_xml = obj_node.find("name").text  # 获取物体类别名称

                # 检查该物体类别是否在我们选定的类别列表中，并且不是背景类
                if class_name_from_xml in self.selected_class_names and class_name_from_xml != 'background':
                    image_contains_selected_class = True  # 标记此图片包含目标物体

                    bbox_xml_node = obj_node.find("bndbox")  # 获取边界框节点
                    try:
                        # 提取边界框坐标 (xmin, ymin, xmax, ymax)
                        xmin, ymin, xmax, ymax = (float(bbox_xml_node.find(tag).text) for tag in
                                                  ["xmin", "ymin", "xmax", "ymax"])
                    except (TypeError, ValueError) as e:
                        # print(f"警告: 图片 {img_id} 中类别 {class_name_from_xml} 的边界框坐标格式错误. 已跳过此物体. 错误: {e}")
                        continue  # 如果坐标格式错误，则跳过此物体

                    # 检查边界框是否有效
                    if xmax <= xmin or ymax <= ymin:
                        # print(f"警告: 图片 {img_id} 中类别 {class_name_from_xml} 的边界框坐标无效. 已跳过此物体.")
                        continue  # 如果是无效框，则跳过

                    current_image_boxes_pascal_voc.append([xmin, ymin, xmax, ymax])
                    current_image_class_indices.append(self.class_to_idx_map[class_name_from_xml])  # 存储类别索引

            # 当前图片包含至少一个选定目标类，并提取到了有效边界框
            if image_contains_selected_class and len(current_image_boxes_pascal_voc) > 0:
                self.image_file_paths.append(os.path.join(self.image_files_dir, img_id + ".jpg"))
                self.raw_annotations_list.append({
                    "boxes_pascal_voc": current_image_boxes_pascal_voc,
                    "class_labels": current_image_class_indices,
                    "image_id_str": img_id
                })
                loaded_image_counter += 1

        # 打印最终加载的图片数量信息
        if loaded_image_counter == 0 and max_num_images is not None and max_num_images > 0:
            print(
                f"警告: 对于图片集 '{image_set_file_name}' (max_images={max_num_images})，未找到包含选定类别的图片. 数据集将为空.")
        elif loaded_image_counter < (
                max_num_images or float('inf')) and loaded_image_counter > 0 and max_num_images is not None:
            print(
                f"已加载 {loaded_image_counter} 张图片 (少于 max_images={max_num_images}) 给 '{image_set_file_name.split('.')[0]}' 数据集.")
        elif loaded_image_counter > 0:
            print(f"已加载 {loaded_image_counter} 张图片给 '{image_set_file_name.split('.')[0]}' 数据集.")

    def __getitem__(self, idx):
        """
        负责加载图像、应用转换，并返回转换后的图像和目标。
        """
        img_path = self.image_file_paths[idx]
        try:
            img_np = np.array(Image.open(img_path).convert("RGB"))
        except FileNotFoundError:
            print(f"错误: 图片文件 {img_path} 未找到. 将返回一个占位图片.")
            img_h, img_w = CONFIG["img_size"]
            img_np = np.zeros((img_h, img_w, 3), dtype=np.uint8);
            img_np[:, :, 0] = 255

            # 为占位图创建一个空的目标字典
            target_for_model = {"boxes": torch.zeros((0, 4), dtype=torch.float32),
                                "labels": torch.zeros(0, dtype=torch.int64),
                                "image_id": torch.tensor([idx]),
                                "area": torch.zeros(0, dtype=torch.float32),
                                "iscrowd": torch.zeros(0, dtype=torch.int64)}

            # 对占位图应用非增强的 Resize, Normalize, ToTensorV2
            placeholder_transforms = get_albumentations_transforms(is_train=False, augmentation_level="none",
                                                                   img_size=(img_h, img_w))
            augmented = placeholder_transforms(image=img_np, bboxes=[], class_labels=[])  # 空的bbox和label
            return augmented['image'], target_for_model

        # 获取该图片对应的原始标注信息
        raw_annotation = self.raw_annotations_list[idx]
        bboxes_pascal = raw_annotation["boxes_pascal_voc"]
        class_indices = raw_annotation["class_labels"]

        target_for_model = {}  # 初始化最终给模型的target字典

        try:
            # 应用Albumentations转换流程
            # image 参数是 NumPy 数组 (H,W,C)
            # bboxes 参数是 [[xmin, ymin, xmax, ymax], ...] 格式的列表
            # class_labels 参数是与bboxes对应的类别标签索引列表
            augmented = self.transforms(image=img_np, bboxes=bboxes_pascal, class_labels=class_indices)

            img_tensor = augmented['image']  # 获取转换后的图像Tensor
            augmented_bboxes = augmented['bboxes']  # 获取转换后的边界框列表
            augmented_labels = augmented['class_labels']  # 获取转换后的类别标签列表

            # 将增强后的bbox和label转换为Tensor格式
            if len(augmented_bboxes) > 0:
                target_for_model["boxes"] = torch.as_tensor(augmented_bboxes, dtype=torch.float32)
                target_for_model["labels"] = torch.as_tensor(augmented_labels, dtype=torch.int64)
            else:
                # 如果数据增强后没有有效的边界框
                target_for_model["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
                target_for_model["labels"] = torch.zeros(0, dtype=torch.int64)

        except Exception as e:
            print(
                f"Albumentations 增强图片 {img_path} (ID: {raw_annotation['image_id_str']}) 失败: {e}. 返回原始图片和标注进行基础转换（仅Resize, Normalize, ToTensor）。")
            # Fallback: 应用基础转换流程
            fallback_transforms = get_albumentations_transforms(is_train=False, augmentation_level="none",
                                                                img_size=CONFIG["img_size"])
            augmented = fallback_transforms(image=img_np, bboxes=bboxes_pascal, class_labels=class_indices)
            img_tensor = augmented['image']
            target_for_model["boxes"] = torch.as_tensor(augmented['bboxes'], dtype=torch.float32) if augmented[
                'bboxes'] else torch.zeros((0, 4), dtype=torch.float32)
            target_for_model["labels"] = torch.as_tensor(augmented['class_labels'], dtype=torch.int64) if augmented[
                'class_labels'] else torch.zeros(0, dtype=torch.int64)

        target_for_model["image_id"] = torch.tensor([idx])

        if target_for_model["boxes"].shape[0] > 0:
            valid_boxes_mask = (target_for_model["boxes"][:, 2] > target_for_model["boxes"][:, 0]) & \
                               (target_for_model["boxes"][:, 3] > target_for_model["boxes"][:, 1])
            target_for_model["boxes"] = target_for_model["boxes"][valid_boxes_mask]
            target_for_model["labels"] = target_for_model["labels"][valid_boxes_mask]

            if target_for_model["boxes"].shape[0] > 0:  # 再次检查过滤后是否还有框
                # 计算边界框面积
                area = (target_for_model["boxes"][:, 3] - target_for_model["boxes"][:, 1]) * \
                       (target_for_model["boxes"][:, 2] - target_for_model["boxes"][:, 0])
                target_for_model["area"] = area
                target_for_model["iscrowd"] = torch.zeros((target_for_model["boxes"].shape[0],), dtype=torch.int64)
            else:  # 如果过滤后没有框
                target_for_model["area"] = torch.zeros(0, dtype=torch.float32)
                target_for_model["iscrowd"] = torch.zeros(0, dtype=torch.int64)
                target_for_model["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
                target_for_model["labels"] = torch.zeros(0, dtype=torch.int64)
        else:  # 如果一开始就没有bbox
            target_for_model["area"] = torch.zeros(0, dtype=torch.float32)
            target_for_model["iscrowd"] = torch.zeros(0, dtype=torch.int64)
            # 确保空bbox和label
            if "boxes" not in target_for_model or target_for_model["boxes"].shape[0] == 0:
                target_for_model["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
                target_for_model["labels"] = torch.zeros(0, dtype=torch.int64)

        return img_tensor, target_for_model

    def __len__(self):
        """返回数据集中样本的总数。"""
        return len(self.image_file_paths)


def collate_batch_fn(batch):
    """将image和target分别聚合到两个列表中。
    """
    return tuple(zip(*batch))


# --- 2. 模型定义函数 ---
def get_faster_rcnn_model(num_total_classes):
    """
    加载一个在COCO数据集上预训练的Faster R-CNN模型 (ResNet-50 FPN backbone)，
    并替换其分类头为我们自定义的类别数量。
    """
    # 加载预训练模型
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # 获取原始模型中RoI Head的box_predictor的输入特征数量
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # 用一个新的FastRCNNPredictor替换掉预训练的头部 (分类器和边界框回归器)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_total_classes)

    return model


# --- 3. 训练、评估和可视化相关函数 ---
def train_single_epoch(model, optimizer, data_loader, current_device, current_epoch, num_total_epochs,
                       print_frequency=20):

    model.train()  # 设置模型为训练模式 (启用Dropout, BatchNorm更新等)
    epoch_start_time = time.time()  # 记录epoch开始时间

    lr_scheduler_warmup = None  # 初始化学习率预热调度器
    if current_epoch == 1:  # 只在第一个epoch进行预热
        warmup_factor = 1.0 / 1000  # 预热起始学习率因子
        # 预热迭代次数，不超过1000次或一个epoch的总批次数减1
        warmup_iterations = min(1000, len(data_loader) - 1)
        if warmup_iterations > 0:  # 确保有足够的迭代次数进行预热
            lr_scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=warmup_factor, total_iters=warmup_iterations
            )

    # 使用tqdm包装data_loader以显示进度条
    progress_bar = tqdm(data_loader, desc=f"Epoch [{current_epoch}/{num_total_epochs}] 训练中", leave=True,
                        unit="batch")
    total_loss_epoch, num_batches_processed = 0.0, 0  # 初始化epoch总损失和已处理批次数

    # 遍历数据加载器中的每个批次
    for batch_idx, (images_batch, targets_batch) in enumerate(progress_bar):
        images_batch = list(image.to(current_device) for image in images_batch)
        targets_batch = [{k: v.to(current_device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in
                         targets_batch]

        # 过滤掉那些在数据增强后可能没有有效边界框的样本
        valid_targets_batch = []
        valid_images_batch = []
        for i_target, t in enumerate(targets_batch):
            if 'boxes' in t and t['boxes'].shape[0] > 0:  # 只保留target中有bbox的样本
                valid_targets_batch.append(t)
                valid_images_batch.append(images_batch[i_target])

        images_batch = valid_images_batch
        targets_batch = valid_targets_batch

        try:
            # 模型前向传播，在训练模式下返回一个包含各种损失的字典
            loss_dict_from_model = model(images_batch, targets_batch)

            # 计算总损失
            total_loss_for_batch = sum(loss for loss in loss_dict_from_model.values())
            current_total_loss_value = total_loss_for_batch.item()  # 获取Python标量值

            # 检查损失是否为无效值
            if not np.isfinite(current_total_loss_value):
                print(f"错误: 损失值为 {current_total_loss_value}, 跳过此批次梯度更新.")
                continue  # 跳过这个批次的梯度更新

            optimizer.zero_grad()  # 清空之前的梯度
            total_loss_for_batch.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新模型参数

            # 如果使用了学习率预热调度器，则更新
            if lr_scheduler_warmup is not None:
                lr_scheduler_warmup.step()

            total_loss_epoch += current_total_loss_value  # 累加批次损失到epoch总损失
            num_batches_processed += 1  # 成功处理的批次数加1

            # 计算并更新进度条后缀显示的平均损失
            current_avg_loss = total_loss_epoch / num_batches_processed if num_batches_processed > 0 else 0
            progress_bar.set_postfix_str(
                f"批次总损失: {current_total_loss_value:.4f}, Epoch均损失: {current_avg_loss:.4f}")

        except RuntimeError as e:
            print(f"\n训练中捕获到运行时错误 (批次 {batch_idx}): {e}\n跳过此批次...")
            continue

    progress_bar.close()
    epoch_end_time = time.time()  # 记录epoch结束时间
    epoch_duration = epoch_end_time - epoch_start_time
    avg_loss_final = total_loss_epoch / num_batches_processed if num_batches_processed > 0 else float('nan')
    print(
        f"Epoch [{current_epoch}/{num_total_epochs}] 训练完成. 平均损失: {avg_loss_final:.4f}. 耗时: {epoch_duration:.2f} 秒 ({epoch_duration / 60:.2f} 分钟)")


def evaluate_and_get_map(model_to_eval, data_loader_val, current_device, class_names_list):
    """
    在验证集上评估模型并计算mAP
    """
    model_to_eval.eval()  # 设置模型为评估模式

    map_metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True).to(current_device)

    print("\n开始在验证集上评估 (计算mAP)...")
    with torch.no_grad():  # 评估时不需要计算梯度
        for images_batch, targets_batch in tqdm(data_loader_val, desc="评估中", leave=False):
            images_batch = list(img.to(current_device) for img in images_batch)

            processed_targets_for_metric = []
            for t_dict in targets_batch:
                processed_targets_for_metric.append({
                    "boxes": t_dict["boxes"].to(current_device),
                    "labels": t_dict["labels"].to(current_device)
                })

            try:
                predictions_from_model = model_to_eval(images_batch)  # 获取模型预测结果

                # 更新mAP度量
                map_metric.update(predictions_from_model, processed_targets_for_metric)
            except Exception as e:
                print(f"评估中更新指标时发生错误: {e}")
                continue

    try:
        map_results = map_metric.compute()  # 计算最终的mAP等结果
        print("\n评估结果 (mAP):")
        print(f"  mAP (IoU=0.50:0.95): {map_results['map'].item():.4f}")
        print(f"  mAP_50 (IoU=0.50):   {map_results['map_50'].item():.4f}")
        print(f"  mAP_75 (IoU=0.75):   {map_results['map_75'].item():.4f}")

        # 打印每个类别的AP
        if 'classes' in map_results and 'map_per_class' in map_results and \
                map_results['map_per_class'] is not None and map_results['classes'] is not None:
            print("  各类别 AP (IoU=0.50:0.95):")

            original_labels_detected = map_results['classes'].cpu().tolist()
            ap_values_for_detected_labels = map_results['map_per_class'].cpu().tolist()

            for i, label_value in enumerate(original_labels_detected):
                if 0 < label_value < len(class_names_list):
                    class_name = class_names_list[label_value]
                    print(f"    {class_name:<15}: {ap_values_for_detected_labels[i]:.4f}")
                else:
                    print(
                        f"    警告：检测到标签 {label_value} 的AP，但无法映射到已知类别名或为背景类: {ap_values_for_detected_labels[i]:.4f}")
        return map_results
    except Exception as e:
        print(f"计算或打印mAP时发生错误: {e}")
        return None


def visualize_and_save_predictions(model_to_eval, dataset_to_visualize, current_device, num_samples_to_show=3,
                                   score_display_threshold=0.5, epoch_identifier="final"):
    """
    在给定的数据集样本上可视化模型的预测结果，并与真实标签对比。
    图像显示时会进行反归一化。
    """
    model_to_eval.eval()  # 设置为评估模式
    if len(dataset_to_visualize) == 0:
        print("验证数据集为空，跳过可视化步骤.")
        return

    predictions_output_dir = "prediction_images"
    os.makedirs(predictions_output_dir, exist_ok=True)  # 创建保存图片的文件夹

    actual_num_samples = min(num_samples_to_show, len(dataset_to_visualize))
    if actual_num_samples == 0:
        print("没有样本可供可视化.")
        return

    sample_indices = random.sample(range(len(dataset_to_visualize)), actual_num_samples)  # 随机抽样

    # 定义反归一化的均值和标准差
    mean_for_unnorm = torch.tensor([0.485, 0.456, 0.406], device=current_device)
    std_for_unnorm = torch.tensor([0.229, 0.224, 0.225], device=current_device)

    for i, sample_idx in enumerate(sample_indices):
        # 从数据集中获取已经过预处理的图像Tensor和对应的target字典
        processed_image_tensor, ground_truth_target = dataset_to_visualize[sample_idx]
        image_tensor_for_model_input = processed_image_tensor.to(current_device)

        with torch.no_grad():
            prediction_output = model_to_eval([image_tensor_for_model_input])[0]

        img_to_display_tensor_gpu = image_tensor_for_model_input.clone()  # 克隆一份用于显示

        # 反归一化 unnormalized = normalized * std + mean
        unnormalized_img_tensor = img_to_display_tensor_gpu * std_for_unnorm.view(3, 1, 1) + mean_for_unnorm.view(3, 1,
                                                                                                                  1)
        # 将像素值裁剪到[0,1]范围，然后转换为PIL Image
        image_for_display = T.ToPILImage()(unnormalized_img_tensor.cpu().clamp(0, 1))

        fig, ax = plt.subplots(1, figsize=(12, 9));
        ax.imshow(image_for_display)

        # 绘制真实边界框
        if ground_truth_target and "boxes" in ground_truth_target and ground_truth_target["boxes"].numel() > 0:
            for box_coord, label_idx in zip(ground_truth_target['boxes'], ground_truth_target['labels']):
                xmin, ymin, xmax, ymax = box_coord.cpu().numpy()
                ax.add_patch(patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='lime',
                                               facecolor='none'))
                plt.text(xmin, ymin - 10, f"真: {dataset_to_visualize.selected_class_names[label_idx.item()]}",
                         color='green', fontsize=10, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0))

        # 绘制模型预测的边界框
        predicted_scores = prediction_output['scores'].cpu().numpy()
        predicted_boxes_coords = prediction_output['boxes'].cpu().numpy()
        predicted_class_labels_idx = prediction_output['labels'].cpu().numpy()

        for box_idx, score_value in enumerate(predicted_scores):
            if score_value > score_display_threshold:  # 只显示置信度高于阈值的预测
                box_coord = predicted_boxes_coords[box_idx]
                predicted_label_index = predicted_class_labels_idx[box_idx]
                predicted_label_name = dataset_to_visualize.selected_class_names[
                    predicted_label_index] if predicted_label_index < len(
                    dataset_to_visualize.selected_class_names) else f"未知({predicted_label_index})"
                xmin, ymin, xmax, ymax = box_coord
                ax.add_patch(patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='red',
                                               facecolor='none'))
                plt.text(xmin, ymin - 5, f"{predicted_label_name}: {score_value:.2f}",
                         color='red', fontsize=10, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0))

        plt.title(f"预测 vs 真实 (Epoch {epoch_identifier}) - 样本 {i + 1}");
        ax.axis('off')
        image_save_path = os.path.join(predictions_output_dir,
                                       f"prediction_epoch_{epoch_identifier}_sample_{i + 1}.png")
        plt.savefig(image_save_path);
        print(f"已保存预测图片到: {image_save_path}");
        plt.close(fig)


# --- 主执行逻辑 ---
if __name__ == "__main__":
    print(f"当前配置: {CONFIG}")
    print(f"当前使用的设备: {DEVICE}");
    print(f"选定的检测类别: {SELECTED_CLASSES}")
    if not os.path.exists(PASCAL_VOC_ROOT):
        print(f"错误: PASCAL VOC 数据集根目录 '{PASCAL_VOC_ROOT}' 未找到.");
        exit()

    # 使用CONFIG字典中的参数创建数据集和数据加载器
    dataset_train = VOCDataset(PASCAL_VOC_ROOT, "trainval.txt", SELECTED_CLASSES, True,
                               CONFIG["data_augmentation_level"], CONFIG["img_size"], CONFIG["max_train_images"])
    dataset_val = VOCDataset(PASCAL_VOC_ROOT, "test.txt", SELECTED_CLASSES, False,
                             "none", CONFIG["img_size"], CONFIG["max_val_images"])

    if len(dataset_train) == 0:
        print("错误: 训练数据集为空. 请检查路径、选定类别或max_images设置.");
        exit()

    data_loader_train = torch.utils.data.DataLoader(dataset_train, CONFIG["batch_size"], shuffle=True,
                                                    num_workers=2, collate_fn=collate_batch_fn)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False,
                                                  num_workers=2, collate_fn=collate_batch_fn)

    # 初始化模型
    model = get_faster_rcnn_model(NUM_CLASSES).to(DEVICE)

    # 定义优化器
    parameters_to_optimize = [p for p in model.parameters() if p.requires_grad]
    if CONFIG["optimizer_type"].lower() == "adam":
        optimizer = torch.optim.Adam(parameters_to_optimize, lr=CONFIG["learning_rate"], weight_decay=0.0001)
    else:
        optimizer = torch.optim.SGD(parameters_to_optimize, lr=CONFIG["learning_rate"],
                                    momentum=0.9, weight_decay=0.0005)

    # 学习率调度器
    learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                              step_size=CONFIG["scheduler_step_size"],
                                                              gamma=CONFIG["scheduler_gamma"])

    print("\n开始训练...")
    best_map_overall = 0.0  # 用于跟踪训练过程中最佳的mAP值 (map@0.50:0.95)

    # 训练循环
    for epoch_count in range(1, CONFIG["num_epochs"] + 1):
        train_single_epoch(model, optimizer, data_loader_train, DEVICE, epoch_count, CONFIG["num_epochs"])

        # 更新学习率
        if learning_rate_scheduler:
            learning_rate_scheduler.step()

        current_map_results = None
        if len(dataset_val) > 0:  
            # 在每个epoch结束后，进行mAP评估
            current_map_results = evaluate_and_get_map(model, data_loader_val, DEVICE, SELECTED_CLASSES)

            # 如果mAP优于之前的最佳mAP，保存
            if current_map_results and 'map' in current_map_results and isinstance(current_map_results['map'],
                                                                                   torch.Tensor):
                current_map_value = current_map_results['map'].item()  # 获取mAP的标量值
                if current_map_value > best_map_overall:
                    best_map_overall = current_map_value
                    save_path = f"fasterrcnn_best_map_e{epoch_count}_map{best_map_overall:.4f}.pth"
                    torch.save(model.state_dict(), save_path)
                    print(f"*** 新的最佳 mAP: {best_map_overall:.4f}，模型已保存为 {save_path} ***")

            print(f"\n--- Epoch {epoch_count} 结束, 在验证集上进行可视化 ---")
            visualize_and_save_predictions(model, dataset_val, DEVICE, 3, epoch_identifier=str(epoch_count))
        else:
            print(f"Epoch {epoch_count} 已完成. 验证集为空, 跳过评估和可视化.")

    print("\n训练完成!")
    # 评估和可视化
    if len(dataset_val) > 0:
        print("\n--- 训练结束, 在验证集上进行最终评估和可视化 ---")
        final_map_results = evaluate_and_get_map(model, data_loader_val, DEVICE, SELECTED_CLASSES)
        visualize_and_save_predictions(model, dataset_val, DEVICE, 5, epoch_identifier="final_trained")

    # 保存最后一个epoch的模型权重
    model_weights_save_path = f"fasterrcnn_final_e{CONFIG['num_epochs']}.pth"
    torch.save(model.state_dict(), model_weights_save_path)
    print(f"\n最终模型权重已保存到: {model_weights_save_path}")

    if best_map_overall > 0:  # 确保 best_map_overall 被赋过有效值
        print(f"训练过程中的最佳 mAP (IoU=0.50:0.95) 为: {best_map_overall:.4f}")