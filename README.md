# Faster-R-CNN
本项目为一个基于Faster R-CNN迁移学习的PASCAL VOC多类别目标检测系统
本项目实现的是目标检测，用于定位图像中物体的位置并识别其类别。该技术需要同时完成物体定位和分类两个子任务。
本项目采用Faster R-CNN作为基础模型，结合迁移学习和数据增强技术。核心评估指标使用mAP，通过数据增强、模型微调和标准化评估，探索提升目标检测性能的方法。
主要挑战包括多类别检测的准确性、小物体检测效果以及训练数据的充分性。预期在PASCAL VOC数据集上实现20类物体的准确检测。

数据集可前往VOC2007数据集官网下载：http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html

部分识别效果展示：
<img width="1787" height="1795" alt="image" src="https://github.com/user-attachments/assets/9102ce69-efae-4b38-a203-af7e08c7ddbf" />
<img width="1860" height="1747" alt="image" src="https://github.com/user-attachments/assets/8f658adf-7e26-4f2e-b618-b7ca464f796c" />
