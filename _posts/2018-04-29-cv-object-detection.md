---
layout: post
category: "cv"
title: "目标检测汇总"
tags: [目标检测, rcnn, ssd, yolo, retainnet, fast rcnn, faster rcnn, fpn, yolo nano, ]
---

目录

<!-- TOC -->

- [历史回顾](#%e5%8e%86%e5%8f%b2%e5%9b%9e%e9%a1%be)
- [开源库：Detectron](#%e5%bc%80%e6%ba%90%e5%ba%93detectron)
- [YOLO nano](#yolo-nano)

<!-- /TOC -->

## 历史回顾

[从RCNN到SSD，这应该是最全的一份目标检测算法盘点](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650741534&idx=1&sn=02dc164ffcedbf22124b97841ba67fe5&chksm=871adf60b06d567690fa2328b161c012a464687768e50f812a51b5533a7d68b99af1cf8f02b8&scene=0&pass_ticket=INCrGaryVZRn7Xp0qFQ7uod1VN14o8mkpvq1bswtroEgKQavvDm7mmg4E7yTOH6d#rd)


同时参考[一文读懂目标检测：R-CNN、Fast R-CNN、Faster R-CNN、YOLO、SSD](https://mp.weixin.qq.com/s?__biz=MzU1NTUxNTM0Mg==&mid=2247489070&idx=1&sn=d7b00a6e66de9191d898ee654be448cc&chksm=fbd27a8fcca5f39950a9ef8d423afcc39f4ad5ba01f84bedca905975b53f34817f660fca1795&mpshare=1&scene=1&srcid=0723zzseCRdY3SSK8GKzxTvO&pass_ticket=dxpPUDHz41aBj1bP227WEg1oWcCDfep3IeGSCzYrlaZP9ZqENPugpUQrWsVELUK8#rd)

1. 传统的目标检测算法：Cascade + HOG/DPM + Haar/SVM以及上述方法的诸多改进、优化；
2. 候选区域/框 + 深度学习分类：通过提取候选区域，并对相应区域进行以深度学习方法为主的分类的方案，如：
+ R-CNN（Selective Search + CNN + SVM）
+ SPP-net（ROI Pooling）
+ Fast R-CNN（Selective Search + CNN + ROI）
+ Faster R-CNN（RPN + CNN + ROI）
+ R-FCN等系列方法；
3. 基于深度学习的回归方法：YOLO/SSD/DenseBox 等方法；以及最近出现的结合RNN算法的RRC detection；结合DPM的Deformable CNN等

<html>
<br/>
<img src='../assets/rcnns.webp' style='max-height: 350px'/>
<br/>
</html>

## 开源库：Detectron

[https://github.com/facebookresearch/Detectron](https://github.com/facebookresearch/Detectron)

如果你正在寻找最先进的物体检测算法，那么你可以使用Detectron。

它由Facebook开发，是AI Research软件系统的一部分。它利用Caffe2深度学习框架和Python。

## YOLO nano

[比Tiny YOLOv3小8倍，性能提升11个点，4MB的网络也能做目标检测](https://mp.weixin.qq.com/s/cl3HBzt_u1Edp35YLmBVsw)

[YOLO Nano: a Highly Compact You Only Look Once Convolutional Neural Network for Object Detection](https://arxiv.org/abs/1910.01271)

研究者提出了名为 YOLO Nano 的网络。这一模型的大小在 4.0MB 左右，比 Tiny YOLOv2 和 Tiny YOLOv3 分别小了 15.1 倍和 8.3 倍。在计算上需要 4.57B 次推断运算，比后两个网络分别少了 34% 和 17%。

在性能表现上，在 VOC2007 数据集取得了 69.1% 的 mAP，准确率比后两者分别提升了 12 个点和 10.7 个点。研究者还在 Jetson AGX Xavier 嵌入式模块上，用不同的能源预算进行了测试，进一步说明 YOLO Nano 非常适合边缘设备与移动端。
