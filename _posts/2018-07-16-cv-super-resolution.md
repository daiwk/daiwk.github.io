---
layout: post
category: "cv"
title: "超分辨率"
tags: [超分辨率, super resolution]
---

目录

<!-- TOC -->

- [SRCNN](#srcnn)
- [FSRCNN](#fsrcnn)
- [ESPCN](#espcn)
- [VDSR](#vdsr)
- [DRCN](#drcn)
- [RED](#red)
- [DRRN](#drrn)
- [LapSRN](#lapsrn)
- [SRDenseNet](#srdensenet)
- [SRGAN(SRResNet)](#srgansrresnet)
- [EDSR](#edsr)
- [deblurGAN](#deblurgan)
- [讨论](#%e8%ae%a8%e8%ae%ba)
- [优酷数据集](#%e4%bc%98%e9%85%b7%e6%95%b0%e6%8d%ae%e9%9b%86)

<!-- /TOC -->

综述：[从网络设计到实际应用，深度学习图像超分辨率综述](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650757585&idx=3&sn=10b7f6aa96e5a7717eaa079f05696935&chksm=871a9dafb06d14b9a99edc0070caac73d22a5febed2045ac103419903909cff833a38089e577&mpshare=1&scene=1&srcid=&pass_ticket=4hmIYvO6GJcf2XjDjrkA6v22Y3ZUCDsA30spOD3nAyih4OfDpXcZPiTcotPvF%2FnT#rd)

参考[https://blog.csdn.net/sinat_39372048/article/details/81628945](https://blog.csdn.net/sinat_39372048/article/details/81628945)

可以自己搞

1. 找图，然后加运动模糊（随机添加运动方向和位移）
2. 视频压缩所导致的模糊，自己压缩构造样本（边缘块状）

图片物理变化：用pil直接调锐度、亮度、对比度（有参照的，在原图基础上）之类的，其中亮度、锐度可以有个整体平均灰度值，可以往这个均值附近靠


## SRCNN

## FSRCNN

## ESPCN

## VDSR

## DRCN

## RED

## DRRN

## LapSRN

## SRDenseNet

## SRGAN(SRResNet)

## EDSR

## deblurGAN

## 讨论

最新的能精确控制某个神经元对应哪一部分（比如眼睛 鼻子之类的）

视频：用gan时，细节并没有损失，切帧，每一帧去做传统超分，可以。但做清晰度重建不行，因为运动的图片模糊部分比如帧1补了个脚。。可能帧2会补个手。。

传统超分可以先超分成大图，再下采样变小。可以省带宽，传输时下采样变小。

分辨率重建，是本身图片大小不变，直接改图。

还有另一个相似的课题：图像补全(image inpainting)

[https://blog.csdn.net/gavinmiaoc/article/details/80802967](https://blog.csdn.net/gavinmiaoc/article/details/80802967)

## 优酷数据集

[优酷发布最大工业级超高清视频数据集，超分辨率算法大赛落幕](https://mp.weixin.qq.com/s/tsJOP1bHiFV1ZI_jYRE92g)

[https://tianchi.aliyun.com/dataset/dataDetail?datald=39568](https://tianchi.aliyun.com/dataset/dataDetail?datald=39568)
