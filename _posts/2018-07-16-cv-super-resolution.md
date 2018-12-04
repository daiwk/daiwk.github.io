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
- [讨论](#%E8%AE%A8%E8%AE%BA)

<!-- /TOC -->


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
