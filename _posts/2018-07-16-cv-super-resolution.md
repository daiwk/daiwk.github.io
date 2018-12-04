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
- [inpainting](#inpainting)
- [cyclegan](#cyclegan)

<!-- /TOC -->


参考[https://blog.csdn.net/sinat_39372048/article/details/81628945](https://blog.csdn.net/sinat_39372048/article/details/81628945)

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

用GAN使模糊图片变清晰(ECCV2018)

g: resnet+反卷积

7x7conv,3x3conv,9个resblock,再接convtranspose(反卷积)

d: 简单的cnn

loss：adversarial_loss + lambda \* content_loss 

content_loss: 生成的图片和真实图片过vgg，得到第一层的输出，算perceptual loss(本质是l2 loss)

adversarial_loss: wasserstein distance


数据集：gopro有1k数据，成对的数据：效果好，但会产生伪影和亮点。因为都是运动图片

还会生成一些棋盘图（相邻像素灰度值一个高一个低）==》因为图片size不同，所以在采样时会出现重叠，然后在重叠处==》把一个反卷积改成上采样(邻近插值)

参考[Deconvolution and Checkerboard Artifacts](https://www.jianshu.com/p/36ff39344de5)

可以自己搞

1. 找图，然后加运动模糊（随机添加运动方向和位移）
2. 视频压缩所导致的模糊，自己压缩构造样本（边缘块状）

图片物理变化：用pil直接调锐度、亮度、对比度（有参照的，在原图基础上）之类的，其中亮度、锐度可以有个整体平均灰度值，可以往这个均值附近靠


## inpainting

图像补全(image inpainting)

[https://blog.csdn.net/gavinmiaoc/article/details/80802967](https://blog.csdn.net/gavinmiaoc/article/details/80802967)

## cyclegan

模糊-》清晰

清晰-》模糊

一组到另一组的标记，不需要像素级相似（背景、姿态之类的）


输入文本--》图片：stack gan (反img caption)

能精确控制某个神经元对应哪一部分（比如眼睛 鼻子之类的）
