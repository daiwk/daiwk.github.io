---
layout: post
category: "cv"
title: "deblurGAN"
tags: [deblurgan, ]
---

目录

<!-- TOC -->


<!-- /TOC -->


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

pytorch的修改方式参考：

[https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/pull/64/commits/3b6a5bb36b018ffc6dd43833c5d31af1e7a5b770](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/pull/64/commits/3b6a5bb36b018ffc6dd43833c5d31af1e7a5b770)

用```nn.UpsamplingNearest2d(scale_factor=2)```替换```nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2)```
