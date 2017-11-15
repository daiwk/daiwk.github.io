---
layout: post
category: "cv"
title: "wgan"
tags: [wgan]
---

目录


<!-- TOC -->


<!-- /TOC -->

参考：[微信公众号"机器之心"文章](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650722818&idx=3&sn=03cb67c8a8ee7f83a7448b518f4336ab&chksm=871b167cb06c9f6a018a99b79d8b2764b207be2b4d03f132151d99124edf2aff4c116a9dc98d&scene=0&pass_ticket=vjEpmxe2DG4P%2By4GjgdfVEMIt0g0SpbViafCaNrBt8viOsGkibUK9SIS47UfCM27#rd)

参考：[知乎专栏:令人拍案叫绝的Wasserstein GAN](https://zhuanlan.zhihu.com/p/25071913)

参考顾险峰老师的文章：[看穿机器学习（W-GAN模型）的黑箱](https://mp.weixin.qq.com/s?__biz=MzA3NTM4MzY1Mg==&mid=2650813024&idx=1&sn=31e326bd79ed24f5f47b35091385b9ab&chksm=8485c46bb3f24d7d36d1a93b48d9f4d0335262b1152de0bd0f2f1d09527e4acb2ae3d4730913&mpshare=1&scene=1&srcid=02121jrcKo3ax5gCkgeZ7aS6&pass_ticket=6F3WrFmalMgZ5Oj086HJDIpCgEgR3p0cvrPVa2BxK2A4pl0bhEU19uXGpI43TaYF#rd)

相关数学知识：

[最优传输理论（一）](https://mp.weixin.qq.com/s?__biz=MzA3NTM4MzY1Mg==&mid=401810859&idx=1&sn=085601ed23400b162363c724651b98cb&mpshare=1&scene=1&srcid=0212spaxFITf4n7M3jA1OV8L&pass_ticket=6F3WrFmalMgZ5Oj086HJDIpCgEgR3p0cvrPVa2BxK2A4pl0bhEU19uXGpI43TaYF#rd)

[最优传输理论（二）](https://mp.weixin.qq.com/s?__biz=MzA3NTM4MzY1Mg==&mid=401899637&idx=1&sn=ae6ddb620b9b2d0bc140dc10aaef0e39&mpshare=1&scene=1&srcid=0212pgfZXnZLASC3s00v6ALz&pass_ticket=6F3WrFmalMgZ5Oj086HJDIpCgEgR3p0cvrPVa2BxK2A4pl0bhEU19uXGpI43TaYF#rd)

[最优传输理论（三）](https://mp.weixin.qq.com/s?__biz=MzA3NTM4MzY1Mg==&mid=402015289&idx=1&sn=7c547abab1c6c33460795ebb1019d29a&mpshare=1&scene=1&srcid=0212GqLHCpF8c1uVyCjUdEht&pass_ticket=6F3WrFmalMgZ5Oj086HJDIpCgEgR3p0cvrPVa2BxK2A4pl0bhEU19uXGpI43TaYF#rd)

[最优传输理论（四）](https://mp.weixin.qq.com/s?__biz=MzA3NTM4MzY1Mg==&mid=402434159&idx=1&sn=cc0ece42454fcb464be8ecd03d97b56a&mpshare=1&scene=1&srcid=0212Cb7GHtEVM0VfqRq70P3B&pass_ticket=6F3WrFmalMgZ5Oj086HJDIpCgEgR3p0cvrPVa2BxK2A4pl0bhEU19uXGpI43TaYF#rd)

[最优传输理论应用：色彩变换算法](https://mp.weixin.qq.com/s?__biz=MzA3NTM4MzY1Mg==&mid=2650812934&idx=1&sn=1f1475529ee55c794c1dfab749700e3a&chksm=8485c40db3f24d1b89a4f2c985c54a50c4e7ce4b68b9aa60e39378e21e49881d7990caede42c&mpshare=1&scene=1&srcid=02128ANnHWrmdy8dJSOYPzwJ&pass_ticket=6F3WrFmalMgZ5Oj086HJDIpCgEgR3p0cvrPVa2BxK2A4pl0bhEU19uXGpI43TaYF#rd)




主要效果如下：

+　彻底解决GAN**训练不稳定**的问题，不再需要小心平衡生成器和判别器的训练程度
+　基本解决了**collapse mode**的问题，确保了生成样本的多样性 
+　训练过程中终于有一个像交叉熵、准确率这样的**数值来指示训练的进程**，这个**数值越小代表GAN训练得越好**，代表生成器产生的图像质量越高(如下图所示)

![](../assets/wgan-progress.jpg)

+　以上一切好处不需要精心设计的网络架构，**最简单的多层全连接网络就可以做到**



主要改进有以下四点：

+ **判别器最后一层去掉sigmoid**(原始GAN的判别器做的是true/false二分类任务，所以最后一层是sigmoid，但是现在WGAN中的判别器做的是近似拟合**Wasserstein距离，属于回归任务**，所以要把最后一层的sigmoid拿掉。)
+ **生成器和判别器的loss不取log**
+ **每次更新判别器的参数之后把它们的绝对值截断到不超过一个固定常数c**
+ **不要用基于动量的优化算法**（包括momentum和Adam），**推荐RMSProp**，SGD也行

