---
layout: post
category: "cv"
title: "gan with the wind"
tags: [gan, ]
---

目录

<!-- TOC -->

- [背景](#背景)
- [参考文献](#参考文献)

<!-- /TOC -->

## 背景

参考[虚构的对抗，GAN with the wind](https://mp.weixin.qq.com/s?__biz=MzA3NTM4MzY1Mg==&mid=2650814302&idx=1&sn=87642b1c0662b3b71666e45a1cb2a0b3&chksm=8485c155b3f248433dbdb33d92d36851d25791a261c7ae09164155314e1cee7d578140ef0456&mpshare=1&scene=1&srcid=1018lhAqPZOlGOOGCzx0bqrH&pass_ticket=h3hZq0WHA7Cyui0YBndmrxji2MHJPRFf2%2F6zqKyUTOnTIhZZuESFoAbpmgeoETVa#rd)

Goodfellow 【1】于2014年提出了GAN的概念，他的解释如下：GAN的核心思想是构造两个深度神经网络：判别器D和生成器G，用户为GAN提供一些真实货币作为训练样本，生成器G生成假币来欺骗判别器D，判别器D判断一张货币是否来自真实样本还是G生成的伪币；判别器和生成器交替训练，能力在博弈中同步提高，最后达到平衡点的时候判别器无法区分样本的真伪，生成器的伪造功能炉火纯青，生成的货币几可乱真。这种阴阳互补，相克相生的设计理念为GAN的学说增添了魅力。

GAN模型的优点来自于自主生成数据。机器学习的关键在于海量的数据样本，GAN自身可以生成不尽的样本，从而极大地减少了对于训练数据的需求，因此极其适合无监督学习；GAN的另外一个优点是对于所学习的概率分布要求非常宽泛，这一概率分布由训练数据的测度分布来表达，不需要有显式的数学表示。

GAN虽然在工程实践中取得巨大的成功，但是缺乏严格的理论基础。大量的几何假设，都是仰仗似是而非的观念；其运作的内在机理，也是依据肤浅唯像的经验解释。丘成桐先生率领团队在学习算法的基础理论方面进行着不懈的探索。我们用最优传输（Optimal mass Transportation）理论框架来阐释对抗生成模型，同时用凸几何（Convex Geometry）的基础理论来为最优传输理论建立几何图景。通过理论验证，我们发现关于对抗生成模型的一些基本观念有待商榷：理论上，Wasserstein GAN中生成器和识别器的竞争是没有必要的，生成器网络和识别器网络的交替训练是徒劳的，此消彼长的对抗是虚构的。最优的识别器训练出来之后，生成器可以由简单的数学公式所直接得到。详细的数学推导和实验结果可以在【7】中找到。

## 参考文献

+ Goodfellow, Ian J.; Pouget-Abadie, Jean; Mirza, Mehdi; Xu, Bing; Warde-Farley, David; Ozair, Sherjil; Courville, Aaron; Bengio, Yoshua (2014). "Generative Adversarial Networks". arXiv:1406.2661 
+ A. D. Alexandrov. “Convex polyhedra” Translated from the 1950 Russian edition by N. S. Dairbekov, S. S. Kutateladze and A. B. Sossinsky. Springer Monographs in Mathematics. Springer-Verlag, Berlin, 2005.
+ Martin Arjovsky, Soumith Chintala, and Léon Bottou. Wasserstein generative adversarial networks. In International Conference on Machine Learning, pages 214–223, 2017.
+ Yann Brenier. Polar factorization and monotone rearrangement of vector-valued functions. Comm. Pure Appl. Math., 44(4):375–417, 1991.
+ Xianfeng Gu, Feng Luo, Jian Sun, and Tianqi Wu. A discrete uniformization theorem for polyhedral surfaces. Journal of Differential Geometry (JDG), 2017.
+ Xianfeng Gu, Feng Luo, Jian Sun, and Shing-Tung Yau. Variational principles for minkowski type problems, discrete optimal transport, and discrete monge-ampere equations. Asian Journal of Mathematics (AJM), 20(2):383 C 398, 2016.
+ Na Lei,Kehua Su,Li Cui,Shing-Tung Yau,David Xianfeng Gu, A Geometric View of Optimal Transportation and Generative Model, arXiv:1710.05488.


