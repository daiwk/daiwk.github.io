---
layout: post
category: "rl"
title: "alphago-zero"
tags: [alphago zero, ]
---

目录

<!-- TOC -->

- [背景](#背景)
- [论文地址](#论文地址)

<!-- /TOC -->

参考
[【21天完虐Master】AlphaGo Zero横空出世，DeepMind Nature论文解密不使用人类知识掌握围棋](https://mp.weixin.qq.com/s?__biz=MzI3MTA0MTk1MA==&mid=2652006424&idx=1&sn=d0be3ac450e735cbd9946de4f155fc42&chksm=f1211ce9c65695ff159a3f44d31abc92ad9494bef122082a09089d3ffeaa9e74678db1256392&mpshare=1&scene=1&srcid=10192t2Uq9DfDmj8F6gQ652a&pass_ticket=h3hZq0WHA7Cyui0YBndmrxji2MHJPRFf2%2F6zqKyUTOnTIhZZuESFoAbpmgeoETVa#rd)

[今日Nature: 人工智能从0到1, 无师自通完爆阿法狗100-0 \| 深度解析](https://mp.weixin.qq.com/s?__biz=MzA4NDQwNDQ2Nw==&mid=2650480675&idx=1&sn=9cfb29cb37fa28892cae59775ba6b816&chksm=87e831bfb09fb8a92f24ad7516e51ca215cefa627d3fa0d4a82b334418a8311dd37ee02e9e29&mpshare=1&scene=1&srcid=1019QrErNgHVFjp0D7xARMyj&pass_ticket=h3hZq0WHA7Cyui0YBndmrxji2MHJPRFf2%2F6zqKyUTOnTIhZZuESFoAbpmgeoETVa#rd)

[AlphaGo Zero 没有告诉你的秘密](https://mp.weixin.qq.com/s?__biz=MjM5ODIzNDQ3Mw==&mid=2649967631&idx=1&sn=c3ebeaa66be9a920bfdad91020da3ed1&chksm=beca3c0989bdb51f9ff1d3f89cbe05d5b0822db9fa89d912abfe6660e6c070abeabcf44b733f&mpshare=1&scene=1&srcid=1019F9cifstsNiGxC0gaOwGc&pass_ticket=h3hZq0WHA7Cyui0YBndmrxji2MHJPRFf2%2F6zqKyUTOnTIhZZuESFoAbpmgeoETVa#rd)

## 背景

人工智能长期以来的一个目标是创造一个能够在具有挑战性的领域，以超越人类的精通程度学习的算法，“tabula rasa”（译注：一种认知论观念，认为指个体在没有先天精神内容的情况下诞生，所有的知识都来自于后天的经验或感知）。此前，AlphaGo成为首个在围棋中战胜人类世界冠军的系统。AlphaGo的那些神经网络使用人类专家下棋的数据进行监督学习训练，同时也通过自我对弈进行强化学习。

在这里，我们介绍一种仅基于强化学习的算法，不使用人类的数据、指导或规则以外的领域知识。AlphaGo成了自己的老师。我们训练了一个神经网络来预测AlphaGo自己的落子选择和AlphaGo自我对弈的赢家。这种神经网络提高了树搜索的强度，使落子质量更高，自我对弈迭代更强。从“tabula rasa”开始，我们的新系统AlphaGo Zero实现了超人的表现，以100：0的成绩击败了此前发表的AlphaGo。

## 论文地址

[Mastering the Game of Go without Human Knowledge](../assets/agz_unformatted_nature.pdf)


