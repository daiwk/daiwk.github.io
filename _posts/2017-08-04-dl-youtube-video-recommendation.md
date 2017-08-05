---
layout: post
category: "dl"
title: "youtube视频推荐系统"
tags: [youtube视频推荐系统, ]
---

参考[http://www.sohu.com/a/155797861_465975](http://www.sohu.com/a/155797861_465975)

[Deep neural networks for youtube recommendations](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf)

YouTube是世界上最大的视频上传、分享和发现网站，YouTube推荐系统为超过10亿用户从不断增长的视频库中推荐个性化的内容。整个系统由两个神经网络组成：**候选生成网络**和**排序网络**。候选生成网络从**百万量级**的视频库中生成**上百个**候选，排序网络对候选进行打分排序，输出**排名最高的数十个结果**。



