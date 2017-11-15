---
layout: post
category: "platform"
title: "openai-universe"
tags: [openai, universe]
---

目录

<!-- TOC -->


<!-- /TOC -->

官网：[https://universe.openai.com/](https://universe.openai.com/)

博客：[https://openai.com/blog/universe/](https://openai.com/blog/universe/)

2016 年的最后一个月，OpenAI 在 NIPS 2016 来临之际发布 Universe，一个在世界范围内的游戏、网站及其他应用上衡量和训练 AI 通用智能的平台。

Universe 可以让一个 AI agent 像人一样来使用电脑：观看屏幕像素并操作一个虚拟键盘和鼠标。我们必须在期望其解决的广泛任务上训练出 AI 系统，而 Universe 就可以让单个 agent 在任何人类可以通过计算机完成的任务上进行训练。

4 月份，我们启动了 Gym，这是用来开发和比较强化学习算法的工具。而借助 Universe，任何程序都可以被转换成一个 Gym 的环境。Universe 通过自动启动程序在一个 VNC 远程桌上进行工作——所以它不需要对程序内部、源码或者 bot API 的特别访问。

今天的发布内容包括一千个环境如 Flash Game，Browser task，以及slither.io GTA V 的游戏。其中数百个已经可以直接测试强化学习算法，而几乎所有的都可以使用 universe python 库自由运行