---
layout: post
category: "audio"
title: "音频处理常用库"
tags: [LibROSA, Madmom, pyAudioAnalysis, ]
---

目录

<!-- TOC -->

- [LibROSA](#librosa)
- [Madmom](#madmom)
- [pyAudioAnalysis](#pyaudioanalysis)

<!-- /TOC -->

参考[吐血整理！绝不能错过的24个顶级Python库](https://zhuanlan.zhihu.com/p/76112940)

## LibROSA

[https://librosa.github.io/librosa/](https://librosa.github.io/librosa/)

LibROSA是一个用于音乐和音频分析的Python库。它提供了创建音乐信息检索系统所需的构建块。

安装指南传送门：[https://librosa.github.io/librosa/install.html](https://librosa.github.io/librosa/install.html)

这是一篇关于音频处理及其工作原理的深度文章：

[利用深度学习开始音频数据分析（含案例研究）](https://www.analyticsvidhya.com/blog/2017/08/audio-voice-processing-deep-learning/)

## Madmom

[https://github.com/CPJKU/madmom](https://github.com/CPJKU/madmom)

Madmom是一个用于音频数据分析的很棒的Python库。它是一个用Python编写的音频信号处理库，主要用于音乐信息检索（MIR）任务。

以下是安装Madmom的必备条件：

+ NumPy
+ SciPy
+ Cython
+ Mido

以下软件包用于测试安装：

+ PyTest
+ Fyaudio
+ PyFftw

安装Madmom的代码：

```shell
pip install madmom
```

下文可用以了解Madmom如何用于音乐信息检索：

[学习音乐信息检索的音频节拍追踪（使用Python代码）](https://www.analyticsvidhya.com/blog/2018/02/audio-beat-tracking-for-music-information-retrieval/)

## pyAudioAnalysis

[https://github.com/tyiannak/pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis)

pyAudioAnalysis是一个用于音频特征提取、分类和分段的Python库，涵盖广泛的音频分析任务，例如：

+ 对未知声音进行分类
+ 检测音频故障并排除长时间录音中的静音时段
+ 进行监督和非监督的分割
+ 提取音频缩略图等等

可以使用以下代码进行安装：

```shell
pip install pyAudioAnalysis
```
