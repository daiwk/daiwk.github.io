---
layout: post
category: "audio"
title: "fastspeech"
tags: [fastspeech, ]
---

目录

<!-- TOC -->


<!-- /TOC -->

[FastSpeech: Fast, Robust and Controllable Text to Speech](https://arxiv.org/pdf/1905.09263.pdf)


[将文本转语音速度提高38倍，这个FastSpeech真的很fast](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650763134&idx=5&sn=47795c4f17d677fb591689db5bef6729&chksm=871aab00b06d2216b21034bfbfb6c53351a479dc2af74bfd052526c50b699d24d2420ab03945&mpshare=1&scene=1&srcid=&pass_ticket=TloMdmvUbLd5jnKvVTzrccQhGuskwL6KQ0HhJLF56Nwtcb16%2BVvMA09bw32tFrjs#rd)

基于神经网络的端到端文本语音转换（TTS）显著改善了合成语音的质量。一些主要方法（如 Tacotron 2）通常首先从文本生成梅尔频谱（mel-spectrogram），然后使用诸如 WaveNet 的声码器从梅尔频谱合成语音。

与基于连接和统计参数的传统方法相比，基于神经网络的端到端模型有一些不足之处，包括推理速度较慢，合成语音不稳健（即某些词被跳过或重复），且缺乏可控性（语音速度或韵律控制）。

本文提出了一种基于 Transformer 的新型前馈网络，用于为 TTS 并行生成梅尔频谱。具体来说就是，从基于编码器-解码器的教师模型中提取注意力对齐（attention alignments），用于做音素（phoneme）持续时间预测。长度调节器利用这一预测来扩展源音素序列，以匹配目标梅尔频谱序列的长度，从而并行生成梅尔频谱。