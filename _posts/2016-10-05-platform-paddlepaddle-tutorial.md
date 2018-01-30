---
layout: post
category: "platform"
title: "paddlepaddle快速入门教程"
tags: [paddlepaddle,]
---

目录

<!-- TOC -->

- [输出日志(Log)](#输出日志log)

<!-- /TOC -->

## 输出日志(Log)

```
TrainerInternal.cpp:160]  Batch=20 samples=2560 AvgCost=0.628761 CurrentCost=0.628761 Eval: classification_error_evaluator=0.304297  CurrentEval: classification_error_evaluator=0.304297
```
模型训练会看到这样的日志，详细的参数解释如下面表格：
<center>
<table border="2" cellspacing="0" cellpadding="6" rules="all" frame="border">

<thead>
<tr>
<th scope="col" class="left">名称</th>
<th scope="col" class="left">解释</th>
</tr>
</thead>
<tbody>
<tr>
<td class="left">Batch=20</td>
<td class="left"> 表示过了20个batch </td>
</tr>

<tr>
<td class="left">samples=2560</td>
<td class="left"> 表示过了2560个样本 </td>
</tr>

<tr>
<td class="left">AvgCost</td>
<td class="left"> 每个pass的第0个batch到当前batch所有样本的平均cost </td>
</tr>

<tr>
<td class="left">CurrentCost</td>
<td class="left"> 当前log_period个batch所有样本的平均cost </td>
</tr>

<tr>
<td class="left">Eval: classification_error_evaluator</td>
<td class="left"> 每个pass的第0个batch到当前batch所有样本的平均分类错误率 </td>
</tr>

<tr>
<td class="left">CurrentEval: classification_error_evaluator</td>
<td class="left"> 当前log_period个batch所有样本的平均分类错误率 </td>
</tr>

</tbody>
</table>
</center>
<br>
