---
layout: post
category: "cv"
title: "image-qa"
tags: [mqa, image-qa, 图片问答]
---

项目主页：
[http://idl.baidu.com/FM-IQA.html](http://idl.baidu.com/FM-IQA.html)

## 1. 摘要

```
In this paper, we present the mQA model, which is able to answer questions about the content of an image. The answer can be a sentence, a phrase or a single word.
Our model contains four components: a Long Short-Term Memory (LSTM) to extract the question representation, a Convolutional Neural Network (CNN) to extract the visual representation, an LSTM for storing the linguistic context in an answer, and a fusing component to combine the information from the first three
components and generate the answer.
```

## 2. 数据准备

### 2.1 下载MSCOCO数据

```
#!/bin/bash

function download_mscoco()

{
mkdir data
cd data

# http://mscoco.org/dataset/#download
#2014 Training images [80K/13GB]
#2014 Val. images [40K/6.2GB]
#2014 Testing images [40K/6.2GB]
#2015 Testing images [80K/12.4G]
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip
wget http://msvocds.blob.core.windows.net/coco2014/test2014.zip
wget http://msvocds.blob.core.windows.net/coco2015/test2015.zip


#annotations
#2014 Train/Val object instances [158MB]
#2014 Train/Val person keypoints [70MB]
#2014 Train/Val image captions [18.8MB]
#2014 Testing Image info [0.74MB]
#2015 Testing Image info [1.83MB]
wget http://msvocds.blob.core.windows.net/annotations-1-0-3/instances_train-val2014.zip
wget http://msvocds.blob.core.windows.net/annotations-1-0-3/person_keypoints_trainval2014.zip
wget http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip
wget http://msvocds.blob.core.windows.net/annotations-1-0-4/image_info_test2014.zip
wget http://msvocds.blob.core.windows.net/annotations-1-0-4/image_info_test2015.zip

}

function unzip_data()
{
cd data
#2014 Training images [80K/13GB]
#2014 Val. images [40K/6.2GB]
#2014 Testing images [40K/6.2GB]
#2015 Testing images [80K/12.4G]
unzip train2014.zip
unzip val2014.zip
unzip test2014.zip
unzip test2015.zip


#annotations
#2014 Train/Val object instances [158MB]
#2014 Train/Val person keypoints [70MB]
#2014 Train/Val image captions [18.8MB]
#2014 Testing Image info [0.74MB]
#2015 Testing Image info [1.83MB]
unzip instances_train-val2014.zip
unzip person_keypoints_trainval2014.zip
unzip captions_train-val2014.zip
unzip image_info_test2014.zip
unzip image_info_test2015.zip



}

download_mscoco
unzip_data
```

### 2.2 下载图片－q/a的标注数据集

#### 2.2.1 英文qa:

下载地址为[http://pan.baidu.com/s/1qXh68w8](http://pan.baidu.com/s/1qXh68w8)

数据schema:

### 2.2.2 中文qa:

下载地址为[http://pan.baidu.com/s/1qXh68w8](http://pan.baidu.com/s/1qXh68w8)
数据schema:

## 3. 模型介绍

![](../assets/image-qa/model-intro.jpg)

### 3.1 LSTM(Q)

### 3.1 CNN

### 3.1 LSTM(A)

### 3.4 fusing layer

### 3.5 intermediate and softmax

