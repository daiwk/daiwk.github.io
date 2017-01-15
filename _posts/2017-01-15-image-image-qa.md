---
layout: post
category: "image"
title: "安装PIL"
tags: [paddlepaddle, 集群]
---

img-qa:

[http://idl.baidu.com/FM-IQA.html](http://idl.baidu.com/FM-IQA.html)

## 下载MSCOCO数据

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

## 下载图片－q/a的标注数据集

### 英文qa:

下载地址为[http://pan.baidu.com/s/1qXh68w8](http://pan.baidu.com/s/1qXh68w8)

数据schema:

### 中文qa:

下载地址为[http://pan.baidu.com/s/1qXh68w8](http://pan.baidu.com/s/1qXh68w8)
数据schema: