---
layout: post
category: "platform"
title: "tf的im2txt"
tags: [im2txt]
---

目录

<!-- TOC -->

- [0. 必需的包](#0-必需的包)
- [1. 数据集准备](#1-数据集准备)
- [2. 下载Inception v3 Checkpoint](#2-下载inception-v3-checkpoint)
- [3. 训练](#3-训练)

<!-- /TOC -->

参考：[https://github.com/tensorflow/models/tree/master/im2txt](https://github.com/tensorflow/models/tree/master/im2txt)

论文：
"Show and Tell: Lessons learned from the 2015 MSCOCO Image Captioning Challenge.", Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan., IEEE transactions on pattern analysis and machine intelligence (2016).
[http://arxiv.org/abs/1609.06647](http://arxiv.org/abs/1609.06647)


## 0. 必需的包

+ bazel([官网](https://bazel.build/versions/master/docs/install.html))
+ tf
+ numpy
+ nltk(下载数据可以```python -m nltk.downloader -d /home/data/docker_share/nltk_data all```指定存放目录[约12G])

## 1. 数据集准备

输入数据格式是“native TFRecord format”：

```
The TFRecord format consists of a set of sharded files containing serialized tf.SequenceExample protocol buffers. Each tf.SequenceExample proto contains an image (JPEG format), a caption and metadata such as the image id.

Each caption is a list of words. During preprocessing, a dictionary is created that assigns each word in the vocabulary to an integer-valued id. Each caption is encoded as a list of integer word ids in the tf.SequenceExample protos.

```


使用mscoco数据集。

```
## Make sure there is at least 150G space available!!!!!!!!!!!!!!!

# Location to save the MSCOCO data. 

function prepare()
{
# Build the preprocessing script.
bazel build im2txt/download_and_preprocess_mscoco

# Run the preprocessing script.
bazel-bin/im2txt/download_and_preprocess_mscoco "${MSCOCO_DIR}"

return $?

}
```

当最后一句话是```Finished processing all 20267 image-caption pairs in data set 'test'.```时，就成功了。

最终数据：

+ 256个训练文件：train-?????-of-00256
+ 4个验证文件：val-?????-of-00004
+ 8个测试文件：test-?????-of-00008

## 2. 下载Inception v3 Checkpoint

使用inception v3来初始化img部分的权重。tf专门搞了个**slim**来存这些预训练好的模型（[https://github.com/tensorflow/models/tree/master/slim#tensorflow-slim-image-classification-library](https://github.com/tensorflow/models/tree/master/slim#tensorflow-slim-image-classification-library)）。

```
function get_inception()
{
# Location to save the Inception v3 checkpoint.
export INCEPTION_DIR="${HOME}/im2txt/data"
mkdir -p ${INCEPTION_DIR}

wget "http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz"
tar -xvf "inception_v3_2016_08_28.tar.gz" -C ${INCEPTION_DIR}
rm "inception_v3_2016_08_28.tar.gz"
}
```

注意：这里的inception v3只用于**第一步**的模型初始化，后面整个模型的训练过程中，会有新的checkpoint，这个inception v3就没啥用了。

## 3. 训练

```
function train() 
{

# Inception v3 checkpoint file.
INCEPTION_CHECKPOINT="${HOME}/im2txt/data/inception_v3.ckpt"

# Directory to save the model.
MODEL_DIR="${HOME}/im2txt/model"

# Build the model.
bazel build -c opt im2txt/...

# Run the training script.
bazel-bin/im2txt/train \
  --input_file_pattern="${MSCOCO_DIR}/train-?????-of-00256" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/train" \
  --train_inception=false \
  --number_of_steps=1000000

return $?
}
```