---
layout: post
category: "platform"
title: "tensor-to-tensor[实践篇]"
tags: [tensor-to-tensor, t2t, tensor2tensor]
---

目录

<!-- TOC -->

- [0. Suggested Datasets and Models](#0-suggested-datasets-and-models)
    - [Image Classification](#image-classification)
    - [Language Modeling](#language-modeling)
    - [Sentiment Analysis](#sentiment-analysis)
    - [Speech Recognition](#speech-recognition)
    - [Summarization](#summarization)
    - [Translation](#translation)
- [1. overview](#1-overview)
    - [Datasets](#datasets)
    - [Problems and Modalities](#problems-and-modalities)
    - [Models](#models)

<!-- /TOC -->

[One model to learn them all](https://arxiv.org/abs/1706.05137)

知乎专栏的讨论：[https://zhuanlan.zhihu.com/p/28680474](https://zhuanlan.zhihu.com/p/28680474)



[https://github.com/tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor)

## 0. Suggested Datasets and Models

建议的超参是在Cloud TPUs或者8-GPU machines上训练好的

### Image Classification

数据集方面，

+ ImageNet (a large data-set):可以使用```--problem=image_imagenet```，或者是缩小版的数据集```image_imagenet224, image_imagenet64, image_imagenet32```
+ CIFAR-10: 可以使用```--problem=image_cifar10```，或者是关闭了data augmentation的```--problem=image_cifar10_plain```
+ CIFAR-100: 可以使用```--problem=image_cifar100```
+ MNIST: 可以使用```--problem=image_mnist```

模型方面，

+ ImageNet: 建议使用ResNet或者Xception
    + ```--model=resnet --hparams_set=resnet_50```，resnet的top-1 accuracy能达到76%以上。
    + ```--model=xception --hparams_set=xception_base```
+ CIFAR and MNIST：建议使用shake-shake模型，```--model=shake_shake --hparams_set=shakeshake_big```。当```--train_steps=700000```的时候，在CIFAR-10上，可以达到97% accuracy。

### Language Modeling

数据集方面，

+ PTB (a small data-set): 
    + word-level的模型：```--problem=languagemodel_ptb10k```
    + character-level的模型：```--problem=languagemodel_ptb_characters```
+ LM1B (a billion-word corpus): 
    + subword-level的模型：```--problem=languagemodel_lm1b32k```
    + character-level的模型：```--problem=languagemodel_lm1b_characters```

模型方面，建议直接上```--model=transformer```，

+ PTB：超参设置为```--hparams_set=transformer_small```
+ LM1B：超参设置为```--hparams_set=transformer_base```

### Sentiment Analysis

IMDB数据集：```--problem=sentiment_imdb```

建议使用模型```--model=transformer_encoder```，由于这个数据集很小，使用```--hparams_set=transformer_tiny```，以及比较少的训练步数就行了```--train_steps=2000```。

### Speech Recognition

数据集：Librispeech (English speech to text)

+ 完整的数据集：```--problem=librispeech```
+ 清洗过的较小数据集：```--problem=librispeech_clean```

### Summarization

将CNN/DailyMail的文章摘要成一些句子的数据集：```--problem=summarize_cnn_dailymail32k```

模型使用```--model=transformer```，超参使用```--hparams_set=transformer_prepend```，这样可以得到不错的ROUGE scores。

### Translation

数据集：

+ English-German: ```--problem=translate_ende_wmt32k```
+ English-French: ```--problem=translate_enfr_wmt32k```
+ English-Czech: ```--problem=translate_encs_wmt32k```
+ English-Chinese: ```--problem=translate_enzh_wmt32k```
+ English-Vietnamese: ```--problem=translate_envi_iwslt32k```
 
如果要将源语言和目标语言调换，那么可以直接加一个```_rev```，也就是German-English即为```--problem=translate_ende_wmt32k_rev```

翻译问题，建议使用```--model=transformer```，在8 GPUs上训练300K steps之后，在English-German上可以达到28的BLEU。如果在单GPU上，建议使用```--hparams_set=transformer_base_single_gpu```，在大数据集上（例如English-French），想要达到很好的效果，使用大模型```--hparams_set=transformer_big```。

针对机器翻译问题的一个基本流程：

```shell
# See what problems, models, and hyperparameter sets are available.
# You can easily swap between them (and add new ones).
t2t-trainer --registry_help

# 1. 设置模型参数
PROBLEM=translate_ende_wmt32k
MODEL=transformer
HPARAMS=transformer_base_single_gpu

DATA_DIR=$HOME/t2t_data
TMP_DIR=/tmp/t2t_datagen
TRAIN_DIR=$HOME/t2t_train/$PROBLEM/$MODEL-$HPARAMS

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

# 2. 生成数据
t2t-datagen \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM

# 3. 训练
# *  If you run out of memory, add --hparams='batch_size=1024'.
t2t-trainer \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR

# 4. Decode

DECODE_FILE=$DATA_DIR/decode_this.txt
echo "Hello world" >> $DECODE_FILE
echo "Goodbye world" >> $DECODE_FILE
echo -e 'Hallo Welt\nAuf Wiedersehen Welt' > ref-translation.de

BEAM_SIZE=4
ALPHA=0.6

t2t-decoder \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
  --decode_from_file=$DECODE_FILE \
  --decode_to_file=translation.en

# 5. 查看翻译结果
cat translation.en

# 6. Evaluate the BLEU score
# Note: Report this BLEU score in papers, not the internal approx_bleu metric.
t2t-bleu --translation=translation.en --reference=ref-translation.de
```

## 1. overview

### Datasets

+ 数据集都是```tensorflow.Example```的protobuf标准化处理过的```TFRecord```文件。
+ 所有数据集通[t2t-datagen](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/bin/t2t-datagen)进行注册与生成。

### Problems and Modalities

+ Problems，指的是训练时针对task和dataset的hyperparameters，主要设置的是输入和输出的modalities（例如，symbol, image, audio, label）以及vocabularies。

所有的problem都在[problem_hparams.py](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/problem_hparams.py)中定义了，或者通过```@registry.register_problem```进行注册。

直接运行```t2t-datagen```可以查看目前支持的problems列表。

+ Modalities(形态)，将input和output的数据类型abstract away，从而使得model可以直接处理modality-independent tensors。

所有的modalities定义在[modality.py](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/modality.py)中。

### Models

```T2TModel```定义了核心的tensor-to-tensor变换，与input/output的modality或者task无关。模型的输入是dense tensors，输出是dense tensors，可以在final step中被一个modality进行变换(例如通过一直final linear transform，产出logits供softmax over classes使用)。

在models目录下的[__init__.py](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/__init__.py)中import了所有model。

这些模型的基类是[T2TModel](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/t2t_model.py)，都是通过[@registry.register_model](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/registry.py)来进行注册的。

### Hyperparameter Sets

超参数集合通过[@registry.register_hparams](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/registry.py)进行注册，并通过[tf.contrib.training.HParams](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/training/python/training/hparam.py)编码成对象。

HParams对model和problem都是可用的。

[common_hparams.py](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_hparams.py)中定义了基本的超参集，而且超参集的函数可以组成其他超参集的函数。

### Trainer

支持分布式训练，参考[https://tensorflow.github.io/tensor2tensor/distributed_training.html](https://tensorflow.github.io/tensor2tensor/distributed_training.html)，包括：

+ 多GPU
+ synchronous (1 master, many workers)
+ asynchronous (independent workers synchronizing through a parameter server)

## 2. 新增components

## 3. 新增数据集


## 4. 可视化

参考：[https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/insights](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/insights)

首先，安装nodejs的npm：[https://nodejs.org/en/](https://nodejs.org/en/)

然后用npm安装Bower：

```shell
npm install -g bower
```

然后用bower对本项目的insights部分进行安装：

```shell
pushd tensor2tensor/insights/polymer
bower install
popd
```

还需要

```shell
pip install oauth2client
```

然后写一个json，表示模型的各种配置：

```shell
  {
    "configuration": [{
      "source_language": "en",
      "target_language": "de",
      "label": "transformers_wmt32k",
      "transformer": {
        "model": "transformer",
        "model_dir": "/tmp/t2t/train",
        "data_dir": "/tmp/t2t/data",
        "hparams": "",
        "hparams_set": "transformer_base_single_gpu",
        "problem": "translate_ende_wmt32k"
      },
    }],
    "language": [{
      "code": "en",
      "name": "English",
    },{
      "code": "de",
      "name": "German",
    }]
  }
```

然后启动：

```shell
t2t-insights-server \
          --configuration=configuration.json \
          --static_path=`pwd`/tensor2tensor/insights/polymer
```
