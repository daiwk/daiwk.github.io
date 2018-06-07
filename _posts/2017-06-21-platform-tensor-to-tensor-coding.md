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
    - [Hyperparameter Sets](#hyperparameter-sets)
    - [Trainer](#trainer)
- [2. 新增components](#2-新增components)
- [3. 新增数据集](#3-新增数据集)

<!-- /TOC -->

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

### Problems and Modalities

### Models

### Hyperparameter Sets

### Trainer

支持分布式训练，参考[https://tensorflow.github.io/tensor2tensor/distributed_training.html](https://tensorflow.github.io/tensor2tensor/distributed_training.html)，包括：

+ 多GPU
+ synchronous (1 master, many workers)
+ asynchronous (independent workers synchronizing through a parameter server)

## 2. 新增components

## 3. 新增数据集

