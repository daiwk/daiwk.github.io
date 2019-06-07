---
layout: post
category: "nlp"
title: "paddle的LARK(ERNIE/BERT等)+bert的各种变种"
tags: [paddle, bert, lark, ernie, ]
---

目录

<!-- TOC -->

- [bert](#bert)
    - [finetune和跑预测并save模型](#finetune和跑预测并save模型)
    - [线上infer部分](#线上infer部分)
- [ernie](#ernie)
    - [ernie(baiduNLP)](#erniebaidunlp)
    - [ernie(清华ACL2019版)](#ernie清华acl2019版)
        - [模型结构](#模型结构)
        - [finetune方法](#finetune方法)
- [XLMs/MASS/UNILM](#xlmsmassunilm)
    - [XLM(facebook2019)](#xlmfacebook2019)
    - [LASER(facebook2018)](#laserfacebook2018)
    - [MASS(microsoft)](#massmicrosoft)
    - [UNILM(microsoft)](#unilmmicrosoft)

<!-- /TOC -->

paddle的LARK里包含了一些nlp模型

## bert

### finetune和跑预测并save模型

```shell
BERT_BASE_PATH=./pretrained_model/chinese_L-12_H-768_A-12/
TASK_NAME="XNLI"
#DATA_PATH=./data/XNLI-1.0-demo/
DATA_PATH=./data/XNLI-MT-1.0-dwk/
INIT_CKPT_PATH=./output/step_50
SAVE_INFERENCE_PATH=./output/infer_step_50 ## 这个目录下会有个__model__文件，给在线infer用的，注意paddle的版本要用1.3.1以上的，1.3.0生成的这个目录有bug
python=../../python-2.7.14-paddle-1.3.1/bin/python


export FLAGS_enable_parallel_graph=1
export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CPU_NUM=3 ## 设置跑的cpu核数

TASK_NAME='XNLI'
CKPT_PATH=./output/

function finetune_xnli()
{

### 如果有56核的cpu，会占114g。。；而如果是12核的cpu，只会占25g内存
DATA_PATH=./data/XNLI-MT-1.0-dwk/
$python -u run_classifier.py --task_name ${TASK_NAME} \
           --use_cuda false \
           --do_train true \
           --do_val true \
           --do_test true \
           --batch_size 1 \
           --in_tokens false \
           --init_pretraining_params ${BERT_BASE_PATH}/params \
           --data_dir ${DATA_PATH} \
           --vocab_path ${BERT_BASE_PATH}/vocab.txt \
           --checkpoints ${CKPT_PATH} \
           --save_steps 50 \
           --weight_decay  0.01 \
           --warmup_proportion 0.0 \
           --validation_steps 2500 \
           --epoch 1 \
           --max_seq_len 8 \
           --bert_config_path ${BERT_BASE_PATH}/bert_config.json \
           --learning_rate 1e-4 \
           --skip_steps 1 \
           --random_seed 1
}

function save_inference_model() 
{

### 如果是56核cpu，会占22g内存..
DATA_PATH=./data/XNLI-1.0-demo/
$python -u predict_classifier.py --task_name ${TASK_NAME} \
           --use_cuda false \
           --batch_size 1 \
           --data_dir ${DATA_PATH} \
           --vocab_path ${BERT_BASE_PATH}/vocab.txt \
           --do_lower_case true \
           --init_checkpoint ${INIT_CKPT_PATH} \
           --max_seq_len 8 \
           --bert_config_path ${BERT_BASE_PATH}/bert_config.json \
           --do_predict true \
           --save_inference_model_path ${SAVE_INFERENCE_PATH}

}

function main()
{
    finetune_xnli
    [[ $? -ne 0 ]] && exit 1
    save_inference_model
    [[ $? -ne 0 ]] && exit 1
    return 0
}

main 2>&1 
```

### 线上infer部分

参考这个readme，[https://github.com/PaddlePaddle/LARK/tree/develop/BERT/inference](https://github.com/PaddlePaddle/LARK/tree/develop/BERT/inference)

生成demo文件可以参考

```shell
TASK_NAME="xnli"
DATA_PATH=../data/XNLI-1.0/
BERT_BASE_PATH=../pretrained_model/chinese_L-12_H-768_A-12/
python=../../../python-2.7.14-paddle-1.3.1/bin/python
$python gen_demo_data.py \
           --task_name ${TASK_NAME} \
           --data_path ${DATA_PATH} \
           --vocab_path "${BERT_BASE_PATH}/vocab.txt" \
           --batch_size 1 \
           > bert_data.demo

#           --in_tokens \
```

运行示例：

```shell
INFERENCE_MODEL_PATH=./output/infer_step_50
DATA_PATH=./bert_data.demo
REPEAT_TIMES=1
./bin/bert_demo --logtostderr \
        --model_dir $INFERENCE_MODEL_PATH \
        --data $DATA_PATH \
        --repeat $REPEAT_TIMES \
        --output_prediction
```

## ernie

也可以参考[站在BERT肩膀上的NLP新秀们（PART I）](https://blog.csdn.net/Kaiyuan_sjtu/article/details/90757442)

### ernie(baiduNLP)

参考[中文任务全面超越BERT：百度正式发布NLP预训练模型ERNIE](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650758722&idx=1&sn=6742b0f86982890d78cb3ec3be9865b3&scene=0#wechat_redirect)

[ERNIE: Enhanced Representation through Knowledge Integration](https://arxiv.org/pdf/1904.09223.pdf)

使用entity-level masking和phrase-level masking两种mask方法

输入的每个样本由5个 ';' 分隔的字段组成，数据格式：

+ token_ids
+ sentence_type_ids：两句话，第一句都是0，第二句都是1
+ position_ids
+ seg_labels：分词边界信息: 0表示词首、1表示非词首、-1为占位符, 其对应的词为 CLS 或者 SEP；
+ next_sentence_label

例如：

```shell
1 1048 492 1333 1361 1051 326 2508 5 1803 1827 98 164 133 2777 2696 983 121 4 19 9 634 551 844 85 14 2476 1895 33 13 983 121 23 7 1093 24 46 660 12043 2 1263 6 328 33 121 126 398 276 315 5 63 44 35 25 12043 2;0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55;-1 0 0 0 0 1 0 1 0 0 1 0 0 1 0 1 0 0 0 0 0 0 1 0 1 0 0 1 0 1 0 0 0 0 1 0 0 0 0 -1 0 0 0 1 0 0 1 0 1 0 0 1 0 1 0 -1;0
```

和bert在mask上的区别：

<html>
<br/>
<img src='../assets/ernie-bert-masking-diff.png' style='max-height: 200px'/>
<br/>
</html>

一个句子的不同level的mask方式：

<html>
<br/>
<img src='../assets/ernie-different-mask-level.png' style='max-width: 500px'/>
<br/>
</html>

### ernie(清华ACL2019版)

参考[ACL 2019 \| 清华等提出ERNIE：知识图谱结合BERT才是「有文化」的语言模型](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650762696&idx=4&sn=70c25ea24d15ed53880f45c511938813&chksm=871aa9b6b06d20a0536c7602a5757e28f995600bdffdd52ccb791927ba17aaaa10bfc15a209d&scene=0&xtrack=1&pass_ticket=10gfACXwvjCg3%2FChSZlp60K3dPTbQYhHe4njUqeSGdqo1x0Esjyqks8weqv1u2O0#rd)

ppt：[https://pan.baidu.com/s/1TjGOB2myXT-bln3OpFcbgA](https://pan.baidu.com/s/1TjGOB2myXT-bln3OpFcbgA) 提取码：35fv

[ERNIE: Enhanced Language Representation with Informative Entities](https://arxiv.org/pdf/1905.07129.pdf)

代码：[https://github.com/thunlp/ERNIE](https://github.com/thunlp/ERNIE)

+ 对于抽取并编码的知识信息，研究者首先**识别**文本中的**命名实体**，然后将这些提到的实体**与知识图谱中的实体进行匹配**。

研究者并不直接使用 KG 中基于图的事实，相反他们通过**知识嵌入**算法（例如，TransE，参考[Translating Embeddings for Modeling Multi-relational Data](https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf)，简单说就是(head,relation,tail)这个三元组，每个元素都是一个向量，期望head+relation尽可能=tail）**编码KG的图结构**，并将多信息实体嵌入作为ERNIE的输入。基于**文本和知识图谱的对齐**，ERNIE 将知识模块的实体表征整合到语义模块的隐藏层中。

+ 与 BERT 类似，研究者采用了带 Mask 的语言模型，以及预测下一句文本作为预训练目标。除此之外，为了更好地融合文本和知识特征，研究者设计了一种新型预训练目标，即随机**Mask掉**一些**对齐了输入文本的命名实体**，并要求模型从知识图谱中选择合适的实体以完成对齐。

现存的预训练语言表征模型只利用局部上下文预测 Token，但 ERNIE 的新目标要求模型**同时聚合上下文**和**知识事实**的信息，并**同时**预测**Token和实体**，从而构建一种知识化的语言表征模型。

研究者针对两种知识驱动型 NLP 任务进行了实验，即实体分型（entity typing）和关系分类。实验结果表明，ERNIE在知识驱动型任务中效果显著超过当前最佳的 BERT，因此 ERNIE 能完整利用词汇、句法和知识信息的优势。研究者同时在其它一般 NLP 任务中测试 ERNIE，并发现它能获得与 BERT 相媲美的性能。

#### 模型结构

<html>
<br/>
<img src='../assets/thu-ernie-arch.png' style='max-height: 400px'/>
<br/>
</html>

由两个堆叠的模块构成：

+ 底层的文本编码器（T-Encoder），负责获取输入 token 的词法和句法信息；
+ 上层的知识型编码器（K-Encoder），负责将额外的面向token的实体知识信息整合进来自底层的文本信息。这样我们就可以在一个统一的特征空间中表征token和实体的异构信息了。注意，**输出也是两部分**，token output和entity output，然后这两部分**各自过self-attention**，再进行information fusion。

N表示T-Encoder的层数，M表示K-Encoder的层数。

#### finetune方法

+ 针对relation classification问题，```[HD]```表示head entity，```[TL]```表示tail entity，然后```[CLS]```表示这个pair对的关系分类的label
+ 针对entity typing问题，其实是个简化版的relation classification问题，只要```[ENT]```这个就行了，然后这个实体是哪一类的，用```[CLS]```来表示，每条样本只预测一个```[ENT]```的分类。另一个实体用占位符替代。
+ 对于普通任务，把上述的```[TL]```、```[HD]```和```[ENT]```用占位符来替代就行。

<html>
<br/>
<img src='../assets/thu-ernie-finetune.png' style='max-height: 200px'/>
<br/>
</html>

## XLMs/MASS/UNILM

参考[站在BERT肩膀上的NLP新秀们：XLMs、MASS和UNILM](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247497613&idx=2&sn=11932f04ecc6a6999910385c0ffb4806&chksm=96ea280da19da11b1c4b6c7d93ca52a2f4aa035ec3a483d6936fc6e86aaf25aa59f6fb62e163&mpshare=1&scene=1&srcid=06063Y6CfgNf0h03W2Hd9iQ2&pass_ticket=TloMdmvUbLd5jnKvVTzrccQhGuskwL6KQ0HhJLF56Nwtcb16%2BVvMA09bw32tFrjs#rd)

参考[站在BERT肩膀上的NLP新秀们（PART II）](https://blog.csdn.net/Kaiyuan_sjtu/article/details/90814031)

### XLM(facebook2019)

facebook提出[Cross-lingual Language Model Pretraining](https://arxiv.org/abs/1901.07291)

### LASER(facebook2018)

[Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond](https://arxiv.org/abs/1812.10464)

XLM的主要思想还是来自于这片文章，借用了BERT的框架最后成了XLM。本文提出了LASER（Language-Agnostic SEntence Representations）

### MASS(microsoft)

bert只使用了Transformer的encoder部分，其下游任务也主要是适用于自然语言理解（NLU），对于类似文本摘要、机器翻译、对话应答生成等自然语言生成（NLG）任务显然是不太合适的。

[MASS: Masked Sequence to Sequence Pre-training for Language Generation](https://arxiv.org/abs/1905.02450)

受到bert的启发，作者们提出联合训练encoder和decoder的模型

训练步骤主要分为两步：

+ Encoder： 输入为被随机mask掉连续部分token的句子，使用Transformer对其进行编码；这样处理的目的是可以使得encoder可以更好地捕获没有被mask掉词语信息用于后续decoder的预测；
+ Decoder： 输入为与encoder同样的句子，但是mask掉的正好和encoder相反，和翻译一样，使用attention机制去训练，但只预测encoder端被mask掉的词。该操作可以迫使decoder预测的时候更依赖于source端的输入而不是前面预测出的token，防止误差传递。


### UNILM(microsoft)

[Unified Language Model Pre-training for Natural Language Understanding and Generation](https://arxiv.org/abs/1905.03197)

使用的核心框架还是 Transformer，不同的是预训练的目标函数结合了以下三个： 

+ 单向语言模型（同 ELMO/GPT） 
+ 双向语言模型（同 BERT） 
+ seq2seq 语言模型（同上一篇）

这里的 Transformer 是同一个，即三个 LM 目标参数共享，有点 multi-task learning 的感觉，可以学习到更 genneral 的文本表示。


以上四篇的小结：

+ BERT仅关注encoder，LASER考虑了encoder-decoder的联合训练。但从LASER模型框架上也可以看出，其**无法提取word级别的特征**，擅长于跨语言任务；
+ XLM和MASS都是跨语言模型，补充了BERT在NLG任务上的不足。模型层面而言XLM没有使用传统的encoder-decoder框架；
+ UNILM可以同时处理NLU和NLG任务。后续的改进可以考虑加入跨语言任务的预训练，比如XLM和MASS做的工作。
