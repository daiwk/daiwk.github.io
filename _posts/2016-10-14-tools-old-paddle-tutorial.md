---
layout: post
category: "tools"
title: "老Paddle使用整理"
tags: [老paddle,]
---

# 老Paddle使用整理

以**bi-lstm+crf**进行品牌词识别为例，对老paddle的使用进行总结。

## 1. 预处理

### 1.1 预处理基本知识

### 1.1.1 预处理的文本结构：

>label;slotid0 fea1[:weight] fea2[:weight];slotid1 fea1[:weight] ...;

- fea:可以是离散值也可以是连续值
- slot数至少为1，也就是，至少有两个分号

<html>
<center>
<table border="2" cellspacing="0" cellpadding="6" rules="all" frame="border">

<thead>
<tr>
<th scope="col" class="left">说明</th>
<th scope="col" class="left">样本例子</th>
</tr>
</thead>

<tbody>
<tr>
<td class="left">连续特征，单slot，特征维数是5</td>
<td class="left">0;0 3.15 0 0 0 6.28;<br>
1;0 3.14 6.28 9.42; 非法，对于连续特征，每个slot必须包含的特征数应该跟维数一致。</td>
</tr>
<tr>
<td class="left">连续特征，2个slot，特征维数都是5</td>
<td class="left">0;0 0 1 2 3 4.5;1 3.14 0 0 0 0;<br>
1;1 3.14 6.28 0 0 0; 合法，slot0为空。</td>
</tr>
<tr>
<td class="left">离散特征，单slot，特征维数是1024</td>
<td class="left">0;0 1 2 3 5;<br>
1;0 1 32 64 512 1023;<br>
0;0 1023 1024; 非法，特征维数是1024，从0开始，所以最大的特征index只能是1023，特征越界。</td>
</tr>
<tr>
<td class="left">离散特征，3个slot，第1个slot有1024维特征，第2、3个slot有512维特征</td>
<td class="left">0;0 1 4 1023;1 4 7 3;2 2 6 511;<br>
1;2 1 5 88 511;</td>
</tr>
<tr>
<td class="left">离散带权特征，单slot，特征维数是1024</td>
<td class="left">0;0 1:3.14 2:6.28 0:0 4:12.56 1023:3.1415926;</td>
</tr>

</tbody>
</table></center>
</html>

#### 1.1.2 7个param：

#### 1.1.3 proto格式
假定一条样本为中文句子**“百度 公司 创立于 2000年”**，样本的类别为“0”。假定“百度”在词表中的id为23，“公司”为35，“创立于”为10，“2000年”为87，词表大小为23984，共有3个类别。
首先我们将其转换成文本格式（label; slot_id word_id1 word_id2 word_id3 word_id4……）【注意，slotid没有实际意义，只是一个编号，表示有多少维特征，从0开始递增】:

>0;0 23 35 10 87;

而这里我们要需要进行序列标注，所以在wordid这一维特征之外，还要用到别的两个特征，总共如下：

- wordid：假设训练集使用jieba分词完有4870个词。
- 词性：pos，假设使用jieba进行词性标注，那么有110个类别。[ICTCLAS 汉语词性标注集](http://fhqllt.iteye.com/blog/947917)
- 占位符：全部写0

比如，一个单词有8个汉字，那么，我们转化为：

>0;0 1383 2523 4396 1253 3967 4333 490 613;1 48 94 94 32 86 17 70 25;2 0 0 0 0 0 0 0 0;

接下来，使用txt2proto工具进行转换：

##### 1.1.3.1 local预处理

>cat INPUT_FILE | txt2proto OUTPUT_FILE "1" "23984 3"

##### 1.1.3.2 cluster预处理

```shell
# Local dir.
LOCAL_HADOOP_HOME=~/daiwenkai/local/hadoop-client-mulan/hadoop

# HDFS.
conf_hdfs=hdfs://nmg01-mulan-hdfs.dmop.baidu.com:54310
conf_ugi=fcr-opt,J798HJ9Kl
conf_tracker=nmg01-mulan-job.dmop.baidu.com:54311

# Data dir.
input_dir="/app/ecom/fcr-opt/daiwenkai/paddle/brand_recognize/input/data_test"
output_dir="/app/ecom/fcr-opt/daiwenkai/paddle/brand_recognize/preprocess/data_test_pb"
# Dictionary dir
dict_dir=
#If force_reuse_output_path is True ,paddle will remove outputdir without check outputdir exist
force_reuse_output_path=
# Job parameters.
JOBNAME="daiwenkai_gen_proto_test_brand_recognize"
MAPCAP=5000
REDCAP=5000
MAPNUM=1000
REDNUM=1000
```

#### 1.1.4 使用pyDataProvider格式


## 2. 集群版本