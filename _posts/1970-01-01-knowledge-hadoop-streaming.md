---
layout: post
category: "knowledge"
title: "hadoop streaming"
tags: [hadoop streaming, ]
---

目录

<!-- TOC -->

- [排序相关](#排序相关)
- [如何确定map任务数](#如何确定map任务数)
- [如何确定reduce任务数](#如何确定reduce任务数)

<!-- /TOC -->

参考。。内部文档

## 排序相关

通过partition来简单地对key进行二次排序

```shell
# -D map.output.key.field.separator=.
# -D stream.num.map.output.key.fields=2: 输出有n部分，前两列是key
# -D stream.map.output.field.separator=.
# -D num.key.fields.for.partition=1: key里面第1列拿来分桶。第一列相同的分在一个桶里，然后桶内再按第2列排序 
```

还可以通过```-D mapred.text.key.partitioner.options=spec```来指定更详细的分桶方式

而```KeyFieldBasedComparator```可以指定key的排序方式

## 如何确定map任务数

确定map任务数时依次优先参考如下几个原则：

+ 每个map任务使用的内存不超过800M，尽量在500M以下

比如处理256MB数据需要的时间为10分钟，内存为800MB，此时如果处理128MB时，内存可以减小为400MB，则选择每一个map的处理数据量为128MB

+ 每个map任务运行时间控制在大约20分钟，最好1-3分钟

比如处理256MB数据需要的时间为30分钟，内存为200MB，则应该考虑减小map的计算时间，比如将每一个map的处理数据量设置为128MB，将时间减小为15分钟。

+ 每个map任务处理的最大数据量为一个HDFS块大小（目前为256MB），一个map任务处理的输入不能跨文件

比如指定map任务数为N，输入数据总量为S。如果S / N > 256MB，平台会自动增加map任务数使每个map任务处理数据量不超过256MB；如果S / N < 256MB，平台认为每个map任务最多处理S/N大小的数据，但是一个map任务的输入不能跨文件，所以可能有的文件切分到最后一部分时较小于S/N，那么下一个map任务的输入小于平均，最终的map任务数大于N。最终实际运行的map任务数可以在JobTracker监控页面查看。

+ map任务总数不超过平台可用的任务槽位

如果在一个map处理256MB时就能将平台可用的任务槽位占满，此时不应该再增加map任务数。

## 如何确定reduce任务数

确定reduce任务数时依次优先参考如下几个方面：

+ 每个reduce任务使用的内存不超过800M，尽量在500M以下
+ 每个reduce任务运行时间控制在大约20分钟，最好1-3分钟
+ 整个reduce阶段的输入数据总量

reduce的输入是map的输出，整个reduce阶段的输入数据总量就是所有map任务的输出数据总量，如果map输出的中间结果较大，推荐使用lzo进行压缩

+ 每个reduce任务处理的数据量控制在500MB以内

由于一个reducer处理的数据会按照key进行排序，每个reduce任务处理数据量过大会导致shuffle和排序时间很长

+ map任务数与reduce任务数的乘积

在reduce的shuffle阶段每一个reducer需要到多个mapper去用tcp连接来下载自己要处理的数据。如果map任务数和reduce任务数乘积较大，可能造成需要建立过多的tcp连接，如果每一次连接传输数据又很少，就会导致reduce的 shuffle时间很长。建议每个reduce的一次连接下载的数据量 ＝ map输出数据总量 /（map数*reduce数）不要小于100KB。

+ 输出数据要求

如果reduce计算后的结果数据用于下一次MapReduce计算，或者结果文件不宜太小，那么在满足或者大致满足以上原则的前提下，可以调小reduce任务数，以便每一个reduce任务数的输出不会太小。
