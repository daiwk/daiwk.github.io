---
layout: post
category: "nlp"
title: "albert"
tags: [albert, ]
---

目录

<!-- TOC -->


<!-- /TOC -->

预训练小模型也能拿下13项NLP任务，谷歌ALBERT三大改造登顶GLUE基准](https://mp.weixin.qq.com/s/kvSoDr0E_mvsc7lcLNKmgg)

ALBERT 模型在 GLUE、RACE 和 SQuAD 基准测试上都取得了新的 SOTA 效果，并且参数量还少于 BERT-large。

[ALBERT: a lite bert for self-supervised learning of language representations](https://openreview.net/pdf?id=H1eA7AEtvS)

emb只有128？

<html>
<br/>
<img src='../assets/albert-params.jpg' style='max-height: 100px'/>
<br/>
</html>

通过对词嵌入矩阵进行因式分解，再为下游任务共享不同层的所有参数，这样可以大大降低 BERT 的参数量。

还提出了一种新型句间连贯性损失函数，它可以强迫模型学习句间的连贯性表达，从而有利于各种下游 NLP 任务。

ALBERT 通过两个参数削减技术克服了扩展预训练模型面临的主要障碍。第一个技术是对嵌入参数化进行因式分解。研究者将大的词汇嵌入矩阵分解为两个小的矩阵，从而将隐藏层的大小与词汇嵌入的大小分离开来。这种分离使得隐藏层的增加更加容易，同时不显著增加词汇嵌入的参数量。

第二种技术是跨层参数共享。这一技术可以避免参数量随着网络深度的增加而增加。两种技术都显著降低了 BERT 的参数量，同时不对其性能造成明显影响，从而提升了参数效率。ALBERT 的配置类似于 BERT-large，但参数量仅为后者的 1/18，训练速度却是后者的 1.7 倍。这些参数削减技术还可以充当某种形式的正则化，可以使训练更加稳定，而且有利于泛化。

为了进一步提升 ALBERT 的性能，研究者还引入了一个自监督损失函数，用于句子级别的预测（SOP）。SOP 主要聚焦于句间连贯，用于解决原版 BERT 中下一句预测（NSP）损失低效的问题。

参考[谷歌全新轻量级新模型ALBERT刷新三大NLP基准！](https://mp.weixin.qq.com/s/-Kzr4w7pv2DCOUlilg2Wfg)





例子，albert_tiny：

input_ids先查word_embeddings(`\(V\times E=21118*128)`)，得到dim=128的表示，再查word_embeddings_2(`\(E\times M =128*312\)`)，得到dim=312的表示。

搞positionembedding时，并不用输入0 1 2...，只需要做一些slice的变换就行了

```python
    with tf.control_dependencies([assert_op]):
      full_position_embeddings = tf.get_variable(
          name=position_embedding_name,
          shape=[max_position_embeddings, width],
          initializer=create_initializer(initializer_range))
      # Since the position embedding table is a learned variable, we create it
      # using a (long) sequence length `max_position_embeddings`. The actual
      # sequence length might be shorter than this, for faster training of
      # tasks that do not have long sequences.
      #    
      # So `full_position_embeddings` is effectively an embedding table
      # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
      # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
      # perform a slice.
      position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                     [seq_length, -1]) 
      num_dims = len(output.shape.as_list())

      # Only the last two dimensions are relevant (`seq_length` and `width`), so
      # we broadcast among the first dimensions, which is typically just
      # the batch size.
      position_broadcast_shape = [] 
      for _ in range(num_dims - 2):
        position_broadcast_shape.append(1)
      position_broadcast_shape.extend([seq_length, width])
      position_embeddings = tf.reshape(position_embeddings,
                                       position_broadcast_shape)
      output += position_embeddings
```

然后会通过create_attention_mask_from_input_mask把input_ids和input_mask搞一下，得到attention_mask去和attention做mask，主要是算loss啥的，把后面的mask掉不算



