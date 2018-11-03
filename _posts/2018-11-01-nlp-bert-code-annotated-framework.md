---
layout: post
category: "nlp"
title: "bert代码解读——framework"
tags: [bert代码, bert code, framework]
---

目录

<!-- TOC -->

- [modeling.py](#modelingpy)
  - [BertConfig](#bertconfig)
- [extract_features.py](#extractfeaturespy)
- [optimization.py](#optimizationpy)
- [tokenization.py](#tokenizationpy)

<!-- /TOC -->

## modeling.py

高仿[https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/transformer_layers.py#L99](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/transformer_layers.py#L99)的transformer_encoder部分。

### BertConfig

```python
class BertConfig(object):
  """Configuration for `BertModel`."""

  def __init__(self,
               vocab_size, # xxx
               hidden_size=768,
               num_hidden_layers=12,
               num_attention_heads=12,
               intermediate_size=3072,
               hidden_act="gelu",
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               max_position_embeddings=512,
               type_vocab_size=16,
               initializer_range=0.02):
```


## extract_features.py

## optimization.py

## tokenization.py
