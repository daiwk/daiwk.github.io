---
layout: post
category: "nlp"
title: "paddlepaddle layers"
tags: [paddlepaddle, layers]
---

## Base

### LayerType

### LayerOutput

## Data layer

# data_layer

# Fully Connected Layers

## fc_layer

全连接层，用法：

```python
fc = fc_layer(input=layer,
              size=1024,
              act=LinearActivation(),
              bias_attr=False)
```

等价于：

```python
with mixed_layer(size=1024) as fc:
    fc += full_matrix_projection(input=layer)
```

<html>
<center>
<table border="2" cellspacing="0" cellpadding="6" rules="all" frame="border">

<tbody>
<tr>
<td class="left">train.txt</td>
<td class="left">220663</td>
<td class="left">8936</td>
</tr>
<tr>
<td class="left">test.txt</td>
<td class="left">49389</td>
<td class="left">2012</td>
</tr>

</tbody>
</table></center>
</html>
<br>


## selective_fc_layer

# Conv Layers

## conv_operator

## conv_shift_layer

## img_conv_layer

## context_projection

# Image Pooling Layer

## img_pool_layer

# Norm Layer

## img_cmrnorm_layer

## batch_norm_layer

## sum_to_one_norm_layer

# Recurrent Layers

## recurrent_layer

## lstmemory

## lstm_step_layer

## grumemory

## gru_step_layer

# Recurrent Layer Group

## recurrent_group

## beam_search

## get_output_layer

# Mixed Layer

## mixed_layer

## embedding_layer

## dotmul_projection

## dotmul_operator

## full_matrix_projection

## identity_projection

## table_projection

## trans_full_matrix_projection

# Aggregate Layers

## pooling_layer

## last_seq

## first_seq

## concat_layer

# Reshaping Layers

## block_expand_layer

## expand_layer

# Math Layers

## addto_layer

## linear_comb_layer

## interpolation_layer

## power_layer

## scaling_layer

## slope_intercept_layer

## tensor_layer

## cos_sim

## trans_layer

# Sampling Layers

## maxid_layer

## sampling_id_layer

# Cost Layers

## cross_entropy

## cross_entropy_with_selfnorm

## multi_binary_label_cross_entropy

## huber_cost

## lambda_cost

## rank_cost

## crf_layer

## crf_decoding_layer

## ctc_layer

## hsigmoid

# Check Layer

## eos_layer


