---
layout: post
category: "nlp"
title: "paddlepaddle layers"
tags: [paddlepaddle, layers]
---

## Base

### LayerType

layer type enumerations.

+ Params:
	+ type_name (basestring) – layer type name. Because layer type enumerations are strings.
+ Returns:
	+ True if is a layer_type
+ Return type:
	+ bool

### LayerOutput

layer函数的输出，主要用于：
+ 检查layer的connection是否make sense： 例如，FC(Softmax) => Cost(MSE Error) is not good
+ tracking layer connection
+ 作为layer方法的输入

+ Params:
	+ name (basestring) – Layer output name.
	+ layer_type (basestring) – Current Layer Type. One of LayerType enumeration.
	+ activation (BaseActivation.) – Layer Activation.
	+ parents (list|tuple|collection.Sequence) – Layer’s parents.
	
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

+ Params:
	+ name (basestring) : The Layer Name.
	+ input (LayerOutput/list/tuple) : The input layer. Could be a list/tuple of input layer.
	+ size (int) : The layer dimension.
	+ act (BaseActivation) – Activation Type. Default is **tanh**.
	+ param_attr (ParameterAttribute) – The Parameter Attribute/list.
	+ bias_attr (ParameterAttribute/None/Any) – The Bias Attribute. If no bias, then pass False or something not type of ParameterAttribute. None will get a default Bias.
	+ layer_attr (ExtraLayerAttribute/None) – Extra Layer config.
+ Returns:
	+ LayerOutput object
+ Return Type:
	+ LayerOutput

## selective_fc_layer

与全连接层的区别：输出可能是sparse的。有一个select参数，指定several selected columns for output。如果select参数没有被指定，那么他和fc_layer是一样的。用法如下：

```python
sel_fc = selective_fc_layer(input=input, size=128, act=TanhActivation())
```

+ Params:
	+ name (basestring) : The Layer Name.
	+ input (LayerOutput/list/tuple) : The input layer. Could be a list/tuple of input layer.
	+ **select (LayerOutput)** : The select layer. The output of select layer should be **a sparse binary matrix**, and treat as the mask of selective fc.
	+ size (int) : The layer dimension.
	+ act (BaseActivation) – Activation Type. Default is **tanh**.
	+ param_attr (ParameterAttribute) – The Parameter Attribute/list.
	+ bias_attr (ParameterAttribute/None/Any) – The Bias Attribute. If no bias, then pass False or something not type of ParameterAttribute. None will get a default Bias.
	+ layer_attr (ExtraLayerAttribute/None) – Extra Layer config.
+ Returns:
	+ LayerOutput object
+ Return Type:
	+ LayerOutput

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


