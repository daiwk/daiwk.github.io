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
	+ parents (list/tuple/collection.Sequence) – Layer’s parents.
	
## Data layer

# data_layer

数据层的定义。用法：

```python
data = data_layer(name="input",
                  size=1000)
```

+ Params:
	+ name (basestring) – Name of this data layer.
	+ size (int) – Size of this data layer.
	+ layer_attr (ExtraLayerAttribute.) – Extra Layer Attribute.
+ Returns:	
	+ LayerOutput object.
+ Return type:	
	+ LayerOutput


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

与img_conv_layer不同，conv_op是一个Operator，能在mixed_layer里面使用。conv_op需要两个input来perform convolution。第一个input是image，第二个input是filter kernel。**只支持GPU mode**。用法：

```python
op = conv_operator(img=input1,
                   filter=input2,
                   filter_size=3,
                   num_filters=64,
                   num_channels=64)
```

+ Params:
	+ **img (LayerOutput) – input image**
	+ **filter (LayerOutput) – input filter**
	+ filter_size (int) – The x dimension of a filter kernel.
	+ filter_size_y (int) – The y dimension of a filter kernel. Since PaddlePaddle now supports rectangular filters, the filter’s shape can be (filter_size, filter_size_y).
	+ num_filters (int) – channel of output data.
	+ num_channel (int) – channel of input data.
	+ stride (int) – The x dimension of the stride.
	+ stride_y (int) – The y dimension of the stride.
	+ padding (int) – The x dimension of padding.
	+ padding_y (int) – The y dimension of padding.
+ Returns:
	+ a ConvOperator Object.
+ Return tpye:
	+ ConvOperator

## conv_shift_layer

计算两个input的cyclic(环形的，循环的) convolution：

`\[
c[i]=\sum _{j-(N-1)/2}^{(N-1)/2}a_{i+1}*b_j
\]`

上式中，`\(a\)`有`\(M\)`个元素，`\(b\)`有`\(N\)`个元素(`\(N\)`为奇数)，`\(c\)`有`\(M\)`个元素。当`\(a\)`或`\(b\)`的下标为负数时，表示从右往左的下标。

用法：

```python
conv_shift = conv_shift_layer(input=[layer1, layer2])
```

+ Params:
	+ name (basestring) – layer name
	+ a (LayerOutput) – Input layer a.
	+ b (LayerOutput) – input layer b
+ Returns:
	+ LayerOutput object
+ Return type:
	+ LayerOutput

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


