---
layout: post
category: "platform"
title: "paddlepaddle layers"
tags: [paddlepaddle, layers]
---

目录

暂无（格式有问题）

参考：

Layers:

[http://www.paddlepaddle.org/doc/ui/api/trainer_config_helpers/layers.html](http://www.paddlepaddle.org/doc/ui/api/trainer_config_helpers/layers.html)

ParameterAttribute:
[http://www.paddlepaddle.org/doc/ui/api/trainer_config_helpers/attrs.html](http://www.paddlepaddle.org/doc/ui/api/trainer_config_helpers/attrs.html)

## Parameter & Extra Layer Attribute

#### ParameterAtribute

在fine-tuning training的过程中，可以设置这个object来控制training的详情，诸如l1/l2 rate/learning rate/如何初始化参数等

+ Params:
	+ name (basestring) – default parameter name.
	+ is_static (bool) – True if this parameter will be fixed while training.
	+ initial_std (float or None) – Gauss Random initialization standard deviation. None if not using Gauss Random initialize parameter.
	+ initial_mean (float or None) – Gauss Random initialization mean. None if not using Gauss Random initialize parameter.
	+ initial_max (float or None) – Uniform initialization max value.
	+ initial_min (float or None) – Uniform initialization min value.
	+ l1_rate (float or None) – the l1 regularization factor
	+ l2_rate (float or None) – the l2 regularization factor
	+ learning_rate (float or None) – The parameter learning rate. None means 1. The learning rate when optimize is LEARNING_RATE = GLOBAL_LEARNING_RATE * PARAMETER_LEARNING_RATE * SCHEDULER_FACTOR.
	+ momentum (float or None) – The parameter momentum. None means use global value.
	+ sparse_update (bool) – Enable sparse update for this parameter. It will enable both local and remote sparse update

#### set_default_parameter_name

设置parameter的默认名字，如果不设，那就用默认的parameter name

+ Params:
	+ name (basestring) – default parameter name.

#### ExtraLayerAtribute

一些高阶的layer attribute设置，可以设置所有，但有些layer并不支持所有attribute，**一旦设置了不支持的，会报错且core……**

+ Params:
	+ error_clipping_threshold (float) – Error clipping threshold.
	+ drop_rate (float) – Dropout rate. Dropout will create a mask on layer output. The dropout rate is the zero rate of this mask. The details of what dropout is please refer to here.
	+ device (int) – device ID of layer. device=-1, use CPU. device>0, use GPU. The details allocation in parallel_nn please refer to here.

#### ParamAttr

是ParameterAtribute的alias

#### ExtraAttr

是ExtraLayerAttribute的alias

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

image的卷积层。目前Paddle只支持width=height的正方形图片作为输入。卷积层的具体定义见[UFLDL](http://ufldl.stanford.edu/tutorial/supervised/FeatureExtractionUsingConvolution/)。

其中的num_channel是输入的image的channel数，当输入是图片时可以是1或者3（mono【单通道】 or RGB），当输入是layer时，可以是上一个layer的num_filters * num_group

Paddle中有一些group的filter，每个group可以处理inputs的一些channel。例如，一个input num_channel=256, group=4, num_filter=32，那么Paddle会生成**32*4=128个filter**对inputs进行处理。channels会被分成4部分，first 256/4个channels被first 32个filters处理，以此类推。

+ Params:
	+ name (basestring) – Layer name.
	+ input (LayerOutput) – Layer Input.
	+ filter_size (int/tuple/list) – The x dimension of a filter kernel. Or input a tuple for two image dimension.
	+ filter_size_y (int/None) – The y dimension of a filter kernel. Since PaddlePaddle currently supports rectangular filters, the filter’s shape will be (filter_size, filter_size_y).
	+ num_filters – Each filter group’s number of filter
	+ act (BaseActivation) – Activation type. Default is tanh
	+ groups (int) – Group size of filters.
	+ stride (int/tuple/list) – The x dimension of the stride. Or input a tuple for two image dimension.
	+ stride_y (int) – The y dimension of the stride.
	+ padding (int/tuple/list) – The x dimension of the padding. Or input a tuple for two image dimension
	+ padding_y (int) – The y dimension of the padding.
	+ bias_attr (ParameterAttribute/False) – Convolution bias attribute. None means default bias. False means no bias.
	+ num_channels (int) – number of input channels. If None will be set automatically from previous output.
	+ param_attr (ParameterAttribute) – Convolution param attribute. None means default attribute
	+ shared_biases (bool) – Is biases will be shared between filters or not.
	+ layer_attr (ExtraLayerAttribute) – Layer Extra Attribute.
+ Returns:
	+ LayerOutput object
+ Return type:
	+ LayerOutput

## context_projection

将一个序列根据设置的context_len，转化为以context_start=-(context_len - 1) / 2为开头的序列。如果context position超出了序列的长度，如果padding_attr=False，那么将会填充0，否则，padding是可以通过训练得到的（learnable），并将此变量赋值为ParameterAttribute类型。

例如，原始序列是[a b c d e f g]，context_len=3，padding_attr=False，那么，context_start = -1，所以，产生的序列就是[0ab abc bcd cde def efg fg0]

+ Params:
	+ input (LayerOutput) – Input Sequence.
	+ context_len (int) – context length.
	+ context_start (int) – context start position. Default is -(context_len - 1)/2
	+ padding_attr (bool/ParameterAttribute) – Padding Parameter Attribute. If false, it means padding always be zero. Otherwise Padding is learnable, and parameter attribute is set by this parameter.
+ Returns:
	+ Projection
+ Return type:
	+ Projection

# Image Pooling Layer

## img_pool_layer

图像处理中的pooling层，详见[UFLDL的pooling介绍](http://ufldl.stanford.edu/tutorial/supervised/Pooling/)

+ Params：
	+ padding (int) – pooling padding
	+ name (basestring.) – name of pooling layer
	+ input (LayerOutput) – layer’s input
	+ pool_size (int) – pooling size
	+ num_channels (int) – number of input channel.
	+ pool_type (BasePoolingType) – pooling type. MaxPooling or AveragePooling. **Default is MaxPooling.**
	+ stride (int) – stride of pooling.
	+ start (int) – start position of pooling operation.
	+ layer_attr (ExtraLayerAttribute) – Extra Layer attribute.
+ Returns：
	+ LayerOutput object
+ Return type:
	+ LayerOutput

# Norm Layer

## img_cmrnorm_layer

Response normalization across feature maps，详见[Alex的paper(alexnet?)](../assets/ImageNet Classification with Deep Convolutional Neural Networks.pdf)

+ Params:
	+ name (None/basestring) – layer name.
	+ input (LayerOutput) – layer’s input.
	+ size (int) – Normalize in number of sizesize feature maps.
	+ scale (float) – The hyper-parameter.
	+ power (float) – The hyper-parameter.
	+ num_channels – input layer’s filers number or channels. If num_channels is None, it will be set automatically.
	+ layer_attr (ExtraLayerAttribute) – Extra Layer Attribute.
+ Returns:
	+ LayerOutput object.
+ Return type:
	+ LayerOutput

## batch_norm_layer

batch normalization，定义如下(详见[paper](../assets/Batch Normalization Accelerating Deep Network Training by Reducing Internal Covariate Shift.pdf))：

`\[
\mu _\beta\leftarrow \frac{1}{m}\sum _{i=1}^mx_i \ \ //\ mini-batch\ mean \\
\sigma _\beta^2\leftarrow \frac{1}{m}\sum _{i=1}^m(x_i-\mu _\beta)^2 \ \ //\ mini-batch\ variance \\
\hat{x_i}\leftarrow \frac{(x_i-\mu _\beta )}{\sqrt{\sigma _\beta ^2+\epsilon }} \ \ //\ normalize \\
y_i \leftarrow \gamma \hat{x_i}+\beta \ \ //\ scale\ and\ shift \\
\]`

上式中，`\(x\)`是一个mini-batch的input features

+ Params:
	+ name (basestring) – layer name.
	+ input (LayerOutput) – batch normalization input. Better be linear activation. Because there is an activation inside batch_normalization.
	+ batch_norm_type (None|string, None or "batch_norm" or "cudnn_batch_norm") – We have batch_norm and cudnn_batch_norm. batch_norm supports both CPU and GPU. cudnn_batch_norm requires cuDNN version greater or equal to v4 (>=v4). But cudnn_batch_norm is faster and needs less memory than batch_norm. By default (None), we will automaticly select cudnn_batch_norm for GPU and batch_norm for CPU. Otherwise, select batch norm type based on the specified type. If you use cudnn_batch_norm, we suggested you use latest version, such as v5.1.
	+ act (BaseActivation) – Activation Type. Better be relu. Because batch normalization will normalize input near zero.
	+ num_channels (int) – num of image channels or previous layer’s number of filters. None will automatically get from layer’s input.
	+ bias_attr (ParameterAttribute) – ββ, better be zero when initialize. So the initial_std=0, initial_mean=1 is best practice.
	+ param_attr (ParameterAttribute) – γγ, better be one when initialize. So the initial_std=0, initial_mean=1 is best practice.
	+ layer_attr (ExtraLayerAttribute) – Extra Layer Attribute.
	+ use_global_stats (bool/None.) – whether use moving mean/variance statistics during testing peroid. If None or True, it will use moving mean/variance statistics during testing. If False, it will use the mean and variance of current batch of test data for testing.
	+ moving_average_fraction (float.) – Factor used in the moving average computation, referred to as facotr,`\(runningMean=newMean*(1-factor)+runningMean*factor\)`
+ Returns:
	+ LayerOutput object.
+ Return type:
	+ LayerOutput

## sum_to_one_norm_layer

NEURAL TURING MACHINE中用到的sum-to-one normalization:

`\[
out[i]=\frac{in[i]}{\sum _{k-1}^{N}in[k]}
\]`

其中，`\(in\)`是一个输入vector（batch_size * data_dim）,`\(out\)`是一个输出vector（batch_size * data_dim）。用法：

```python
sum_to_one_norm = sum_to_one_norm_layer(input=layer)
```

+ Params:
	+ input (LayerOutput) – Input layer.
	+ name (basestring) – Layer name.
	+ layer_attr (ExtraLayerAttribute.) – extra layer attributes.
+ Returns:	
	+ LayerOutput object.
+ Return type:	
	+ LayerOutput

# Recurrent Layers

## recurrent_layer

最简单的recurrent unit layer，只是全连接层through both time and neural network。

对每个[start, end]的序列，计算：

`\[
out_i=act(in_i)\ for\ i=start \\
out_i=act(in_i+out_{i-1}*W)\ for\ i<start<=end \\
\]`

如果reverse=True，那么：

`\[
out_i=act(in_i)\ for\ i=end \\
out_i=act(in_i+out_{i+1}*W)\ for\ i<=start<end \\
\]`

+ Params:
	+ input (LayerOutput) – Input Layer
	+ act (BaseActivation) – activation.
	+ bias_attr (ParameterAttribute) – bias attribute.
	+ param_attr (ParameterAttribute) – parameter attribute.
	+ name (basestring) – name of the layer
	+ layer_attr (ExtraLayerAttribute) – Layer Attribute.
+ Returns:	
	+ LayerOutput object.
+ Return type:	
	+ LayerOutput

## lstmemory

公式如下：

`\[
i_t=\sigma (W_ \\
\]`


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

**序列输入的pooling层，和图像输入不一样！！！**

```python
seq_pool = pooling_layer(input=layer,
                         pooling_type=AvgPooling(),
                         agg_level=AggregateLevel.EACH_SEQUENCE)
```

+ Params：
	+ agg_level (AggregateLevel) – AggregateLevel.EACH_TIMESTEP or AggregateLevel.EACH_SEQUENCE
	+ name (basestring) – layer name.
	+ input (LayerOutput) – input layer name.
	+ pooling_type (BasePoolingType|None) – Type of pooling, MaxPooling(default), AvgPooling, SumPooling, SquareRootNPooling.
	+ bias_attr (ParameterAttribute|None|False) – Bias parameter attribute. False if no bias.
	+ layer_attr (ExtraLayerAttribute|None) – The Extra Attributes for layer, such as dropout.
+ Returns：
	+ LayerOutput object
+ Return type:
	+ LayerOutput


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


