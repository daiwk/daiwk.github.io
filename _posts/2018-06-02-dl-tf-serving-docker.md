---
layout: post
category: "dl"
title: "tf serving docker+k8s"
tags: [tf serving, docker, k8s ]
---

目录

<!-- TOC -->

- [tf-serving](#tf-serving)
- [基本流程](#基本流程)
    - [将graph进行freeze](#将graph进行freeze)
    - [调整推断代码](#调整推断代码)
    - [容器化](#容器化)
    - [添加API层](#添加api层)
        - [部署可以运行推断脚本的扩展容器](#部署可以运行推断脚本的扩展容器)
        - [部署运行API层的扩展容器](#部署运行api层的扩展容器)
    - [缓解计算成本的累积](#缓解计算成本的累积)
        - [重复使用会话](#重复使用会话)
        - [缓存输入](#缓存输入)
        - [使用任务队列](#使用任务队列)
    - [部署小结](#部署小结)
- [docker + tf-serving[from 阿里云]](#docker--tf-servingfrom-阿里云)
- [docker + k8s + tf-serving[自己搞]](#docker--k8s--tf-serving自己搞)
    - [docker+tf-serving](#dockertf-serving)
        - [创建docker镜像](#创建docker镜像)
        - [编译server](#编译server)
        - [编译examples](#编译examples)
            - [mnist example](#mnist-example)
            - [inception example](#inception-example)
        - [commit container](#commit-container)
        - [在2个容器间进行访问](#在2个容器间进行访问)
- [k8s](#k8s)

<!-- /TOC -->

## tf-serving

[https://github.com/tensorflow/serving](https://github.com/tensorflow/serving)

安装方式见[https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/setup.md](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/setup.md)

## 基本流程

参考[没人告诉你的大规模部署AI高效流程！](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650755184&idx=4&sn=f597d101749e9bbc1186a95900f4a26d&chksm=871a940eb06d1d18070d8c8dadb9e2dd36397a1734da8932776a1b8358b35af694814865a01d&mpshare=1&scene=1&srcid=0107WNE0JGfEKwXhHPQWbClR&pass_ticket=mPnDPDR4heSU20VXT7N8W622Cb1dZmIzkNcF8BygI%2Bp60d7GrSesIej%2FlrFbnO84#rd)

+ 将graph**固化**为Protobuf二进制文件
+ **调整推断代码**，使它可以**处理固化的图**
+ **容器化**应用程序
+ 在最上面加上**API层**

### 将graph进行freeze

「固化」graph要用所有命名节点、权重、架构和检查点元数据，并创建一个protobuf二进制文件。最常用的是tf自己的工具，它可以固化任何给定输出节点名字的graph。参考[https://www.tensorflow.org/guide/extend/model_files#freezing](https://www.tensorflow.org/guide/extend/model_files#freezing)

### 调整推断代码

在大多数情况下，feed_dict 是不变的，主要区别在于添加了加载模型的代码，也许还有输出节点的规范。

### 容器化

只要在 Dockerfile 中设置环境即可

### 添加API层

两种通用的方法：

#### 部署可以运行推断脚本的扩展容器

这些容器根据输入运行脚本，脚本启动一个会话并执行推断，再**通过管道**返回输出结果。这种方法效率是很低的：

+ 对大多数云供应商而言添加一个可以操纵容器和管道进出的 API 层并不容易
+ 在启动容器、分配硬件、启动会话以及推断时会损失宝贵的时间
+ 你让stdin开着并保持管道输出，那么你的脚本就会加速但是会失去可扩展性

#### 部署运行API层的扩展容器

这种方法效率更高:

+ 虽然这需要更多资源，但它已经用了最少资源而且没有垂直扩展
+ 允许每个容器保持运行状态
+ 由于这种情况下 API 是分散的，因此可以将特定的stdin/stout连接到主要的请求路由器上
+ 省去了启动时间，可以在服务多个请求的同时维持速度并保证水平扩展
+ 可以用负载平衡器集中容器，并用Kubernetes保证近乎100%的运行时间并管理集群

### 缓解计算成本的累积

通过容器集群分散 API 的主要缺点在于计算成本会相对较快地累积起来。这在AI中是不可避免的，但有一些方法可以缓解这一问题。

#### 重复使用会话

集群会根据负载成比例地增长和收缩，因此你的目标是**最小化执行推断的时间**，使**容器**可以**释放出来**处理另外的请求。

所以可以初始化```tf.Session```和```tf.Graph```后就将它们**存储起来**并将它们**作为全局变量**传递，以达到重复使用```tf.Session```和```tf.Graph```的目的。

这样做可以减少启动会话和构建图的时间，从而大大提高推断任务的速度，即便是单个容器，这个方法也是有效的。这一技术被广泛用于资源再分配最小化和效率最大化。

#### 缓存输入

如果可能的话还要缓存输出。

**动态规划范式**在AI中是最重要的。缓存输入，你可以节省预处理输入或从远程获得输入的时间；缓存输出，你可以节省运行推断的时间。

通常，你的模型会随着时间的推移变得更好，但这会很大程度上影响你的**输出缓存机制**。例如，可以使用80-20原则，当模型准确率低于80% 时，不会缓存任何输出；一旦准确率到了80%，就**开始缓存**并设置为在**准确率到一定值**（而不是某个时间点）的时候**停止缓存**。

随着模型变得越来越准确，输出也会发生变化，但是在「80-20」缓存中，性能和速度之间存在的权衡更少。

#### 使用任务队列

一般需要运行或大或小的推断任务，对UX来说，使用**堆队列**（heap queue）可能更好，它会优先处理小一些的任务，这样，要运行简单步骤的用户只要等这一步结束就行了，而不必等另一个用户的更大推断任务先完成。

在带有任务队列的专用GPU上训练模型。如果你要将每个交互返回到模型中进行训练，请考虑在单独的服务器或GPU上运行。一旦训练结束，你就可以将模型（在AWS中，你可以将模型repo集中在S3中）部署到容器中了。

### 部署小结

+ 固化图并将推断封装在API下
+ 重复使用会话和图，缓存输入和输出
+ 用Docker容器化应用程序（包括API层）
+ 将大规模应用程序与Kubernetes一起部署在你选择的云上
+ 将训练从推断中分离出来
+ 建立任务队列，提高较小的任务的运行优先级

## docker + tf-serving[from 阿里云]

**注意：1.6版本的docker不行（没有--link参数，-p参数不是port），亲测1.9.1的docker可以。。**

参考：[https://yq.aliyun.com/articles/60894](https://yq.aliyun.com/articles/60894)

这两个镜像[发现bazel是0.3.0，而serving是2016.9时的版本]：

+ registry.cn-hangzhou.aliyuncs.com/denverdino/tensorflow-serving : TensorFlow Serving的基础镜像
+ registry.cn-hangzhou.aliyuncs.com/denverdino/inception-serving : 基于上述基础镜像添加Inception模型实现的服务镜像

## docker + k8s + tf-serving[自己搞]

[https://www.tensorflow.org/serving/serving_inception](https://www.tensorflow.org/serving/serving_inception)

### docker+tf-serving

参考[https://www.tensorflow.org/serving/docker](https://www.tensorflow.org/serving/docker)

#### 创建docker镜像

首先把这个搞下来[https://github.com/tensorflow/serving/blob/master/tensorflow_serving/tools/docker/Dockerfile.devel](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/tools/docker/Dockerfile.devel)：

然后，把bazel的version改成0.11.0（编译1.7版本的serving需要），另外，我还补充了automake/libtool【**当然，目前可以work的是1.4版本的serving，要把bazel版本改成0.5.4**】：

```shell
FROM ubuntu:16.04

MAINTAINER Jeremiah Harmsen <jeremiah@google.com>

RUN apt-get update && apt-get install -y \
        build-essential \
        curl \
        git \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        mlocate \
        pkg-config \
        python-dev \
        python-numpy \
        python-pip \
        software-properties-common \
        swig \
        zip \
        zlib1g-dev \
        libcurl3-dev \
        openjdk-8-jdk\
        openjdk-8-jre-headless \
        wget \
        automake \
        libtool \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set up grpc

RUN pip install mock grpcio

# Set up Bazel.

ENV BAZELRC /root/.bazelrc
# Install the most recent bazel release.
ENV BAZEL_VERSION 0.11.0
WORKDIR /
RUN mkdir /bazel && \
    cd /bazel && \
    curl -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

CMD ["/bin/bash"]
```

```shell
docker build --pull -t $USER/tensorflow-serving-devel -f Dockerfile.tf-serving . 
```

启动docker并进入

```shell
docker run -idt -v /home/disk1/tf_space:/home/work/data $USER/tensorflow-serving-devel /bin/bash
```

#### 编译server

```shell
root@cb189256755# cd /home/work/data/serving/serving_1.4/
root@0cb189256755:/home/work/data/serving/serving_1.4# git clone -b r1.4 --recurse-submodules https://github.com/tensorflow/serving
root@0cb189256755:/home/work/data/serving/serving_1.4# cd serving/tensorflow
root@0cb189256755:/home/work/data/serving/serving_1.4/serving/tensorflow# ./configure
root@0cb189256755:/home/work/data/serving/serving_1.4/serving# cd ..
root@0cb189256755:/home/work/data/serving/serving_1.4/serving# bazel build -c opt tensorflow_serving/model_servers:tensorflow_model_server
```

这样，生成的```bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server```就是我们想要的啦

#### 编译examples

编译一下tf-serving的example:

```shell
root@0cb189256755:/home/work/data/serving/serving_1.4/serving# bazel build -c opt tensorflow_serving/example/...
```

##### mnist example

训练并export一个模型

```
root@0cb189256755:/home/work/data/serving/serving_1.4/serving# rm /tmp/mnist_model/ -rf
root@0cb189256755:/home/work/data/serving/serving_1.4/serving# bazel-bin/tensorflow_serving/example/mnist_saved_model /tmp/mnist_model
Training model...
Extracting /tmp/train-images-idx3-ubyte.gz
Extracting /tmp/train-labels-idx1-ubyte.gz
Extracting /tmp/t10k-images-idx3-ubyte.gz
Extracting /tmp/t10k-labels-idx1-ubyte.gz
2018-06-03 10:29:58.144101: I external/org_tensorflow/tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
training accuracy 0.9092
Done training!
Exporting trained model to /tmp/mnist_model/1
Done exporting!
```

看到```/tmp/mnist_model```下面有一个文件夹1，就代表version，下面有两部分：

+ saved_model.pb： 序列化后的 tensorflow::SavedModel. It includes one or more graph definitions of the model, as well as metadata of the model such as signatures.
+ variables: files that hold the serialized variables of the graphs.

启动server

```shell
root@0cb189256755:/home/work/data/serving/serving_1.4/serving# nohup bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=mnist --model_base_path=/tmp/mnist_model/ &
```

启动client

```shell
root@0cb189256755:/home/work/data/serving/serving_1.4/serving# bazel-bin/tensorflow_serving/example/mnist_client --num_tests=1000 --server=localhost:9000
Extracting /tmp/train-images-idx3-ubyte.gz
Extracting /tmp/train-labels-idx1-ubyte.gz
Extracting /tmp/t10k-images-idx3-ubyte.gz
Extracting /tmp/t10k-labels-idx1-ubyte.gz
........................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................................
Inference error rate: 10.4%
```

##### inception example

export一个训练好的模型并存储到```/tmp/inception-export```

```shell
root@0cb189256755:/home/work/data/serving/serving_1.4/serving# curl -O http://download.tensorflow.org/models/image/imagenet/inception-v3-2016-03-01.tar.gz
root@0cb189256755:/home/work/data/serving/serving_1.4/serving# tar xzf inception-v3-2016-03-01.tar.gz
root@0cb189256755:/home/work/data/serving/serving_1.4/serving# ls inception-v3
README.txt  checkpoint  model.ckpt-157585
root@0cb189256755:/home/work/data/serving/serving_1.4/serving# bazel-bin/tensorflow_serving/example/inception_saved_model --checkpoint_dir=inception-v3 --output_dir=/tmp/inception-export
Successfully loaded model from inception-v3/model.ckpt-157585 at step=157585.
Successfully exported model to /tmp/inception-export
```

启动server

```shell
root@0cb189256755:/home/work/data/serving/serving_1.4/serving# nohup bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=inception --model_base_path=/tmp/inception-export &> inception_log &
```

启动client：

```shell
root@0cb189256755:/home/work/data/serving/serving_1.4/serving# bazel-bin/tensorflow_serving/example/inception_client --server=localhost:9000 --image=persian_cat_image.jpg   
outputs {
  key: "classes"
  value {
    dtype: DT_STRING
    tensor_shape {
      dim {
        size: 1
      }
      dim {
        size: 5
      }
    }
    string_val: "Persian cat"
    string_val: "lynx, catamount"
    string_val: "Egyptian cat"
    string_val: "tabby, tabby cat"
    string_val: "Angora, Angora rabbit"
  }
}
outputs {
  key: "scores"
  value {
    dtype: DT_FLOAT
    tensor_shape {
      dim {
        size: 1
      }
      dim {
        size: 5
      }
    }
    float_val: 9.48267459869
    float_val: 3.10385608673
    float_val: 2.89405298233
    float_val: 2.83001184464
    float_val: 2.81639647484
  }
}
```

大功告成咯~~

#### commit container

```shell
docker commit 0cb189256755 root/tf_serving_1.4
```

然后就可以看到

```shell
docker images
REPOSITORY                                                        TAG                 IMAGE ID            CREATED             VIRTUAL SIZE
root/tf_serving_1.4                                               latest              e6eae34c8754        41 seconds ago      3.014 GB
```

#### 在2个容器间进行访问

然后启动两个容器：

```shell
## 谨慎使用，删除现有所有容器
docker ps -aq| xargs docker stop | xargs docker rm

docker run -dti \
        -p 9001:9000 \
        --name inception-serving \
        root/tf_serving_1.4 \
        /bin/bash


docker run -dti \
        --name client \
        --link inception-serving:serving \
        -v /home/disk1/tf_space:/home/work/data \
        root/tf_serving_1.4 \
        /bin/bash
```

启动客户端，并定义容器link，允许在容器内部通过“serving”别名来访问“inception-serving”容器

此时，

```shell
 docker ps   
CONTAINER ID        IMAGE                           COMMAND             CREATED             STATUS              PORTS                    NAMES
910de959f1ae        root/tf_serving_1.4             "/bin/bash"         41 seconds ago      Up 40 seconds                                client
3e3134158f9c        root/tf_serving_1.4             "/bin/bash"         41 seconds ago      Up 40 seconds       0.0.0.0:9001->9000/tcp   inception-serving
```

进入server ```3e3134158f9c```：

```shell
root@3e3134158f9c:~# cd /home/work/data/serving/serving_1.4/serving/
root@3e3134158f9c:/home/work/data/serving/serving_1.4/serving# nohup bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=inception --model_base_path=/tmp/inception-export &> inception_log &
```

进入client ```910de959f1ae```，注意，这里的```--server=serving:9000```，serving就是刚刚--link取的别名~

```shell
root@910de959f1ae:~# cd /home/work/data/serving/serving_1.4/serving/
root@910de959f1ae:/home/work/data/serving/serving_1.4/serving# bazel-bin/tensorflow_serving/example/inception_client --server=serving:9000 --image=persian_cat_image.jpg 
```

## k8s

