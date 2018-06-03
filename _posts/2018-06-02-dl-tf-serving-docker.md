---
layout: post
category: "dl"
title: "tf serving docker+k8s"
tags: [tf serving, docker, k8s ]
---

目录

<!-- TOC -->

- [tf-serving](#tf-serving)
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

