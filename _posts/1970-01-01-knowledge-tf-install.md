---
layout: post
category: "knowledge"
title: "tf安装"
tags: [tf安装, tensorflow 安装, ]
---

目录

<!-- TOC -->

- [装备工作](#%E8%A3%85%E5%A4%87%E5%B7%A5%E4%BD%9C)
  - [bazel](#bazel)
  - [jdk1.8](#jdk18)
- [源码安装](#%E6%BA%90%E7%A0%81%E5%AE%89%E8%A3%85)
  - [clone](#clone)
  - [configure](#configure)
  - [生成pip_package](#%E7%94%9F%E6%88%90pippackage)
    - [仅cpu](#%E4%BB%85cpu)
    - [gpu](#gpu)
  - [生成whl](#%E7%94%9F%E6%88%90whl)
  - [安装c++库](#%E5%AE%89%E8%A3%85c%E5%BA%93)
    - [手动下载依赖库](#%E6%89%8B%E5%8A%A8%E4%B8%8B%E8%BD%BD%E4%BE%9D%E8%B5%96%E5%BA%93)
    - [重新configure](#%E9%87%8D%E6%96%B0configure)
    - [编译cc的so](#%E7%BC%96%E8%AF%91cc%E7%9A%84so)
    - [拷贝所需头文件](#%E6%8B%B7%E8%B4%9D%E6%89%80%E9%9C%80%E5%A4%B4%E6%96%87%E4%BB%B6)
    - [拷贝所需lib文件](#%E6%8B%B7%E8%B4%9D%E6%89%80%E9%9C%80lib%E6%96%87%E4%BB%B6)
- [源码安装tf-serving](#%E6%BA%90%E7%A0%81%E5%AE%89%E8%A3%85tf-serving)

<!-- /TOC -->

## 装备工作

### bazel

### jdk1.8

## 源码安装

### clone

```shell
git clone https://github.com/tensorflow/tensorflow 
```

### configure

```shell
./configure
```

注意，这里可以配置默认python路径

### 生成pip_package

#### 仅cpu

```shell
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
```

#### gpu

```shell
bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
```

### 生成whl

```shell
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
```

这样，就在```/tmp/tensorflow_pkg```生成了```tensorflow-xxxx-py2-none-any.whl```

### 安装c++库

参考：

[https://github.com/hemajun815/tutorial/blob/master/tensorflow/compilling-tensorflow-source-code-into-C++-library-file.md](https://github.com/hemajun815/tutorial/blob/master/tensorflow/compilling-tensorflow-source-code-into-C++-library-file.md)

#### 手动下载依赖库

进入目录：

```shell
cd tensorflow/contrib/makefile
```

执行文件：

```shell
sh -x ./build_all_linux.sh
cd -
```

注意：

必要时修改```download_dependencies.sh```文件：

+ curl需要支持https(可以升级)，比如，我们发现jumbo的curl是符合预期的，那可以在```build_all_linux.sh```一开头就加上```export PATH=~/.jumbo/bin/:$PATH```
+ wget加上```--no-check-certificate```参数，还有```--secure-protocol=TLSv1.2 ```参数，当然如果版本不够高就升级wget。如果不好升级可以做如下改动：

把

```shell
wget -P "${tempdir}" "${url}"
```

改成

```shell
curl -Ls "${url}" > "${tempdir}"/xxx
```

#### 重新configure

```shell
cd tensorflow
./configure
```

#### 编译cc的so

```shell
cd tensorflow
bazel build :libtensorflow_cc.so
```

产出在```bazel-bin/tensorflow/libtensorflow_cc.so```

#### 拷贝所需头文件

```shell
tensorflow_include=./tensorflow_include
mkdir $tensorflow_include
cp -r tensorflow/contrib/makefile/downloads/eigen/Eigen $tensorflow_include/
cp -r tensorflow/contrib/makefile/downloads/eigen/unsupported $tensorflow_include/
cp -r tensorflow/contrib/makefile/gen/protobuf/include/google $tensorflow_include/
cp tensorflow/contrib/makefile/downloads/nsync/public/* $tensorflow_include/
cp -r bazel-genfiles/tensorflow $tensorflow_include/
cp -r tensorflow/cc $tensorflow_include/tensorflow
cp -r tensorflow/core $tensorflow_include/tensorflow
mkdir $tensorflow_include/third_party
cp -r third_party/eigen3 $tensorflow_include/third_party/
```

#### 拷贝所需lib文件

```shell
tensorflow_lib=./tensorflow_lib
mkdir $tensorflow_lib
cp bazel-bin/tensorflow/libtensorflow_*.so $tensorflow_lib
```

## 源码安装tf-serving

github地址：[https://github.com/tensorflow/serving](https://github.com/tensorflow/serving)

如果要clone某个版本，可以直接

```shell
git clone -b r1.12 https://github.com/tensorflow/serving.git
```

如果在docker内安装：

```shell
./tools/run_in_docker.sh  bazel build tensorflow_serving/model_servers:tensorflow_model_server
```

如果在非docker内安装：

```shell
bazel build tensorflow_serving/model_servers:tensorflow_model_server
```

然后我们就发现产出了这么一个bin文件：```./bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server```。

