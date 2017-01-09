---
layout: post
category: "platform"
title: "paddlepaddle集群版的使用"
tags: [paddlepaddle, 集群]
---

## 0. 内部一键安装

参考
[http://deeplearning.baidu.com/doc/local_compile.html](http://deeplearning.baidu.com/doc/local_compile.html)

```shell
git clone http://gitlab.baidu.com/idl-dl/paddle_internal_release_tools.git
cd paddle_internal_release_tools/idl/paddle/ && sh build.sh [cpu|gpu] [rdma|nonrdma]
source ~/.bashrc
paddle 
```

### 注意：

+ 机器如果能访问github，脚本会自动下载Paddle源码库；机器如果不能访问github，脚本会提示手动下载Paddle源码，并替换Paddle目录。
+ 无需设置安装目录，默认将安装到编译目录下的output目录并导出到环境变量~/.bashrc中，无需sudo权限。
+ 安装不依赖系统任何库，除了glibc相关的系统库
+ 使用自带的python解释器。安装完成后使用，pip等python命令将被重定向到自带python解释器，同时执行pip install $SOMELIB 将会默认安装到自带的python解释器环境中。
+ 编译前可能需要下载COAMKE2工具链，会提示输入svn密码
+ 整体安装脚本将尽量做到不依赖当前系统第三方库，全部采用闭包形式解决编译和安装问题。

### faq:

+ 如何使用Paddle稳定分之进行编译？
[一键编译工具准备源码阶段 采用Submodule的方式引用Paddle开源库，因此只要修改脚本使得在正式编译前， 用git checkout -b $BRANCH， 然后再执行sh build.sh 即可。]
+ 如果不能访问github，如何编译？
[只要手动下载Paddle源码，替换idl/paddle/Paddle目录即可，然后执行sh build.sh]
+ 如何卸载Paddle？
[只要删除~/.bashrc中带有『#PaddlePaddle!@#$%^&*』注释的环境变量设置，然后重新登录shell即可。Paddle一键安装仅仅通过~/.bashrc 几个变量来达到安装的目的。]

## 1. 安装集群二进制包

根据[http://deeplearning.baidu.com/doc/install.html](http://deeplearning.baidu.com/doc/install.html)来搞，

```shell
wget http://deeplearning.baidu.com/resources/releases/platform2_client/paddle_platform_client_ff01840260d5a3607596a7e2b3cd4f705ff75e78.tgz 
wget http://deeplearning.baidu.com/resources/releases/platform2_client/deploy.sh 
sh deploy.sh paddle_platform_client_ff01840260d5a3607596a7e2b3cd4f705ff75e78.tgz ./output
pip install poster
```

deploy.sh 安装脚本将客户端程序安装到./output 目录，它通过修改~/.bashrc达到安装目的。

通过

```shell
rm ./output -fr
```

来进行卸载。

## 2. 更新receivers

客户端会将用户集群训练转发到一个receiver服务器，由receiver服务器分析用户训练配置、准备Paddle二进制镜像、最终将任务提交到物理mpi集群。

一般每个receiver都会保存一组默认的paddle镜像 (获取Receiver后端Paddle核心版本的方法详见[客户端教程FAQ](http://deeplearning.baidu.com/doc/client_tutorial.html) )，支持各种异构mpi集群的运行时环境，最新的receiver支持调度当前所有的mpi异构集群。

```shell
client_path=$(dirname `which cluster_train.sh`)
```

将$client_path/local_config.py替换为最新的：【以
[http://deeplearning.baidu.com/doc/build_private_paddle.html](http://deeplearning.baidu.com/doc/build_private_paddle.html)
为准。】

```shell
yq01-idl-gpu-offline14.yq01.baidu.com:9290
yq01-idl-gpu-offline14.yq01.baidu.com:9291
yq01-idl-gpu-offline14.yq01.baidu.com:9292
yq01-idl-gpu-offline14.yq01.baidu.com:9293
yq01-idl-gpu-offline14.yq01.baidu.com:9294
yq01-idl-gpu-offline14.yq01.baidu.com:9295
yq01-idl-gpu-offline14.yq01.baidu.com:9296
yq01-idl-gpu-offline14.yq01.baidu.com:9297
yq01-idl-gpu-offline14.yq01.baidu.com:9298
yq01-idl-gpu-offline14.yq01.baidu.com:9299
yq01-idl-gpu-offline14.yq01.baidu.com:9390
yq01-idl-gpu-offline14.yq01.baidu.com:9391
yq01-idl-gpu-offline14.yq01.baidu.com:9392
yq01-idl-gpu-offline14.yq01.baidu.com:9393
yq01-idl-gpu-offline14.yq01.baidu.com:9394
yq01-idl-gpu-offline14.yq01.baidu.com:9395
yq01-idl-gpu-offline14.yq01.baidu.com:9396
yq01-idl-gpu-offline14.yq01.baidu.com:9397
yq01-idl-gpu-offline14.yq01.baidu.com:9398
yq01-idl-gpu-offline14.yq01.baidu.com:9399
```

## 3. 使用private版本的bin

因为远端的集群中的paddle是最新的稳定版，如果需要使用trunk版本编译出来的paddle的新特性（如define_by_data_sources2,pyxxxx2之类的），需要额外地往thirdparty中放一个private的paddle进去。参考
[http://deeplearning.baidu.com/doc/build_private_paddle.html](http://deeplearning.baidu.com/doc/build_private_paddle.html)

```shell
git clone http://gitlab.baidu.com/idl-dl/platform2.git 
cd platform2/tools && sh build_private_paddle.sh cpu nonrdma
```

在当前目录下建一个thirdparty目录(记为$thirdparty_dir)

```shell
cp -rf private_output/* $thirdparty_dir
```

修改$thirdparty_dir/before_hook.sh

```shell
function private_script()
{
  local l_thirdparty_dir=$1
  local l_workspace_dir=$2

  # add your script here

}
```

其中，
l_thirdparty_dir 指 $ROOT_WORKSPACE_ROOT/thirdparty/thirdparty

l_workspace_dir 指 $ROOT_WORKSPACE_ROOT/

然后就可以运行了，下面举两个例子：

### sequence_tagging:

run.sh中：

```shell
## run.sh
cp ./*.py ./thirdparty

source ~/.bashrc

####因为字典比较特殊，所以只能用一个num_nodes,而这样会爆内存……所以得改代码…(徐老师说：“这种情况需要事先构建好字典，然后data provider加载”，所以，就需要参考seq2seq的demo去慢慢改了……)
paddle cluster_train \
           --config cluster_conf_dwk.py \
           --time_limit 00:30:00 \
           --submitter daiwenkai \
           --num_nodes 1 \ 
           --job_priority normal \
           --trainer_count 4 \ 
           --num_passes 1 \ 
           --log_period 1000 \
           --dot_period 100 \
           --saving_period 1 \ 
           --where nmg01-idl-dl-cpu-10G_cluster \
           --thirdparty ./thirdparty \
           --job_name daiwenkai_paddle_cluster_demo_v5
```

复制一个网络结构的py为cluster_conf_dwk.py,并进行如下修改：

```python
## cluster_conf_dwk.py:
###新增:
cluster_config(
        fs_name="hdfs://nmg01-mulan-hdfs.dmop.baidu.com:54310",
        fs_ugi="xxxx,xxxx",
        work_dir="/app/ecom/fcr-opt/daiwenkai/paddle/demos/sequence_tagging/"
        )   
###修改：
define_py_data_sources2(
    train_list="./train.list",
    test_list="./test.list",
    module="dataprovider",
    obj="process")
```

修改before_hook.sh中的private_script函数：

```shell
function private_script()
{
  local l_thirdparty_dir=$1
  local l_workspace_dir=$2

  # add your script here
  
  cp $l_thirdparty_dir/*.py $l_workspace_dir/

}
```

集群目录结构如下：

```shell
/app/ecom/fcr-opt/daiwenkai/paddle/demos/sequence_tagging/test/test.txt.gz
/app/ecom/fcr-opt/daiwenkai/paddle/demos/sequence_tagging/train/train.txt.gz
```

### image_classification:

首先要[安装PIL](https://daiwk.github.io/posts/image-install-pil.html)(写一半，完善中)，装完后

```shell
lib_dir=/home/data/mylib/
cp -rf $lib_dir/lib/lib* ./thirdparty
cp -rf $python_path/lib/python2.7/site-packages/PIL ./thirdparty 

```

然后run.sh

```shell
## run.sh
cp ./*.py ./thirdparty

source ~/.bashrc

config=cluster.vgg_16_cifar.py
source ~/.bashrc

paddle cluster_train \
           --config $config \
           --time_limit 00:30:00 \
           --submitter daiwenkai \
           --num_nodes 24 \
           --job_priority normal \
           --trainer_count 4 \ 
           --num_passes 1 \ 
           --log_period 1000 \
           --dot_period 100 \
           --saving_period 1 \ 
           --where nmg01-hpc-off-dmop-cpu-10G_cluster \
           --thirdparty ./thirdparty \
           --job_name daiwenkai_paddle_cluster_demo_image_classificatoin

```

复制一份cluster.vgg_16_cifar.py

```python
###新增:
cluster_config(
        fs_name="hdfs://nmg01-mulan-hdfs.dmop.baidu.com:54310",
        fs_ugi="xxx,xxx",
        work_dir="/app/ecom/fcr-opt/daiwenkai/paddle/demos/image_classification/",

        has_meta_data=True,
        )    
###修改：
meta_path = "./train_data_dir/image_classification/train_meta"
```

由于集群上workspace生成的train.list与本地不一样，所以我们还要直接修改image_provider.py专供集群使用

```python
## image_provider.py
###修改：
@provider(init_hook=hook, min_pool_size=0)
def processData(settings, file_list):
    """ 
    The main function for loading data.
    Load the batch, iterate all the images and labels in this batch.
    file_list: the batch file list. #### in cluster, file_list is the file's name, like ./train_data_dir/train/train_batch_033
    """

    ## cluster version:
#    with open(file_list, 'r') as fdata:
#        print file_list, "filelist, XXXXXXXXX"
#        lines = [line.strip() for line in fdata]
#        random.shuffle(lines)
#        print lines, "lines, XXXXXXXXX"
#        for file_name in lines:
#            print file_name, "filename, XXXXXXXXX"
    with io.open(file_list.strip(), 'rb') as file:
        data = cPickle.load(file)
        indexes = list(range(len(data['images'])))
        if settings.is_train:
            random.shuffle(indexes)
        for i in indexes:
            if settings.use_jpeg == 1:
                img = image_util.decode_jpeg(data['images'][i])
            else:
                img = data['images'][i]
            img_feat = image_util.preprocess_img(
                img, settings.img_mean, settings.img_size,
                settings.is_train, settings.color)
            label = data['labels'][i]
            yield img_feat.astype('float32'), int(label) 

```

修改before_hook.sh中的private_script函数：

```shell
function private_script()
{
  local l_thirdparty_dir=$1
  local l_workspace_dir=$2

  # add your script here
  
  cp $l_thirdparty_dir/*.py $l_workspace_dir/
  cp -rf $l_thirdparty_dir/PIL $l_workspace_dir/
  cp  $l_thirdparty_dir/lib* $l_workspace_dir/

}
```

集群目录结构：

```shell
/app/ecom/fcr-opt/daiwenkai/paddle/demos/image_classification/test/test_batch_00[0－9]
/app/ecom/fcr-opt/daiwenkai/paddle/demos/image_classification/train/train_batch_0[0－49]
/app/ecom/fcr-opt/daiwenkai/paddle/demos/image_classification/train_meta
/app/ecom/fcr-opt/daiwenkai/paddle/demos/image_classification/test_meta
```

其中，xxxbatchxxx和xxxmeta都是通过如下方法得到的

```shell
cd data && sh download_cifar.sh
cd - && sh preprocess.sh
```

###　注意

```shell
#ecom的队列：
nmg01-hpc-off-dmop-cpu-10G_cluster # fcr队列, time_limit<=00:30:00
nmg01-hpc-off-dmop-slow-cpu-10G_cluster # fcr-slow队列, time_limit<=99:59:59

#idl的队列：
nmg01-idl-dl-cpu-10G_cluster
```