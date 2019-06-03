---
layout: post
category: "other"
title: "docker常用命令"
tags: [docker常用命令,]
---

目录

<!-- TOC -->

- [docker进入容器方法汇总](#docker%E8%BF%9B%E5%85%A5%E5%AE%B9%E5%99%A8%E6%96%B9%E6%B3%95%E6%B1%87%E6%80%BB)
  - [docker exec](#docker-exec)
  - [install docker-enter](#install-docker-enter)
- [docker run](#docker-run)
- [docker images](#docker-images)
- [docker ps](#docker-ps)
- [docker pull](#docker-pull)
- [docker commit](#docker-commit)
- [docker push](#docker-push)
- [docker save](#docker-save)
- [docker load](#docker-load)
- [docker export](#docker-export)
- [docker import](#docker-import)
- [注意](#%E6%B3%A8%E6%84%8F)
- [一些疑难杂症](#%E4%B8%80%E4%BA%9B%E7%96%91%E9%9A%BE%E6%9D%82%E7%97%87)

<!-- /TOC -->

## docker进入容器方法汇总

[https://blog.csdn.net/sqzhao/article/details/71307518](https://blog.csdn.net/sqzhao/article/details/71307518)

### docker exec

不用交互式shell，直接执行命令。（**注意：命令不能加""。。。另外，好像也不能有&&之类的[可以后续再探索探索。。。]**）

其中：

+ -d :分离模式: 在后台运行
+ -i :即使没有附加也保持STDIN 打开
+ -t :分配一个伪终端

```shell
cat /home/disk0/daiwk_img_data/demo_scrapy/lp_mining/lp_spider/containerid | xargs -i docker exec -dt {} /bin/bash /home/data/demo_scrap
y/lp_mining/lp_spider/run_all_images.sh 

cat /home/disk0/daiwk_img_data/demo_scrapy/lp_mining/lp_spider/containerid | xargs -i docker exec {} tail /home/data/demo_scrapy/lp_min
ing/lp_spider/crawl.images8.log

```

当然，也可以交互式啦~~~！！

```shell
 docker ps
CONTAINER ID        IMAGE                                               COMMAND             CREATED             STATUS              PORTS                    NAMES
e56da0385825        tensorflow/tensorflow:latest   "/bin/bash"         3 months ago        Up 3 months         6006/tcp, 8888/tcp       boring_gates
```

然后直接如下就行了，即使退出了，再重进也是一样的啦

```shell
[work@myhost ~] docker exec -it e56da0385825 /bin/bash
root@e56da0385825:/notebooks# ll
total 416
drwxr-xr-x  2 root root   4096 Oct 11 06:40 ./
drwxr-xr-x 22 root root   4096 Jul  3 12:43 ../
-rw-rw-r--  1 root root  25023 Apr 28 00:37 1_hello_tensorflow.ipynb
-rw-rw-r--  1 root root 164559 Apr 28 00:37 2_getting_started.ipynb
-rw-rw-r--  1 root root 209951 Apr 28 00:37 3_mnist_from_scratch.ipynb
-rw-rw-r--  1 root root    119 Apr 28 00:37 BUILD
-rw-rw-r--  1 root root    586 Apr 28 00:37 LICENSE
-rw-r--r--  1 root root      0 Oct 11 06:40 x
root@e56da0385825:/notebooks# exit
exit
[work@myhost ~] docker exec -it e56da0385825 /bin/bash
root@e56da0385825:/notebooks# ll
total 416
drwxr-xr-x  2 root root   4096 Oct 11 06:40 ./
drwxr-xr-x 22 root root   4096 Jul  3 12:43 ../
-rw-rw-r--  1 root root  25023 Apr 28 00:37 1_hello_tensorflow.ipynb
-rw-rw-r--  1 root root 164559 Apr 28 00:37 2_getting_started.ipynb
-rw-rw-r--  1 root root 209951 Apr 28 00:37 3_mnist_from_scratch.ipynb
-rw-rw-r--  1 root root    119 Apr 28 00:37 BUILD
-rw-rw-r--  1 root root    586 Apr 28 00:37 LICENSE
-rw-r--r--  1 root root      0 Oct 11 06:40 x
root@e56da0385825:/notebooks# 
```

### install docker-enter

注：mac似乎不能用此方法，因为那个bin是在linux下编的

参考[使用nsenter进入Docker容器](http://blog.csdn.net/fenglailea/article/details/44900401)

```
docker pull jpetazzo/nsenter
docker run --rm -v /usr/bin:/target jpetazzo/nsenter
docker run --rm -v /usr/local/bin:/target jpetazzo/nsenter

wget -P ~ https://raw.githubusercontent.com/yeasy/docker_practice/docker-legacy/_local/.bashrc_docker --no-check-certificate;

echo "[ -f ~/.bashrc_docker ] && . ~/.bashrc_docker" >> ~/.bashrc; source ~/.bashrc
```

使用时，直接找到对应的containerid，然后：

```
docker-enter ffd32a5b82f7
```


## docker run

```
docker run -idt -v /home/work/daiwenkai:/home/data 390b21e493af /bin/bash
```

## docker images

```

docker images
REPOSITORY                                           TAG                 IMAGE ID            CREATED             VIRTUAL SIZE
ubuntu                            ru                   latest              f5bb94a8fac4        11 days ago         117.3 MB
xxx.baidu.com/daiwenkai/deep-learning-factory   1.0.2               390b21e493af        7 weeks ago         2.174 GB
tensorflow/tensorflow                                latest              05a46e5af4d3        8 weeks ago         1.028 GB
xxx.baidu.com/public/centos6u3                  1.0.1               9a6d077da3a3        23 months ago       1.363 GB
```

## docker ps

```
docker ps
CONTAINER ID        IMAGE                 COMMAND             CREATED              STATUS              PORTS                NAMES
ffd32a5b82f7        ubuntu:latest         "/bin/bash"         About a minute ago   Up About a minute                        loving_torvalds     
166ab859e45c        390b21e493af:latest   "/bin/bash"         6 weeks ago          Up 6 weeks          6006/tcp, 8888/tcp   lonely_bartik 
```

kill并rm所有container

```
docker ps -aq | xargs docker rm -f 
```

## docker pull

```
docker pull ubuntu
```

## docker commit

```
docker commit -m "1.0.1 version" ffd32a5b82f7 xxx.baidu.com/daiwenkai/scrapy-framework:1.0.1
```

## docker push

```
docker push xxx.baidu.com/daiwenkai/scrapy-framework:1.0.1
```

## docker save

导出image

```
docker save -o scrapy-framework.docker.tar 8ce4bbc775d2
```

## docker load

导入image

```
docker load -i scrapy-framework.docker.tar 
```

## docker export

导出container

```
docker export ffd32a5b82f7 > scrapy-framework.1.0.1.tar    
```

## docker import

导入container对应的image，启动时，和之前的container一样

```
cat scrapy-framework.tar | docker import - daiwk/scrapy-framework  
cat deep-learning-factory.1.0.3.tar | docker import - test/deep-learning-factory  
```

## 注意

+ 在win7安装时，要用docker toolbox([https://download.docker.com/win/stable/DockerToolbox.exe](https://download.docker.com/win/stable/DockerToolbox.exe))

+ win10以上时，用docker-ce([https://store.docker.com/editions/community/docker-ce-desktop-windows?tab=description](https://store.docker.com/editions/community/docker-ce-desktop-windows?tab=description))


## 一些疑难杂症

发现有的container没法stop/kill/rm的时候，例如

```shell
docker rm -f 316f2eca7a3b
Error response from daemon: Could not kill running container, cannot remove - [2] Container does not exist: container destroyed
Error: failed to remove containers: [316f2eca7a3b]
```

尝试找到docker的daemon进程，并kill，然后重启：

```shell
ps aux | grep docker
root      4157  0.0  0.0 105352   832 pts/3    S+   13:26   0:00 grep docker
root     32816  0.0  0.0 1562760 42500 ?       Sl   Jun02   1:36 /usr/bin/docker -d -b docker0 -H tcp://0.0.0.0:2375 -H unix:///var/run/docker.sock --insecure-registry registry.xxxxx.com -g /home/work/docker
kill -9 32816
 nohup /usr/bin/docker -d -b docker0 -H tcp://0.0.0.0:2375 -H unix:///var/run/docker.sock --insecure-registry registry.xxxxx.com -g /home/work/docker &
```

这个时候就发现container的状态已经是```exited (137) 7 seconds ago```了，再去rm，就可以啦
