---
layout: post
category: "other"
title: "docker常用命令"
tags: [docker常用命令,]
---

目录

<!-- TOC -->

- [install docker-enter](#install-docker-enter)
- [docker run](#docker-run)
- [docker images](#docker-images)
- [docker ps](#docker-ps)
- [docker-enter](#docker-enter)
- [docker pull](#docker-pull)
- [docker commit](#docker-commit)
- [docker push](#docker-push)
- [docker save](#docker-save)
- [docker load](#docker-load)
- [docker export](#docker-export)
- [docker import](#docker-import)
- [docker exec](#docker-exec)
- [注意](#注意)

<!-- /TOC -->


## install docker-enter

参考[使用nsenter进入Docker容器](http://blog.csdn.net/fenglailea/article/details/44900401)
```
docker pull jpetazzo/nsenter
docker run --rm -v /usr/bin:/target jpetazzo/nsenter
docker run --rm -v /usr/local/bin:/target jpetazzo/nsenter

wget -P ~ https://github.com/yeasy/docker_practice/raw/master/_local/.bashrc_docker --no-check-certificate;
echo "[ -f ~/.bashrc_docker ] && . ~/.bashrc_docker" >> ~/.bashrc; source ~/.bashrc
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
registry.baidu.com/daiwenkai/deep-learning-factory   1.0.2               390b21e493af        7 weeks ago         2.174 GB
tensorflow/tensorflow                                latest              05a46e5af4d3        8 weeks ago         1.028 GB
registry.baidu.com/public/centos6u3                  1.0.1               9a6d077da3a3        23 months ago       1.363 GB
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

## docker-enter

```
docker-enter ffd32a5b82f7
```

## docker pull

```
docker pull ubuntu
```

## docker commit

```
docker commit -m "1.0.1 version" ffd32a5b82f7 registry.baidu.com/daiwenkai/scrapy-framework:1.0.1
```

## docker push

```
docker push registry.baidu.com/daiwenkai/scrapy-framework:1.0.1
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

## docker exec

不用交互式shell，直接执行命令。（**注意：命令不能加""。。。另外，好像也不能有&&之类的[可以后续再探索探索。。。]**）

其中：

+ -d :分离模式: 在后台运行
+ -i :即使没有附加也保持STDIN 打开
+ -t :分配一个伪终端

```
cat /home/disk0/daiwk_img_data/demo_scrapy/lp_mining/lp_spider/containerid | xargs -i docker exec -dt {} /bin/bash /home/data/demo_scrap
y/lp_mining/lp_spider/run_all_images.sh 

cat /home/disk0/daiwk_img_data/demo_scrapy/lp_mining/lp_spider/containerid | xargs -i docker exec {} tail /home/data/demo_scrapy/lp_min
ing/lp_spider/crawl.images8.log

```

## 注意

+ 在win7安装时，要用docker toolbox([https://download.docker.com/win/stable/DockerToolbox.exe](https://download.docker.com/win/stable/DockerToolbox.exe))

+ win10以上时，用docker-ce([https://store.docker.com/editions/community/docker-ce-desktop-windows?tab=description](https://store.docker.com/editions/community/docker-ce-desktop-windows?tab=description))
