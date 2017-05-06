---
layout: post
category: "other"
title: "docker常用命令"
tags: [docker常用命令,]
---


## docker run

```
docker run -idt xxx /bin/bash
```

## docker images

```

docker images
REPOSITORY                                           TAG                 IMAGE ID            CREATED             VIRTUAL SIZE
ubuntu                                               latest              f5bb94a8fac4        11 days ago         117.3 MB
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