---
layout: post
category: "platform"
title: "kubernetes"
tags: [kubernetes, k8s ]
---

目录

<!-- TOC -->

- [背景](#%E8%83%8C%E6%99%AF)
    - [物理机时代](#%E7%89%A9%E7%90%86%E6%9C%BA%E6%97%B6%E4%BB%A3)
    - [虚拟机时代](#%E8%99%9A%E6%8B%9F%E6%9C%BA%E6%97%B6%E4%BB%A3)
    - [前容器时代](#%E5%89%8D%E5%AE%B9%E5%99%A8%E6%97%B6%E4%BB%A3)
    - [后容器时代](#%E5%90%8E%E5%AE%B9%E5%99%A8%E6%97%B6%E4%BB%A3)
    - [k8s起源](#k8s%E8%B5%B7%E6%BA%90)
        - [borg](#borg)
        - [omega](#omega)
        - [mesos](#mesos)
        - [nomad](#nomad)
    - [CNCF基金会](#cncf%E5%9F%BA%E9%87%91%E4%BC%9A)
    - [k8s现状](#k8s%E7%8E%B0%E7%8A%B6)
    - [例子](#%E4%BE%8B%E5%AD%90)
- [架构&组件](#%E6%9E%B6%E6%9E%84%E7%BB%84%E4%BB%B6)
    - [k8s & docker](#k8s--docker)
    - [k8s & Node](#k8s--node)
    - [架构](#%E6%9E%B6%E6%9E%84)
- [基础概念&术语](#%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5%E6%9C%AF%E8%AF%AD)
    - [Pod](#pod)
    - [ReplicationSet](#replicationset)
    - [Label](#label)
    - [deployment](#deployment)
    - [rolling update](#rolling-update)
    - [StatefulSet](#statefulset)
    - [daemonSet](#daemonset)
    - [Jobs](#jobs)
    - [service](#service)
    - [headless service](#headless-service)
    - [k8s service对外提供访问](#k8s-service%E5%AF%B9%E5%A4%96%E6%8F%90%E4%BE%9B%E8%AE%BF%E9%97%AE)
    - [DNS](#dns)
    - [native network model](#native-network-model)
    - [Volumes](#volumes)
- [inf k8s](#inf-k8s)
- [k8s示例](#k8s%E7%A4%BA%E4%BE%8B)

<!-- /TOC -->

[https://github.com/kubernetes/kubernetes](https://github.com/kubernetes/kubernetes)

百度前员工刘杰写的《分布式系统原理介绍》【[https://daiwk.github.io/assets/分布式系统原理介绍.pdf](https://daiwk.github.io/assets/分布式系统原理介绍.pdf)】。

参考[http://kubernetes.kansea.com/docs/whatisk8s/](http://kubernetes.kansea.com/docs/whatisk8s/)

## 背景

### 物理机时代

### 虚拟机时代

openstack

### 前容器时代

docker，基于LXC为基础构建的容器引擎，通过namespace和cgroup实现**资源**隔离和调配。将os和应用捆绑，使得应用系统环境标准化、集装箱化。主要问题：缺乏完整的调度部署管理能力。

### 后容器时代

+ 原生docker swarm
+ google的kubernetes(2015年)：容器集群管理系统。可以管理跨机器容器化。
+ apache的mesos。优势在于离线集群任务。

### k8s起源

2004年google开始使用容器，06年cgroup
内部集群资源管理平台borg和omega
k8s起源于borg，参考了omega的经验和教训。borg和omega的创始人都加入k8s。

[https://daiwk.github.io/assets/Borg_Omega_Kubernetes.pdf](https://daiwk.github.io/assets/Borg_Omega_Kubernetes.pdf)

#### borg

[https://daiwk.github.io/assets/Borg.pdf](https://daiwk.github.io/assets/Borg.pdf)

#### omega

[https://daiwk.github.io/assets/Omega.pdf](https://daiwk.github.io/assets/Omega.pdf)

#### mesos

#### nomad

[https://daiwk.github.io/assets/Nomad.pdf](https://daiwk.github.io/assets/Nomad.pdf)

### CNCF基金会

Cloud Native Computing Foundation ==>生态绑定、法律保护、推广、培训等。

### k8s现状

5w+commits，2.5w stars

### 例子

通过kubectl提交，应用描述文件（json/yaml）

一个app下可以有多个实例。

使用label和selector进行deployment和service的对应，nodeport虚拟端口nodeport，对外暴露的端口targetport

## 架构&组件

### k8s & docker

k8s会调度相应的app到对应的docker host上去运行

### k8s & Node

调度和pack各种xxx到不同的nodes上

### 架构

可以参考[https://www.kubernetes.org.cn/kubernetes%E8%AE%BE%E8%AE%A1%E6%9E%B6%E6%9E%84](https://www.kubernetes.org.cn/kubernetes%E8%AE%BE%E8%AE%A1%E6%9E%B6%E6%9E%84)

<html>
<br/>
<img src='../assets/k8s-architecture.png' style='max-height: 300px'/>
<br/>
</html>

单集群可以一次5000台机器上线

有一个中心化的api server，然后把状态存储在etcd中
主节点有scheduler和controller mgr，worker节点有kubelet和service proxy。

api server中有所有资源的增删改查的接口，还有authorization，可以按namespace来划分权限。

scheduler中把定义的pod按照策略调度到相应节点

controller manager和apiserver通信，获取集群的特定信息，然后做出响应的反馈动作。由多个controller组成

etcd是分布式k-v你在什么地方，所有集群数据存放在etcd中，实现组件的无状态化。可以通过watch的方式监听变化，并触发相应动作。

worker节点:

+ kubelet：是节点的agent，接收描述的manifest并处理
+ kubeproxy: 网络agent，部署在各个node，简单的tcp/udp转发，简单的round-robin负载均衡。

## 基础概念&术语

### Pod
pod是若干**相关容器**的组合。pod包含的容器运行在同一台宿主机上，它们使用相同的pid/network/ipc/uts命名空间/ip地址和端口，相互之前能通过localhost来发现和通信，还可以共享一块存储volume空间。其实是容器的更高层次的抽象。

### ReplicationSet
控制管理pod副本。确保任何时候k8s集群中有足够v几个的pod副本在运行。如果少于，会自动启动。

### Label
区分pod/service/replication/controller的k/v对。每个api对象可以有多个label，但每个label的k只能对应一个v。

### deployment
只描述集群的期望状态

### rolling update
升级时，旧实例下一下，上一个新实例这样。但如果是大更新，只能停机更新，旧的全下了再上新的。
参数：
maxSurge:不容许实例减少，且集群资源充裕
maxUnavaliable:集群资源照张，可以先缩容，再更新

### StatefulSet

部署带持久化数据的服务
+ identity严格区分, uniq ordinal value
+ index min会被选举为master
+ nfs是底层存储

搭建分布式db集群
+ storage class用于vpc申请
+ headless service用于维护db的endpoints
+ statefulset

### daemonSet
每个host部署一个服务，也有rolling update

### Jobs

parallelism：最大的共有并行数
completions: 完成了多少个job就算完成了

### service

service是真实应用服务的抽象，定义了pod的逻辑集合和访问这个pod集合的策略。
将代理pod对外表现为一个单一的访问接口，外部不需要了解后端pod是如何运行的。

实现原理：kube-proxy维护一个iptables，会去watch apiserver，然后更新这个table，有dns解析功能

### headless service

不提供round-robin
显式返回所有endpoints
可以自定义策略（如选主）
最佳实践：cacendera的去中心化选主

### k8s service对外提供访问

+ nodeport
+ loadbalancer
  + blvxxxxx: 类似bfe
  +ingress: 7层的routing，nginx的实现(可以访问内部节点，然后开个外网ip，让其他k8s也能访问它内部的节点)

### DNS

skyDNS->KubeDNS。可以对特定的domain自定义上游nameserver。

优先级：
+ kubedns
+ 特定domain的自定义上游namesever
+ 默认的upstream nameserver

### native network model

每个pod有一个ip，需要容器网络

### Volumes

可以将存储在pod上进行挂载，可以直接挂载基于fuse共享存储NFS等。

## inf k8s

docker .baidu.com
registry.baidu.com
dockerhub.baidubce.com

微服务
自动伸缩
使用supervisor,如果主进程被Kill了，会把这个信号量传给每个子进程，避免孤儿进程
节点亲和&应用亲和：例如混布cpu+gpu（后来，1.6的k8s原生支持gpu了）

## k8s示例

使用minikube:

[https://qii404.me/2018/01/06/minukube.html](https://qii404.me/2018/01/06/minukube.html)
