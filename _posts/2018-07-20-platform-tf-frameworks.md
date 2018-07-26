---
layout: post
category: "platform"
title: "tf与开源框架的集成"
tags: [kubeflow, ]
---

目录

<!-- TOC -->

- [kubeflow](#kubeflow)
  - [安装](#%E5%AE%89%E8%A3%85)
    - [安装kubernets](#%E5%AE%89%E8%A3%85kubernets)
    - [从bootstrapper安装kubeflow](#%E4%BB%8Ebootstrapper%E5%AE%89%E8%A3%85kubeflow)
- [tf on k8s](#tf-on-k8s)
- [tf on marathon](#tf-on-marathon)
- [tf on hadoop](#tf-on-hadoop)
- [tf on spark](#tf-on-spark)
- [附录](#%E9%99%84%E5%BD%95)
  - [使用minikube(没成功过…)](#%E4%BD%BF%E7%94%A8minikube%E6%B2%A1%E6%88%90%E5%8A%9F%E8%BF%87%E2%80%A6)
    - [安装kubectl](#%E5%AE%89%E8%A3%85kubectl)
    - [安装minikube(local安装)](#%E5%AE%89%E8%A3%85minikubelocal%08%E5%AE%89%E8%A3%85)
    - [启动minikube](#%E5%90%AF%E5%8A%A8minikube)

<!-- /TOC -->


[https://github.com/tensorflow/ecosystem](https://github.com/tensorflow/ecosystem)


## kubeflow

[https://github.com/kubeflow/kubeflow](https://github.com/kubeflow/kubeflow)

### 安装

#### 安装kubernets

参考[http://blog.sina.com.cn/s/blog_48c95a190102wqpq.html](http://blog.sina.com.cn/s/blog_48c95a190102wqpq.html)

首先安装go：直接去官网[https://www.golangtc.com/download](https://www.golangtc.com/download)搞一个下来，然后解压，然后设置一下```export GOROOT=xxxxx```，再把bin目录下的go*丢到/usr/local/bin下面就行了。

然后从[https://github.com/kubernetes/kubernetes/releases](https://github.com/kubernetes/kubernetes/releases)这里找一个版本的源码进行下载，例如kubernetes-1.9.7.tar.gz这个版本，然后解压

然后进入解压后的目录，直接make就行了，这样会自己把最必需的kubectl、kubelet、kubeadm放到/usr/local/bin下，而且在_output目录下生成kube-apiserver、kube-proxy、kube-controller-manager、kube-scheduler等各种bin。

然后把etcd和flannel搞下来

```shell
wget https://github.com/coreos/etcd/releases/download/v2.3.8/etcd-v2.3.8-linux-amd64.tar.gz
wget https://github.com/coreos/flannel/releases/download/v0.6.2/flannel-v0.6.2-linux-amd64.tar.gz
```

放到/usr/local/bin下面去，然后修改 /etc/sysctl.conf

```shell
net.bridge.bridge-nf-call-ip6tables = 1
net.bridge.bridge-nf-call-iptables = 1
net.bridge.bridge-nf-call-arptables = 1
```

执行 

```shell
sysctl -p
```

```shell
master_hostname=11.11.2.3

nohup etcd --name infra0 --initial-advertise-peer-urls http://${master_hostname}:2380,http://${master_hostname}:7001 --listen-peer-urls http://${master_hostname}:2380,http://${master_hostname}:7001 --listen-client-urls http://${master_hostname}:2379,http://${master_hostname}:4001 --advertise-client-urls http://${master_hostname}:2379,http://${master_hostname}:4001 --initial-cluster-token etcd-cluster --initial-cluster infra0=http://${master_hostname}:2380,infra0=http://${master_hostname}:7001 --data-dir /root/data/etcd/data --initial-cluster-state new &

nohup etcdctl --endpoints=http://${master_hostname}:2379,http://${master_hostname}:4001 mk /coreos.com/network/config '{"Network":"172.17.0.0/16", "SubnetMin": "172.17.1.0", "SubnetMax": "172.17.254.0"}' &

nohup flanneld -etcd-endpoints=http://${master_hostname}:2379,http://${master_hostname}:4001 &
```

#### 从bootstrapper安装kubeflow

```shell
curl -O https://raw.githubusercontent.com/kubeflow/kubeflow/master/bootstrap/bootstrapper.yaml
```

然后

```shell
kubectl create -f bootstrapper.yaml
```

## tf on k8s

## tf on marathon

marathon是基于mesos的

## tf on hadoop

## tf on spark


## 附录

### 使用minikube(没成功过…)

#### 安装kubectl

kubectl即kubernetes的客户端，通过他可以进行类似docker run等容器管理操作

+ ubuntu

```shell
sudo apt-get update && sudo apt-get install -y apt-transport-https
curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo touch /etc/apt/sources.list.d/kubernetes.list
echo "deb http://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee -a /etc/apt/sources.list.d/kubernetes.list
sudo apt-get update
sudo apt-get install -y kubectl
```

+ centos

```shell
cat <<EOF > /etc/yum.repos.d/kubernetes.repo
[kubernetes]
name=Kubernetes
baseurl=https://packages.cloud.google.com/yum/repos/kubernetes-el7-x86_64
enabled=1
gpgcheck=1
repo_gpgcheck=1
gpgkey=https://packages.cloud.google.com/yum/doc/yum-key.gpg https://packages.cloud.google.com/yum/doc/rpm-package-key.gpg
EOF

sudo yum install -y kubectl
```

+ mac

```shell
brew install kubectl
```

#### 安装minikube(local安装)

+ ubuntu/centos

```shell
curl -Lo minikube https://storage.googleapis.com/minikube/releases/v0.28.0/minikube-linux-amd64
chmod +x minikube
sudo mv minikube /usr/local/bin/
```

+ mac

先从官网安装virtualbox [https://www.virtualbox.org/wiki/Downloads](https://www.virtualbox.org/wiki/Downloads)

```shell
curl -Lo minikube https://storage.googleapis.com/minikube/releases/v0.28.0/minikube-darwin-amd64
chmod +x minikube
sudo mv minikube /usr/local/bin/
```

#### 启动minikube

以下参数是kubeflow的最低配置

```shell
minikube start --cpus 4 --memory 8096 --disk-size=40g 
```

如果在linux下，可以指定不要虚拟机：

```shell
minikube start --cpus 4 --memory 8096 --disk-size=40g --vm-driver=none
```

+ mac

如果发现被墙，下不下来…就手动开浏览器下载提示出错的url，例如[ttps://storage.googleapis.com/minikube/iso/minikube-v0.28.0.iso](ttps://storage.googleapis.com/minikube/iso/minikube-v0.28.0.iso)

```
mv git_daiwk/minikube-v0.28.0.iso ~/.minikube/cache/iso/
```

如果kubelet下载超时，如[https://storage.googleapis.com/kubernetes-release/release/v1.10.0/bin/linux/amd64/kubelet](https://storage.googleapis.com/kubernetes-release/release/v1.10.0/bin/linux/amd64/kubelet)和[https://storage.googleapis.com/kubernetes-release/release/v1.10.0/bin/linux/amd64/kubeadm](https://storage.googleapis.com/kubernetes-release/release/v1.10.0/bin/linux/amd64/kubeadm)

然后

```shell
mv kubelet ~/.minikube/cache/v1.10.0/
mv kubeadm ~/.minikube/cache/v1.10.0/
```

+ ubuntu/centos

```shell 
curl --output ./k8s_version_stable.txt https://storage.googleapis.com/kubernetes-release/release/stable.txt # v1.11.1
curl --output ./kubectl "https://storage.googleapis.com/kubernetes-release/release/$(cat /tmp/kubectl_version)/bin/linux/amd64/kubectl"
curl --output ./kubelet "https://storage.googleapis.com/kubernetes-release/release/$(cat /tmp/kubectl_version)/bin/linux/amd64/kubelet"
curl --output ./kubeadm "https://storage.googleapis.com/kubernetes-release/release/$(cat /tmp/kubectl_version)/bin/linux/amd64/kubeadm"
```