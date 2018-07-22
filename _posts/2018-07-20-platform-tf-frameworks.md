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
    - [安装kubectl](#%E5%AE%89%E8%A3%85kubectl)
    - [安装minikube(local安装)](#%E5%AE%89%E8%A3%85minikubelocal%08%E5%AE%89%E8%A3%85)
    - [启动minikube](#%E5%90%AF%E5%8A%A8minikube)
    - [从bootstrapper安装kubeflow](#%E4%BB%8Ebootstrapper%E5%AE%89%E8%A3%85kubeflow)
- [tf on k8s](#tf-on-k8s)
- [tf on marathon](#tf-on-marathon)
- [tf on hadoop](#tf-on-hadoop)
- [tf on spark](#tf-on-spark)

<!-- /TOC -->


[https://github.com/tensorflow/ecosystem](https://github.com/tensorflow/ecosystem)


## kubeflow

[https://github.com/kubeflow/kubeflow](https://github.com/kubeflow/kubeflow)

### 安装

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

