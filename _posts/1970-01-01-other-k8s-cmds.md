---
layout: post
category: "other"
title: "k8s常用命令"
tags: [k8s常用命令, kubectl, ]
---

目录

<!-- TOC -->

- [cluster-info](#cluster-info)
- [get](#get)
  - [get deployments](#get-deployments)
  - [get nodes](#get-nodes)
  - [get pods](#get-pods)
    - [get pod detail](#get-pod-detail)
  - [get rc](#get-rc)
  - [get services](#get-services)
    - [get service detail](#get-service-detail)
- [delete](#delete)
  - [delete pod](#delete-pod)
  - [delete service](#delete-service)
  - [delete deployment](#delete-deployment)
- [describe](#describe)
- [logs](#logs)
- [exec](#exec)
- [demo](#demo)
  - [新建一个deployment](#%E6%96%B0%E5%BB%BA%E4%B8%80%E4%B8%AAdeployment)
  - [自动扩容](#%E8%87%AA%E5%8A%A8%E6%89%A9%E5%AE%B9)
  - [暴露端口](#%E6%9A%B4%E9%9C%B2%E7%AB%AF%E5%8F%A3)
  - [带label地查询](#%E5%B8%A6label%E5%9C%B0%E6%9F%A5%E8%AF%A2)
  - [pod的自动恢复](#pod%E7%9A%84%E8%87%AA%E5%8A%A8%E6%81%A2%E5%A4%8D)
  - [pod的自动分配](#pod%E7%9A%84%E8%87%AA%E5%8A%A8%E5%88%86%E9%85%8D)
  - [构建namespace](#%E6%9E%84%E5%BB%BAnamespace)
  - [从yaml文件出发](#%E4%BB%8Eyaml%E6%96%87%E4%BB%B6%E5%87%BA%E5%8F%91)

<!-- /TOC -->

## cluster-info

```shell
kubectl cluster-info
```

显示：

```shell
Kubernetes master is running at https://xxx.xx.xx.xxx:6443
Heapster is running at https://xxx.xx.xx.xxx:6443/api/v1/namespaces/kube-system/services/heapster/proxy
KubeDNS is running at https://xxx.xx.xx.xxx:6443/api/v1/namespaces/kube-system/services/kube-dns:dns/proxy
monitoring-influxdb is running at https://xxx.xx.xx.xxx:6443/api/v1/namespaces/kube-system/services/monitoring-influxdb/proxy

To further debug and diagnose cluster problems, use 'kubectl cluster-info dump'.
```

## get

### get deployments

```shell
kubectl get deployments
```

显示

？？？

### get nodes

```shell
kubectl get nodes -o wide
```

显示

```shell
NAME            STATUS    ROLES     AGE       VERSION   INTERNAL-IP     EXTERNAL-IP   OS-IMAGE                KERNEL-VERSION              CONTAINER-RUNTIME
192.168.0.142   Ready     <none>    15h       v1.11.5   192.168.0.142   <none>        CentOS Linux 7 (Core)   3.10.0-957.1.3.el7.x86_64   docker://18.9.2
```

### get pods

```shell
kubectl get pods -o wide
```

显示

#### get pod detail

```shell
kubectl get pod xxxxx -template={{.status.podIP}}
```

显示

？？？

### get rc

获取replication controllers信息

```shell
kubectl get rc -o wide
```

显示

？？？

### get services

```shell
kubectl get services
```

显示

```shell
NAME         TYPE        CLUSTER-IP   EXTERNAL-IP   PORT(S)   AGE
kubernetes   ClusterIP   172.16.0.1   <none>        443/TCP   15h
```

#### get service detail

```shell
kubectl get service  kubernetes -o json
```

显示：

```shell
{
    "apiVersion": "v1",
    "kind": "Service",
    "metadata": {
        "creationTimestamp": "2019-02-25T00:52:55Z",
        "labels": {
            "component": "apiserver",
            "provider": "kubernetes"
        },
        "name": "kubernetes",
        "namespace": "default",
        "resourceVersion": "44",
        "selfLink": "/api/v1/namespaces/default/services/kubernetes",
        "uid": "xxxxx-xxx-xxxxxx"
    },
    "spec": {
        "clusterIP": "172.16.0.1",
        "ports": [
            {
                "name": "https",
                "port": 443,
                "protocol": "TCP",
                "targetPort": 6443
            }
        ],
        "sessionAffinity": "None",
        "type": "ClusterIP"
    },
    "status": {
        "loadBalancer": {}
    }
}
```

## delete

### delete pod

```shell
kubectl delete pod <podname>
```

### delete service

```shell
kubectl delete service <instancename>
```

### delete deployment

```shell
kubectl delete deployment <deploymentname>
```

## describe

例如：

```shell
kubectl describe service kubernetes
```

显示

```shell
Name:              kubernetes
Namespace:         default
Labels:            component=apiserver
                   provider=kubernetes
Annotations:       <none>
Selector:          <none>
Type:              ClusterIP
IP:                172.16.0.1
Port:              https  443/TCP
TargetPort:        6443/TCP
Endpoints:         xxx.xx.xx.127:6443,xxx.xx.xx.128:6443,xxx.xx.xx.129:6443
Session Affinity:  None
Events:            <none>
```

## logs

显示一个pod中的container的log

```shell
kubectl logs
```

用法：

```shell
  # Return snapshot logs from pod nginx with only one container
  kubectl logs nginx
  
  # Return snapshot logs from pod nginx with multi containers
  kubectl logs nginx --all-containers=true
  
  # Return snapshot logs from all containers in pods defined by label app=nginx
  kubectl logs -lapp=nginx --all-containers=true
  
  # Return snapshot of previous terminated ruby container logs from pod web-1
  kubectl logs -p -c ruby web-1
  
  # Begin streaming the logs of the ruby container in pod web-1
  kubectl logs -f -c ruby web-1
  
  # Display only the most recent 20 lines of output in pod nginx
  kubectl logs --tail=20 nginx
  
  # Show all logs from pod nginx written in the last hour
  kubectl logs --since=1h nginx
  
  # Return snapshot logs from first container of a job named hello
  kubectl logs job/hello
  
  # Return snapshot logs from container nginx-1 of a deployment named nginx
  kubectl logs deployment/nginx -c nginx-1
```

## exec

在一个pod的一个container中执行一句命令

```shell
kubectl exec
```

用法：

```shell
  # Get output from running 'date' from pod 123456-7890, using the first container by default
  kubectl exec 123456-7890 date
  
  # Get output from running 'date' in ruby-container from pod 123456-7890
  kubectl exec 123456-7890 -c ruby-container date
  
  # Switch to raw terminal mode, sends stdin to 'bash' in ruby-container from pod 123456-7890
  # and sends stdout/stderr from 'bash' back to the client
  kubectl exec 123456-7890 -c ruby-container -i -t -- bash -il
  
  # List contents of /usr from the first container of pod 123456-7890 and sort by modification time.
  # If the command you want to execute in the pod has any flags in common (e.g. -i),
  # you must use two dashes (--) to separate your command's flags/arguments.
  # Also note, do not surround your command and its flags/arguments with quotes
  # unless that is how you would execute it normally (i.e., do ls -t /usr, not "ls -t /usr").
  kubectl exec 123456-7890 -i -t -- ls -t /usr
```

## demo

### 新建一个deployment

跑一个deployment：

```shell
kubectl run kubernetes-demo-daiwk --image=xxx/mynode_daiwenkai:1.0.0 --port=8080
```

看一下deployment：

```shell
kubectl get deployments
NAME                    DESIRED   CURRENT   UP-TO-DATE   AVAILABLE   AGE
kubernetes-demo-daiwk   1         1         1            0           8s
```

看下pod

```shell
kubectl get pods
NAME                                     READY     STATUS    RESTARTS   AGE
kubernetes-demo-daiwk-6549bc9c4b-d6m6r   1/1       Running   0          12m
```

启动本地proxy：

```shell
kubectl proxy &
Starting to serve on 127.0.0.1:8001
```

然后记录一下podname

```shell
export POD_NAME=$(kubectl get pods -o go-template --template '{{range .items}}{{.metadata.name}}{{"\n"}}{{end}}')
echo $POD_NAME
kubernetes-demo-daiwk-6549bc9c4b-d6m6r
```

然后：

```shell
curl http://localhost:8001/api/v1/namespaces/default/pods/$POD_NAME/proxy/
Hello Kubernetes ! | Running on: kubernetes-demo-daiwk-6549bc9c4b-d6m6r | v=1
```

如果出现：

```shell
Error: 'dial tcp 172.19.0.10:8080: connect: connection refused'
Trying to reach: 'http://172.19.0.10:8080/'
```

说明第一步的```kubectl run```并没有成功。

然后看下pod的日志：

```shell
kubectl logs $POD_NAME
Kubernetes Bootcamp App Started At: 2019-02-25T17:59:48.607Z | Running On:  kubernetes-demo-daiwk-6549bc9c4b-d6m6r 

Running On: kubernetes-demo-daiwk-6549bc9c4b-d6m6r | Total Requests: 1 | App Uptime: 10.433 seconds | Log Time: 2019-02-25T17:59:59.040Z
Running On: kubernetes-demo-daiwk-6549bc9c4b-d6m6r | Total Requests: 2 | App Uptime: 160.284 seconds | Log Time: 2019-02-25T18:02:28.891Z
Running On: kubernetes-demo-daiwk-6549bc9c4b-d6m6r | Total Requests: 3 | App Uptime: 379.319 seconds | Log Time: 2019-02-25T18:06:07.926Z
Running On: kubernetes-demo-daiwk-6549bc9c4b-d6m6r | Total Requests: 4 | App Uptime: 491.223 seconds | Log Time: 2019-02-25T18:07:59.830Z
Running On: kubernetes-demo-daiwk-6549bc9c4b-d6m6r | Total Requests: 5 | App Uptime: 670.407 seconds | Log Time: 2019-02-25T18:10:59.014Z
Running On: kubernetes-demo-daiwk-6549bc9c4b-d6m6r | Total Requests: 6 | App Uptime: 852.984 seconds | Log Time: 2019-02-25T18:14:01.591Z
```

然后执行env：

```shell
kubectl exec $POD_NAME env
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
HOSTNAME=kubernetes-demo-daiwk-6549bc9c4b-d6m6r
KUBERNETES_PORT_443_TCP_PROTO=tcp
KUBERNETES_PORT_443_TCP_PORT=443
KUBERNETES_PORT_443_TCP_ADDR=172.16.0.1
KUBERNETES_SERVICE_HOST=172.16.0.1
KUBERNETES_SERVICE_PORT=443
KUBERNETES_SERVICE_PORT_HTTPS=443
KUBERNETES_PORT=tcp://172.16.0.1:443
KUBERNETES_PORT_443_TCP=tcp://172.16.0.1:443
NODE_VERSION=11.10.0
YARN_VERSION=1.13.0
HOME=/root
```

然后就可以对这个pod像docker的container一样来搞啦

```shell
kubectl exec -ti $POD_NAME bash
root@kubernetes-demo-daiwk-6549bc9c4b-d6m6r:/# 
```

### 自动扩容

然后可以对我们的deploymen进行扩容：

```shell
kubectl scale --replicas=3 deployment/kubernetes-demo-daiwk
deployment.extensions/kubernetes-demo-daiwk scaled
```

### 暴露端口

然后可以把端口通过NodePort暴露出来：

```shell
kubectl expose deployment/kubernetes-demo-daiwk --type="NodePort" --port 8080
service/kubernetes-demo-daiwk exposed
```

类似地，我们还可以把clusterip，loadbalancer暴露出来：

```shell
kubectl expose deployment/kubernetes-demo-daiwk --type="LoadBalancer" --port 8080
kubectl expose deployment/kubernetes-demo-daiwk --type="ClusterIP" --port 8080
```

这个时候就可以发现：

```shell
kubectl get services
NAME                    TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)          AGE
kubernetes              ClusterIP   172.16.0.1      <none>        443/TCP          17h
kubernetes-demo-daiwk   NodePort    172.16.182.56   <none>        8080:32561/TCP   15m
```

然后describe一下这个service：

```shell
kubectl describe services/kubernetes-demo-daiwk
Name:                     kubernetes-demo-daiwk
Namespace:                default
Labels:                   run=kubernetes-demo-daiwk
Annotations:              <none>
Selector:                 run=kubernetes-demo-daiwk
Type:                     NodePort
IP:                       172.16.182.56
Port:                     <unset>  8080/TCP
TargetPort:               8080/TCP
NodePort:                 <unset>  32561/TCP
Endpoints:                172.19.0.19:8080,172.19.0.20:8080,172.19.0.21:8080
Session Affinity:         None
External Traffic Policy:  Cluster
Events:                   <none>
```

然后把nodeport存一下：

```shell
export NODE_PORT=$(kubectl get services/kubernetes-demo-daiwk -o go-template='{{(index .spec.ports 0).nodePort}}')
echo $NODE_PORT
32561
```

假设我们的vm ip是106.12.xx.xx(怎么来的。。。)

```shell
curl 106.12.xx.xx:$NODE_PORT
Hello Kubernetes! | Running on: kubernetes-demo-daiwk-6549bc9c4b-7tzbs | v=1
```

然后get pods发现确实scale成了3个：

```shell
kubectl get pods
NAME                                     READY     STATUS    RESTARTS   AGE
kubernetes-demo-daiwk-6549bc9c4b-7tzbs   1/1       Running   0          2m
kubernetes-demo-daiwk-6549bc9c4b-d6m6r   1/1       Running   0          17m
kubernetes-demo-daiwk-6549bc9c4b-hsmkc   1/1       Running   0          2m
```

### 带label地查询

然后我们带条件地去get（加上-l参数）：

```shell
kubectl get pods -l run=kubernetes-demo-daiwk
NAME                                     READY     STATUS    RESTARTS   AGE
kubernetes-demo-daiwk-6549bc9c4b-7tzbs   1/1       Running   0          3m
kubernetes-demo-daiwk-6549bc9c4b-d6m6r   1/1       Running   0          19m
kubernetes-demo-daiwk-6549bc9c4b-hsmkc   1/1       Running   0          3m
```

或者

```shell
kubectl get services -l run=kubernetes-demo-daiwk
NAME                    TYPE       CLUSTER-IP      EXTERNAL-IP   PORT(S)          AGE
kubernetes-demo-daiwk   NodePort   172.16.182.56   <none>        8080:32561/TCP   19m
```

然后我们加个label：

```shell
kubectl label pod kubernetes-demo-daiwk-6549bc9c4b-7tzbs a=x1
kubectl label pod kubernetes-demo-daiwk-6549bc9c4b-d6m6r a=x1 --overwrite
kubectl label pod kubernetes-demo-daiwk-6549bc9c4b-hsmkc a=x2 --overwrite
```

再get一下：

```shell
kubectl get pod -l a=x2
NAME                                     READY     STATUS    RESTARTS   AGE
kubernetes-demo-daiwk-6549bc9c4b-hsmkc   1/1       Running   0          7m
```

还有

```shell
kubectl get pod -l a=x1
NAME                                     READY     STATUS    RESTARTS   AGE
kubernetes-demo-daiwk-6549bc9c4b-7tzbs   1/1       Running   0          7m
kubernetes-demo-daiwk-6549bc9c4b-d6m6r   1/1       Running   0          23m
```

我们来describe一下，可以发现加上了我们指定的label~

```shell
kubectl describe pod kubernetes-demo-daiwk-6549bc9c4b-7tzbs
Name:           kubernetes-demo-daiwk-6549bc9c4b-7tzbs
Namespace:      default
Node:           192.168.0.142/192.168.0.142
Start Time:     Tue, 26 Feb 2019 02:15:32 +0800
Labels:         a=x1
                pod-template-hash=2105675706
                run=kubernetes-demo-daiwk
...
```

### pod的自动恢复

```shell
kubectl delete pod kubernetes-demo-daiwk-6549bc9c4b-hsmkc
```

然后我们可以发现自己又拉了一个起来

```shell
kubectl get pod
NAME                                     READY     STATUS    RESTARTS   AGE
kubernetes-demo-daiwk-6549bc9c4b-7tzbs   1/1       Running   0          10m
kubernetes-demo-daiwk-6549bc9c4b-d6m6r   1/1       Running   0          26m
kubernetes-demo-daiwk-6549bc9c4b-jg7p5   1/1       Running   0          35s
```

### pod的自动分配

2个Node上跑3个pod，1个node跑2个，1个node跑1个，杀掉跑2个pod的node，看pod的变化

看到还是有3个pod在跑，只不过是新增了两个pod，且跑在剩余的node上

扩容原则：

+ 有pod处于pending状态

缩容原则：

+ node少于50%利用率
+ 上面的pod可以被调到其他节点上
+ 没有系统pod
+ 没有用本地磁盘的pod
+ node没有打上不驱逐的label

### 构建namespace

```shell
kubectl create namespace daiwk
```

之后所有操作都带上```-namespace=daiwk```，例如

```shell
kubectl run kubernetes-demo-daiwk --image=xxx/mynode_daiwenkai:1.0.0 --port=8080 --namespace=daiwk
```

### 从yaml文件出发

```shell
kubectl create –f xxx.yaml
```

加上参数```--dry-run```用于测试

yaml文件示例：

Namesapce:

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: daiwk
```

Deployment:

```yaml
apiVersion: extensions/v1beta1
kind: Deployment
metadata:
  name: kubernetes-demo-daiwk
  namespace: daiwk
spec:
  replicas: 3
  template:
    metadata:
      labels:
        app: mynode
        track: stable
        version: 1.0.0
    spec:
      containers:
        - name: mynode
          image: "xxx/mynode_daiwenkai:1.0.0"
          ports:
            - name: http
              containerPort: 8080
```

Service:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: mynode-svc
  labels:
    app: mynode
  namespace: daiwk
spec:
  ports:
  - port: 8080
    targetPort: 8080
  type: NodePort
  selector:
    app: mynode
```

然后呢：

```shell
kubectl get service --namespace=daiwk
NAME                    TYPE       CLUSTER-IP       EXTERNAL-IP   PORT(S)          AGEx
mynode-svc              NodePort   172.16.122.209   <none>        8080:31671/TCP   1m
```

假设我们的vm ip是106.12.xx.xx，完美~

```shell
curl 106.12.xx.xx:31671
Hello Kubernetes ! | Running on: kubernetes-demo-daiwk-78659d8587-bts5s | v=1
```
