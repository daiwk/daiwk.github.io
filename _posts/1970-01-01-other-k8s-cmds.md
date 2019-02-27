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
  - [get replicasets](#get-replicasets)
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
- [自动扩容](#%E8%87%AA%E5%8A%A8%E6%89%A9%E5%AE%B9-1)
- [版本更新](#%E7%89%88%E6%9C%AC%E6%9B%B4%E6%96%B0)
- [小流量](#%E5%B0%8F%E6%B5%81%E9%87%8F)
- [abtest](#abtest)

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

```shell
kubectl get deployments
NAME                  DESIRED   CURRENT   UP-TO-DATE   AVAILABLE   AGE
kubernetes-bootcamp   1         1         1            1           2m
```


### get nodes

```shell
kubectl get nodes -o wide
```

显示

```shell
NAME            STATUS    ROLES     AGE       VERSION   INTERNAL-IP     EXTERNAL-IP   OS-IMAGE                KERNEL-VERSION              CONTAINER-RUNTIME
192.168.0.142   Ready     <none>    15h       v1.11.5   192.168.0.142   <none>        CentOS Linux 7 (Core)   3.10.0-957.1.3.el7.x86_64   docker://18.9.2
```

### get replicasets

```shell
kubectl get replicasets -o wide 
```

显示

```shell
kubectl get replicasets -o wide 
NAME                             DESIRED   CURRENT   READY     AGE       CONTAINERS            IMAGES                                   SELECTOR
kubernetes-bootcamp-8465764b6b   1         1         1         4m        kubernetes-bootcamp   hub.baidubce.com/bootcamp/mynode:1.0.0   pod-template-hash=4021320626,run=kubernetes-bootcamp
```

### get pods

```shell
kubectl get pods -o wide
```

显示

```shell
kubectl get pods -o wide
NAME                                   READY     STATUS    RESTARTS   AGE       IP            NODE            NOMINATED NODE
kubernetes-bootcamp-8465764b6b-8555s   1/1       Running   0          3m        172.19.0.29   192.168.0.142   <none>
```

命名：deployment名-replicaset名-pod名

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

看下pod，可见这个deployment里有1个pod，如果我们设置replicas为3，那应该有3个pods

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

然后可以把deploy通过端口再通过NodePort暴露出来，这样外网就可以访问了，否则只有登上k8s集群内部的机器才能访问：

```shell
kubectl expose deployment/kubernetes-demo-daiwk --type="NodePort" --port 8080
service/kubernetes-demo-daiwk exposed
```

这个时候就可以发现：

```shell
kubectl get services
NAME                    TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)          AGE
kubernetes              ClusterIP   172.16.0.1      <none>        443/TCP          17h
kubernetes-demo-daiwk   NodePort    172.16.182.56   <none>        8080:32561/TCP   15m
```

类似地，我们还可以把clusterip，loadbalancer暴露出来：

```shell
kubectl expose deployment/kubernetes-demo-daiwk --type="LoadBalancer" --port 8080
kubectl expose deployment/kubernetes-demo-daiwk --type="ClusterIP" --port 8080
```

变成loadbalancer的时候，external-ip就会有一个值：

```shell
kubectl get service --namespace=daiwk
NAME                    TYPE           CLUSTER-IP       EXTERNAL-IP     PORT(S)          AGE
daiwk-1                 NodePort       172.16.177.108   <none>          8080:32689/TCP   34m
daiwk-2                 LoadBalancer   172.16.32.199    106.12.43.102   8080:31405/TCP   13m
kubernetes-demo-daiwk   NodePort       172.16.193.53    <none>          8080:31763/TCP   17h
```

这样就可以

```shell
 curl 106.12.43.102:8080
Hello Kubernetes! | Running on: daiwk-2-56d68cc894-mbmhj | v=1
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

kubectl exec可以直接进到容器里，然后可以直接curl servicename:port

```shell
kubectl exec -it daiwk-1-748867bfb-h65tr /bin/bash --namespace=daiwk    
root@daiwk-1-748867bfb-h65tr:/# curl daiwk-1:8080                
Hello Kubernetes bootcamp! | Running on: daiwk-1-748867bfb-h65tr | v=1
root@daiwk-1-748867bfb-h65tr:/# curl daiwk-1-748867bfb-h65tr:8080
Hello Kubernetes bootcamp! | Running on: daiwk-1-748867bfb-h65tr | v=1
```

常见的配置文件处理方式有几种：

1. 使用env的方式，将配置以环境变量方式注入到容器内
2. 使用kubernetes的configmap，将配置文件挂载到容器内/注入到环境变量中
3. 同代码一同build到容器内
4. 将配置保存到宿主机，通过目录挂载到容器内

对于1，需要update/create deployment
对于2，在configmap更新后重启pod即可
对于3的方式比较简单，重新上传容器，rolling update即可
对于4的方式，如果程序支持热加载，不需要做任何操作，但是缺点是损失无状态，且配置修改需要登录到机器上操作

这几种方式各有优劣，基本上要根据实际情况做分析和选型

如果使用kubernetes的configmap或secret，kubernetes会自动实施更新，所以如果程序支持热加载配置，也同样可以不需要重启pod

如果delete了一个deployment，但service还在，那么：

```shell
kubectl describe service mypython-svc --namespace=daiwenkai-2
Name:                     mypython-svc
Namespace:                daiwenkai-2
Labels:                   app=mypython
Annotations:              <none>
Selector:                 app=mypython
Type:                     NodePort
IP:                       172.16.72.253
Port:                     <unset>  8080/TCP
TargetPort:               8080/TCP
NodePort:                 <unset>  31753/TCP
Endpoints:                <none>
Session Affinity:         None
External Traffic Policy:  Cluster
Events:                   <none>
```

可以发现endpoints没东西了，说明这个service死掉了。。那么我们可以再搞个deployment，只要能select到，就又活了

## 自动扩容

通过如下Dockerfile进行build并Push一个镜像：

```shell
FROM node:slim
EXPOSE 8080
COPY server.js .
CMD node server.js
```

其中的server.js如下：

```js
var http = require('http');
var requests=0;
var podname= process.env.HOSTNAME;
var startTime;
var host;
var handleRequest = function(request, response) {
  // run loop for increasing usage of cpu
  for (var i=0; i< 1000000000; i++) {

  }
  response.setHeader('Content-Type', 'text/plain');
  response.writeHead(200);
  response.write("Hello Kubernetes bootcamp! | Running on: ");
  response.write(host);
  response.end(" | v=3\n");
  console.log("Running On:" ,host, "| Total Requests:", ++requests,"| App Uptime:", (new Date() - startTime)/1000 , "seconds", "| Log Time:",new Date());
}
var www = http.createServer(handleRequest);
www.listen(8080,function () {
    startTime = new Date();
    host = process.env.HOSTNAME;
    console.log ("Kubernetes Bootcamp App Started At:",startTime, "| Running On: " ,host, "\n" );
});
```

通过如下yaml建立一个deployment：

```yaml
apiVersion: extensions/v1beta1
kind: Deployment
metadata:
  name: mynode-deployment
  namespace: daiwenkai-2
spec:
  replicas: 1
  template:
    metadata:
      labels:
        app: mynode
        track: stable
        version: 1.0.0
    spec:
      containers:
        - name: mynode
          resources:
              requests:
                  cpu: "300m"
                  memory: 1Gi
              limits:
                  cpu: "500m"
                  memory: 2Gi
          imagePullPolicy: Always
          image: "xxxx/mynode_node_daiwenkai:autoscale"
          ports:
            - name: http
              containerPort: 8080
```

然后自动扩容

```shell
kubectl autoscale deployment mynode-deployment --min=1 --max=4 --cpu-percent=2 --namespace=daiwenkai-2
```

成功的话，会提示：```horizontalpodautoscaler.autoscaling/mynode-deployment autoscaled```

然后通过以下yaml建立一个service：

```yaml
apiVersion: v1
kind: Service
metadata:
  name: mynode-svc
  namespace: daiwenkai-2
  labels:
    app: mynode
spec:
  ports:
  - port: 8080
    targetPort: 8080
  type: NodePort
  selector:
    app: mynode
```

然后获取端口：

```shell
export NODE_PORT=$(kubectl get services/mynode-svc -o go-template='{{(index .spec.ports 0).nodePort}}' --namespace=daiwenkai-2)
```

然后我们用ab来压测（如果是Centos，可以通过```yum -y install httpd-tools```来安装）：

```shell
```

然后我们可以监控实例：

```shell
watch -n 1 'kubectl top pod --namespace=daiwenkai-2'
```

一开始可能是：

```shell
Every 1.0s: kubectl top pod --namespace=daiwenkai-2                                                                                      Wed Feb 27 20:12:34 2019

NAME                                 CPU(cores)   MEMORY(bytes)
mynode-deployment-64c64d98b9-6rx85   164m         8Mi
```

当压力大的时候(cpu超过200m的时候)，就会自动扩容：

```shell
Every 1.0s: kubectl top pod --namespace=daiwenkai-2                                                                                      Wed Feb 27 20:25:15 2019

NAME                                 CPU(cores)   MEMORY(bytes)
mynode-deployment-64c64d98b9-2gphw   0m           8Mi
mynode-deployment-64c64d98b9-rc6ng   500m         13Mi
mynode-deployment-64c64d98b9-rtnrq   0m           10Mi
mynode-deployment-64c64d98b9-zmlgd   22m          8Mi
```

删除自动扩容策略：

```shell
kubectl delete horizontalpodautoscaler.autoscaling/mynode-deployment --namespace=daiwenkai-2
```

## 版本更新

set image可以改image

例如：

```shell
kubectl set image deployments/mypython-deployment mypython=xxxx/bootcamp_7/mynode_python_daiwenkai:2.0.0 --namespace=daiwenkai-2
```

然后就可以看到：

```shell
kubectl get pods --namespace=daiwenkai-2                                                                               NAME                                   READY     STATUS        RESTARTS   AGE
mypython-deployment-5989ddfd4f-652kd   1/1       Running       0          8s
mypython-deployment-5989ddfd4f-fx8cj   1/1       Running       0          9s
mypython-deployment-5989ddfd4f-lfzwt   1/1       Running       0          11s
mypython-deployment-5989ddfd4f-ngpm2   1/1       Running       0          11s
mypython-deployment-66bc86658-qvmm5    1/1       Terminating   0          15m
mypython-deployment-66bc86658-tcjdf    1/1       Terminating   0          11m
mypython-deployment-66bc86658-z4t4m    1/1       Terminating   0          15m
mypython-deployment-66bc86658-z9c25    1/1       Terminating   0          15m
```

再过一会儿，就是：

```shell
kubectl get pods --namespace=daiwenkai-2
NAME                                   READY     STATUS    RESTARTS   AGE
mypython-deployment-5989ddfd4f-652kd   1/1       Running   0          41s
mypython-deployment-5989ddfd4f-fx8cj   1/1       Running   0          42s
mypython-deployment-5989ddfd4f-lfzwt   1/1       Running   0          44s
mypython-deployment-5989ddfd4f-ngpm2   1/1       Running   0          44s
```

可以查看进度(显示的是running的个数，如果还有在terminating的，只要都running了，就是succ)：

```shell
kubectl rollout status deployments/mypython-deployment --namespace=daiwenkai-2
deployment "mypython-deployment" successfully rolled out
```

然后我们可以回滚：

```shell
kubectl rollout undo deployment/mypython-deployment --namespace=daiwenkai-2
```

## 小流量

先创建一个deployment，里面有3个pod，version是1

```yaml
apiVersion: extensions/v1beta1
kind: Deployment
metadata:
  name: mypython-deployment
  namespace: daiwenkai-2
spec:
  replicas: 3
  template:
    metadata:
      labels:
        app: mypython
        track: stable
        version: 1.0.0
    spec:
      containers:
        - name: mypython
          image: "xxx/bootcamp_7/mynode_python_daiwenkai:1.0.0"
          ports:
            - name: http
              containerPort: 8080
```

再创建一个deployment，里面有1个pod，version是2

```yaml
apiVersion: extensions/v1beta1
kind: Deployment
metadata:
  name: mypython-canary
  namespace: daiwenkai-2
spec:
  replicas: 1
  template:
    metadata:
      labels:
        app: mypython
        track: canary
        version: 2.0.0
    spec:
      containers:
        - name: hello
          image: "xxx/bootcamp_7/mynode_python_daiwenkai:2.0.0"
          ports:
            - name: http
              containerPort: 8080
```

然后创建一个service，把这四个pod连接起来（通过app=mypython连接到一起~）：

```yaml
apiVersion: v1
kind: Service
metadata:
  name: mypython-svc
  namespace: daiwenkai-2
  labels:
    app: mypython
spec:
  ports:
  - port: 8080
    targetPort: 8080
  type: NodePort
  selector:
    app: mypython
```

这样，我们往这个Service发请求，就会发现75%的流量在v1上，25%的流量在v2上

## abtest

先建一个deployment：

```yaml
apiVersion: extensions/v1beta1
kind: Deployment
metadata:
  name: mypython-deployment
  namespace: daiwenkai-2
spec:
  replicas: 3
  template:
    metadata:
      labels:
        app: mypython
        track: stable
        version: 1.0.0
    spec:
      containers:
        - name: mypython
          image: "xxx/bootcamp_7/mynode_python_daiwenkai:1.0.0"
          ports:
            - name: http
              containerPort: 8080
```

再建一个deployment，唯一的区别就是version和name不一样，其他都一样，包括replicas：

```yaml
apiVersion: extensions/v1beta1
kind: Deployment
metadata:
  name: mypython-green
  namespace: daiwenkai-2
spec:
  replicas: 3
  template:
    metadata:
      labels:
        app: mypython
        track: stable
        version: 2.0.0
    spec:
      containers:
        - name: mypython
          image: "xxx/bootcamp_7/mynode_python_daiwenkai:2.0.0"
          ports:
            - name: http
              containerPort: 8080
```

同样地，用一个service把这6个node都连起来(selector是app=mypython)：

```yaml
apiVersion: v1
kind: Service
metadata:
  name: mypython-svc
  namespace: daiwenkai-2
  labels:
    app: mypython
spec:
  ports:
  - port: 8080
    targetPort: 8080
  type: NodePort
  selector:
    app: mypython
```

然后我们可以通过```apply -f```命令来通过yaml文件来修改对象：

比如我们用```mynode-1.0.0.yaml```这个文件来选定v1版本```kubectl apply -f mynode-1.0.0.yaml```，这样发请求，这6个节点就都是v1了：

```yaml
kind: Service
apiVersion: v1
metadata:
  name: mypython-svc
  namespace: daiwenkai-2
spec:
  selector:
    app: mypython
    version: 1.0.0
```

类似地，我们用```mynode-2.0.0.yaml```这个文件来选定v2版本```kubectl apply -f mynode-2.0.0.yaml```，这样发请求，这6个节点就都是v2了：

```yaml
kind: Service
apiVersion: v1
metadata:
  name: "mypython-svc"
  namespace: daiwenkai-2
spec:
  selector:
    app: "mypython"
    version: 2.0.0
```
