---
layout: post
category: "dl"
title: "tf serving docker+k8s"
tags: [tf serving, docker, k8s ]
---

目录

<!-- TOC -->

- [docker + tf-serving](#docker--tf-serving)
- [k8s](#k8s)

<!-- /TOC -->

## docker + tf-serving

**注意：1.6版本的docker不行（没有--link参数，-p参数不是port），亲测1.9.1的docker可以。。**

参考：[https://yq.aliyun.com/articles/60894](https://yq.aliyun.com/articles/60894)

首先，获取两个镜像：

+ registry.cn-hangzhou.aliyuncs.com/denverdino/tensorflow-serving : TensorFlow Serving的基础镜像
+ registry.cn-hangzhou.aliyuncs.com/denverdino/inception-serving : 基于上述基础镜像添加Inception模型实现的服务镜像

```shell
docker pull registry.cn-hangzhou.aliyuncs.com/denverdino/tensorflow-serving
docker pull registry.cn-hangzhou.aliyuncs.com/denverdino/inception-serving
```

我们利用Docker命令启动名为 “inception-serving” 容器作为TF Serving服务器

```shell
docker run -d -p 9000:9000 --name inception-serving registry.cn-hangzhou.aliyuncs.com/denverdino/inception-serving
```

启动 “tensorflow-serving” 镜像作为客户端，并定义容器link，允许在容器内部通过“serving”别名来访问“inception-serving”容器

```shell
docker run -dti --name client --link inception-serving:serving        registry.cn-hangzhou.aliyuncs.com/denverdino/tensorflow-serving
```

找到刚才的容器id，用docker-enter来搞：

```shell
docker ps
CONTAINER ID        IMAGE                                                             COMMAND                  CREATED             STATUS              PORTS                    NAMES
471be814a041        registry.cn-hangzhou.aliyuncs.com/denverdino/tensorflow-serving   "/bin/bash"              4 seconds ago       Up 3 seconds        9000/tcp                 client
26d6fa778d23        registry.cn-hangzhou.aliyuncs.com/denverdino/inception-serving    "/serving/bazel-bin/t"   22 minutes ago      Up 22 minutes       0.0.0.0:9000->9000/tcp   inception-serving
```

发现我们的client是471be814a041

```shell
root@471be814a041:~# curl http://f.hiphotos.baidu.com/baike/w%3D268%3Bg%3D0/sign=6268660aafec8a13141a50e6cf38f6b2/32fa828ba61ea8d3c85b36e1910a304e241f58dd.jpg -o persian_cat_image.jpg

  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0/serving/bazel-bin/tensorflow_serving/example/inception_client --server=serving:900100 10703  100 10703    0     0   232k      0 --:--:-- --:--:-- --:--:--  237k
root@471be814a041:~# 
root@471be814a041:~# /serving/bazel-bin/tensorflow_serving/example/inception_client --server=serving:9000 --image=$PWD/persian_cat_image.jpg
D0602 12:58:22.647239059      43 ev_posix.c:101]             Using polling engine: poll
outputs {
  key: "classes"
  value {
    dtype: DT_STRING
    tensor_shape {
      dim {
        size: 1
      }
      dim {
        size: 5
      }
    }
    string_val: "Persian cat"
    string_val: "lynx, catamount"
    string_val: "Egyptian cat"
    string_val: "Angora, Angora rabbit"
    string_val: "tabby, tabby cat"
  }
}
outputs {
  key: "scores"
  value {
    dtype: DT_FLOAT
    tensor_shape {
      dim {
        size: 1
      }
      dim {
        size: 5
      }
    }
    float_val: 9.49124622345
    float_val: 3.08905720711
    float_val: 2.88204526901
    float_val: 2.85937643051
    float_val: 2.78746366501
  }
}

E0602 12:58:23.565837691      43 chttp2_transport.c:1810]    close_transport: {"created":"@1527944303.565790675","description":"FD shutdown","file":"src/core/lib/iomgr/ev_poll_posix.c","file_line":427}
```



## k8s
