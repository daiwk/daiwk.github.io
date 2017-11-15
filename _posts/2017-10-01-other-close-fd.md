---
layout: post
category: "other"
title: "关闭正在运行的程序正在使用的文件句柄"
tags: [fd, ]
---

目录

<!-- TOC -->


<!-- /TOC -->

[https://stackoverflow.com/questions/323146/how-to-close-a-file-descriptor-from-another-process-in-unix-systems](https://stackoverflow.com/questions/323146/how-to-close-a-file-descriptor-from-another-process-in-unix-systems)

一个爬虫程序写了for循环，但中间一个网络请求卡住了，后面的一直排不上，需要把这个请求kill掉，让后面的继续：

```shell
ps aux | grep new_image_spider.py| grep -v grep | awk -F' ' '{print $2}' 

## 得到的进程号是1234
ll /proc/1234/fd
total 0
lr-x------  1 work work 64 Oct 12 17:58 0 -> /dev/null
l-wx------  1 work work 64 Oct 12 17:58 1 -> /home/disk1/xxx/4.txt
l-wx------  1 work work 64 Oct 12 17:58 2 -> /home/disk1/xxx/3.txt
lr-x------  1 work work 64 Oct 12 17:58 3 -> /home/disk1/xxx/2.txt
l-wx------  1 work work 64 Oct 12 17:58 4 -> /home/disk1/xxx/1.txt
lrwx------  1 work work 64 Oct 12 17:58 5 -> socket:[4273883397]

## 发现5这个fd，也就是socket是红的且卡住了，那么
gdb -p 1234

## 然后
p close(5)

## 再check:
ll /proc/6099/fd
total 0
lr-x------  1 work work 64 Oct 12 17:58 0 -> /dev/null
l-wx------  1 work work 64 Oct 12 17:58 1 -> /home/disk1/xxx/4.txt
l-wx------  1 work work 64 Oct 12 17:58 2 -> /home/disk1/xxx/3.txt
lr-x------  1 work work 64 Oct 12 17:58 3 -> /home/disk1/xxx/2.txt
l-wx------  1 work work 64 Oct 12 17:58 4 -> /home/disk1/xxx/1.txt
lrwx------  1 work work 64 Oct 12 17:58 5 -> socket:[444444555]

## 发现5这个fd换了，成功了！。。
```
