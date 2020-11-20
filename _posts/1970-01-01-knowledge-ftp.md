---
layout: post
category: "knowledge"
title: "ftp"
tags: [ftp, ftpasswd, ]
---

目录

<!-- TOC -->


<!-- /TOC -->

设置FTP虚拟账号密码

修改proftpd配置文件（早期系统镜像的默认FTP配置路径为/etc/proftpd.conf）

注释```<Anonymous ~ftp> … </Anonymous>```之间相关配置

```<Anonymous ~ftp> … </Anonymous>```之外新增如下配置

```shell
AuthOrder mod_auth_file.c

AuthUserFile /etc/proftpd.passwd

RequireValidShell off
```

获取ftpasswd用于设置虚拟账号

```shell
wget http://www.castaglia.org/proftpd/contrib/ftpasswd
mv ftpasswd /usr/sbin
chmod +x /usr/sbin/ftpasswd
```

设置账号密码

```shell
ftpasswd --file=/etc/proftpd.passwd --home=xxx --shell=/bin/false --name=xxx --uid=99 --gid=99 --passwd
```

配置说明

```shell
--home=xxx 指定 ftp 用户登录后的根目录（eg. --home=/home）
--name=xxx 指定 ftp 用户名
--uid=99 --gid=99 指定账号关联对应系统用户和组（eg. 99:nobody/501:work 可通过命令id work查看用户对应uid/gid）
```

例如：

```shell
 ./ftpasswd --file=/etc/proftpd.passwd --home=// --shell=/bin/sh --name=dwk --uid=505 --gid=506 --passwd
```

强密码生成命令（仅供参考）：

```shell
strings /dev/urandom |tr -dc A-Za-z0-9 | head -c20; echo
```

wget获取文件路径以–home指定路径为基础进行拼接（eg. --home=/home; wget ftp://…/home/work => wget ftp://…/work ）
删除ftpasswd
cd /usr/sbin; rm -f ftpasswd
重启FTP服务(如启动失败，proftpd -t -d5 检查配置文件出错点)

```shell
service proftpd restart
```

使用时：

```shell
wget ftp://xxxxxxx/x.txt --ftp-user=dwk --ftp-password=dwk
```
