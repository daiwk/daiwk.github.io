---
layout: post
category: "other"
title: "在docker中不要使用sshd"
tags: [docker, ssh]
---

目录

<!-- TOC -->

- [备份我的数据?](#备份我的数据)
- [检查日志?](#检查日志)
- [重启service?](#重启service)
- [修改我的配置文件](#修改我的配置文件)
- [调试我的应用？](#调试我的应用)
- [如何远程访问呢？](#如何远程访问呢)

<!-- /TOC -->

参考[博客](http://www.oschina.net/translate/why-you-dont-need-to-run-sshd-in-docker?cmp)

# 备份我的数据?

你的数据应该存在于 volume中. 然后你可以使用--volumes-from选项来运行另一个容器，与第一个容器共享这个volume。这样做的好处：如果你需要安装新的工具（如s75pxd）来将你备份的数据长期保存，或将数据转移到其他永久存储时，你可以在这个特定的备份容器中进行，而不是在主服务容器中。这很简洁。

# 检查日志?

再次使用 volume! 如果你将所有日志写入一个特定的目录下，且这个目录是一个volume的话，那你可以启动另一个log inspection" 容器（使用--volumes-from，还记得么?)且在这里面做你需要做的事。如果你还需要特殊的工具（或只需要一个有意思的ack-grep），你可以在这个容器中安装它们，这样可以保持主容器的原始环境。

# 重启service?

基本上所有service都可以通过信号来重启。当你使用/etc/init.d/foo restart或service foo restart时，实际上它们都会给进程发送一个特定的信号。你可以使用docker kill -s <signal>来发送这个信号。一些service可能不会监听这些信号，但可以在一个特定的socket上接受命令。如果是一个TCP socket，只需要通过网络连接上就可以了。如果是一个UNIX套接字，你可以再次使用volume。将容器和service的控制套接字设置到一个特定的目录中，且这个目录是一个volume。然后启动一个新的容器来访问这个volume；这样就可以使用UNIX套接字了。

“但这也太复杂了吧！”－其实不然。假设你名为foo的servcie 在/var/run/foo.sock创建了一个套接字，且需要你运行fooctl restart来完成重启。只需要使用-v /var/run(或在Docker文件中添加VOLUME /var/run)来启动这个service就可以了。当你想重启的时候，使用--volumes-from选项并重载命令来启动相同的镜像。像这样：

```
# Starting the service
CID=$(docker run -d -v /var/run fooservice)
# Restarting the service with a sidekick container
docker run --volumes-from $CID fooservice fooctl restart
```

# 修改我的配置文件

如果你正在执行一个持久的配置变更，你最好把他的改变放在image中，因为如果你又启动一个container，那么服务还是使用的老的配置，你的配置变更将丢失。所以，没有您的SSH访问！“但是我需要在服务存活期间，改变我的配置；例如增加一个新的虚拟站点！”这种情况下，你需要使用……等待……volume！配置应该在volume中，并且该volume应该和一个特殊目的“配置编辑器”容器共享。你可以在这个容器中使用任何你喜欢的东西：SSH + 你最喜欢的编辑器，或一个接受API调用的web服务，或一个从外部源抓取信息的定时任务；诸如此类。另外，分离关注：一个容器运行服务，另外一个处理配置更新。“但是我做临时更改，因为我正在测试不同的值！”在这种情况下，查看下一章节！

# 调试我的应用？

这可能是唯一需要进入container的场景了。因为你要运行gdb, strace, tweak配置，等。这种情况下，你需要 nsenter。

nsenter是一个小的工具，用来进入命名空间中。技术上，它可以进入现有的命名空间，或者产生一个进程进入新的一组命名空间。“命名空间是什么?”他们是容器的重要组成部分。简单点说：通过使用 nsenter ，你可以进入一个已经存在的container中，尽管这个container没有运行ssh 或者任意特殊用途的守护进程。

```shell
#安装nsenter
docker run -v /usr/local/bin:/target jpetazzo/nsenter

#登陆到容器中
imageid=8aa68fd0bc55 # e.g.
PID=$(docker inspect --format {{.State.Pid}} $imageid)
nsenter --target $PID --mount --uts --ipc --net --pid
```

# 如何远程访问呢？

如果你需要从一个远程主机进入一个容器，有（至少）两个方法：

+ SSH 进入 Docker 主机，并使用 nsenter;
+ SSH 进入 Docker 主机，通过一个特殊的密钥参数授权nsenter命令  (也就是，nsenter)。

第一种方法相对简单；但是需要root权限访问Docker主机（从安全角度来说不是很好）。第二种方法在 SSH 的 authorized_keys 文件中使用 command= 模式。你可能熟悉 “古典的” authorized_keys文件，它看起来像这样： 

```
ssh-rsa bbb…QOID== aaa@ccc
```


（当然，实际上一个真正的密钥是很长的，一般都会占据好几行。）你也可以强制使用一个专有的命令。如果你想要在你的系统上查看一个远程的主机上可以有效使用的内存，可以使用SSH密钥，但是你不会希望交出所有的shell权限，你可以在authorized_keys文件中输入下面的内容：

```
command="free" ssh-rsa bbb…QOID== aaa@ccc
```

现在，当使用专有的密钥进行连接时，替换取得的shell，它可以执行free命令。除此之外，就不能做其他的。（通常，你可能还想要添加no-port-forwarding；如果希望了解更多信息可以查看authorized_keys(5)的手册（manpage））。这种机制的关键是使得责任分离。Alice把服务放在容器内部；她不用处理远程的访问，登陆等事务。Betty会添加SSH层，在特殊情况（调试奇怪的问题）下使用。Charlotte会考虑登陆。等等。