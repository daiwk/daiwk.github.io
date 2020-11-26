---
layout: post
category: "knowledge"
title: "disk"
tags: [disk, ]
---

目录

<!-- TOC -->


<!-- /TOC -->


先看一下现状

```shell
[root@daiwk-dev ~]# df -h
Filesystem            Size  Used Avail Use% Mounted on
/dev/vda1              40G  9.3G   29G  25% /
tmpfs                 3.9G   11M  3.9G   1% /dev/shm
```

看看当前有多少磁盘待挂载

```shell
[root@daiwk-dev ~]#  fdisk -l

Disk /dev/vda: 42.9 GB, 42949672960 bytes
16 heads, 63 sectors/track, 83220 cylinders
Units = cylinders of 1008 * 512 = 516096 bytes
Sector size (logical/physical): 512 bytes / 512 bytes
I/O size (minimum/optimal): 512 bytes / 512 bytes
Disk identifier: 0xea92f8b1

   Device Boot      Start         End      Blocks   Id  System
/dev/vda1               1       83220    41942848+  83  Linux

Disk /dev/vdb: 536.9 GB, 536870912000 bytes
16 heads, 63 sectors/track, 1040253 cylinders
Units = cylinders of 1008 * 512 = 516096 bytes
Sector size (logical/physical): 512 bytes / 512 bytes
I/O size (minimum/optimal): 512 bytes / 512 bytes
Disk identifier: 0x00000000


Disk /dev/vdc: 536.9 GB, 536870912000 bytes
16 heads, 63 sectors/track, 1040253 cylinders
Units = cylinders of 1008 * 512 = 516096 bytes
Sector size (logical/physical): 512 bytes / 512 bytes
I/O size (minimum/optimal): 512 bytes / 512 bytes
Disk identifier: 0x00000000


Disk /dev/vdd: 536.9 GB, 536870912000 bytes
16 heads, 63 sectors/track, 1040253 cylinders
Units = cylinders of 1008 * 512 = 516096 bytes
Sector size (logical/physical): 512 bytes / 512 bytes
I/O size (minimum/optimal): 512 bytes / 512 bytes
Disk identifier: 0x00000000

```

格式化掉，要等一段时间

```shell
[root@daiwk-dev ~]# mkfs.ext4 /dev/vdb
mke2fs 1.41.12 (17-May-2010)
Filesystem label=
OS type: Linux
Block size=4096 (log=2)
Fragment size=4096 (log=2)
Stride=0 blocks, Stripe width=0 blocks
65536000 inodes, 131072000 blocks
6553600 blocks (5.00%) reserved for the super user
First data block=0
Maximum filesystem blocks=4294967296
4000 block groups
32768 blocks per group, 32768 fragments per group
16384 inodes per group
Superblock backups stored on blocks:
	32768, 98304, 163840, 229376, 294912, 819200, 884736, 1605632, 2654208,
	4096000, 7962624, 11239424, 20480000, 23887872, 71663616, 78675968,
	102400000

Writing inode tables: done
Creating journal (32768 blocks): done
Writing superblocks and filesystem accounting information:
done

This filesystem will be automatically checked every 35 mounts or
180 days, whichever comes first.  Use tune2fs -c or -i to override.
```

挂载到/home/work下：

```shell
[root@daiwk-dev ~]# mount -t ext4 /dev/vdb /home/
[root@daiwk-dev ~]# df -h
Filesystem            Size  Used Avail Use% Mounted on
/dev/vda1              40G   14G   25G  35% /
tmpfs                 3.9G   11M  3.9G   1% /dev/shm
/dev/vdb              485G   70M  460G   1% /home
```

编辑配置文件使云盘开机自动挂载  （注意，文件格式错误会导致机器重启失败。）

```
vim /etc/fstab 
```

加入一行

```
/dev/vdc        /home   auto    defaults,nofail,comment=cloudconfig     0       2
```

同样地，可以挂一下vdc，但不要也挂到/home/work下，可以挂到/home/disk1下

如果不慎将/dev/vdc 挂载到了 /home/，导致原来数据被隐藏，可以用root用户

```
vim /etc/mtab
```

将最底下的一行

```shell
/dev/vdc /home ext4 rw 0 0
```

删除，然后

```shell
mkdir -p /home/disk1
mount -t ext4 /dev/vdc /home/disk1
```

这样原来的数据就可以找回来，并且将新盘挂载到了/home/disk1目录

另外，可以通过```df -Th```把Type打出来

```shell
[root@daiwk-dev ~]# df -Th
Filesystem    Type    Size  Used Avail Use% Mounted on
/dev/vda1     ext4     40G   14G   25G  35% /
tmpfs        tmpfs    3.9G   11M  3.9G   1% /dev/shm
/dev/vdb      ext4    485G  1.7G  458G   1% /home
/dev/vdc      ext4    485G   70M  460G   1% /home/disk1
```
