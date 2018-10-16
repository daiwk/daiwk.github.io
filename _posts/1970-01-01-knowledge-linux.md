---
layout: post
category: "knowledge"
title: "linux"
tags: [linux, ]
---

目录

<!-- TOC -->

- [查看机器支持的指定集](#%E6%9F%A5%E7%9C%8B%E6%9C%BA%E5%99%A8%E6%94%AF%E6%8C%81%E7%9A%84%E6%8C%87%E5%AE%9A%E9%9B%86)

<!-- /TOC -->

## 查看机器支持的指定集

```shell
work@xxx$ cat /proc/cpuinfo | grep flags | uniq
flags           : fpu vme de pse tsc msr pae mce cx8 apic mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon pebs bts rep_good xtopology nonstop_tsc aperfmperf pni pclmulqdq dtes64 ds_cpl vmx smx est tm2 ssse3 fma cx16 xtpr pdcm dca sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch arat epb xsaveopt pln pts dts tpr_shadow vnmi flexpriority ept vpid fsgsbase bmi1 hle avx2 smep bmi2 erms invpcid rtm
```

