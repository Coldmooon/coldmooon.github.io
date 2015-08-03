---
layout:     post
title:      "从零安装 Caffe"
date:       2015-08-03 23:16:00
author:     "Coldmooon"
header-img: "img/home-bg.jpg"
---

## 一、前言
本文记录了在新安装的 `ubuntu 14.04` 系统下安装 `caffe` 的过程。这里主要参考了两个链接：
1) <http://caffe.berkeleyvision.org/installation.html>
2) <http://ouxinyu.github.io/Blogs/20140723001.html>

很多人不太喜欢看官方教程，但其实 `caffe` 的官方安装指导做的非常好。我在看到 2) 之前，曾根据官方
指导在 `OSX 10.9`, `10.10`, `Ubuntu 12.04`, `14.04` 下安装过 10 多次不同版本的 `caffe`，都成功了。

本文有不少内容参考了 1）和 2），但又有一些内容与二者不同。例如，2）中对 `gcc` 进行了降级，而我却对 `gcc` 进行了升级；与 2）的安装顺序也有些不同。我按照下面的顺序在一台新买的电脑上安装 `caffe`: 

安装 `win7/10`（略） --> 安装 `ubuntu 14.04`（略） --> 升级 `gcc 4.9` -> 安装 `nvidia` 显卡驱动 -> 安装 `cuda` 和 `cudnn` --> 安装 `anaconda` --> 安装 `Opencv 2.4.11` --> 安装 `Matlab` --> 安装 `Caffe`

我之所以要在装完系统的第一时间升级 `gcc 4.9`，是因为 `nvidia` 的驱动和 `cuda` 都需要 `gcc` 进行编译。如果
先安装驱动和 `cuda`，再升级 `gcc` ，那么有时候会出现问题（我就遇到了）。当然，`ubuntu 14.04` 自带的 `gcc-4.8` 已经够用了。但我是个版本控，喜欢用最新的稳定版，所以就选择了升级 `gcc 4.9`。

后文将按照上述安装顺序来写

-------------------------------------------------------

## 二、升级 gcc 4.9

如果只用 `ubuntu 14.04` 自带的 `gcc-4.8` 则本节可以跳过。

```
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install g++-4.9
```
这样安装后，系统默认还是使用 `gcc 4.8`，必须要重新做一下软连接

```
sudo ln -sf /usr/bin/gcc-4.9 /usr/bin/gcc
sudo ln -sf /usr/bin/gcc-ar-4.9 /usr/bin/gcc-ar
sudo ln -sf /usr/bin/gcc-ranlib-4.9 /usr/bin/gcc-ranlib
```
然后，可以用 `gcc -v` 来查看 `gcc` 的版本。

参考链接:
<http://askubuntu.com/questions/428198/getting-installing-gcc-g-4-9-on-ubuntu>

-------------------------

## 三、安装 nvidia 显卡驱动

网上的许多教程都指出要进入 `tty`，把 `lightdm` 关了。但我发现直接用 `apt-get` 安装的话，无需关闭 `lightdm`。

```
sudo add-apt-repository ppa:xorg-edgers/ppa
sudo apt-get update
sudo apt-get install nvidia-352 nvidia-settings nvidia-prime
```

若要安装其他版本你的驱动，则输入:

```
# 331 driver
sudo apt-get install nvidia-331

# 334 driver
sudo apt-get install nvidia-334

# install the latest version
sudo apt-get install nvidia-current
```
安装完后，输入 `prime-select query` 查看当前正在使用的显卡。
输入 `cat /proc/driver/nvidia/version` 查看正在使用的 `nvidia` 驱动版本和编译时采用的 `gcc` 版本

虽然 `cuda` 里已经包含了 `nvidia` 驱动，但是根据 `caffe` 官方指导，`cuda` 与显卡驱动最好分开安装。

参考链接:
<http://ubuntuhandbook.org/index.php/2015/04/install-nvidia-driver-346-59-in-ubuntu-from-ppa/>
<http://www.binarytides.com/install-nvidia-drivers-ubuntu-14-04/>
<http://my.oschina.net/eechen/blog/227134>


---------------------------------


## 四、安装 cuda 7.0
从 <https://developer.nvidia.com/cuda-downloads> 下载对应的 `deb`包。然后双击，在软件中心里安装。
到这里时，并没有完成安装，`deb` 包只是告诉系统去哪里下载 `cuda` 而已。

```
Why doesn't the cuda-repo package install the CUDA Toolkit and Drivers?

When using RPM or Deb, the downloaded package is a repository package. Such
a package only informs the package manager where to find the actual installation
packages, but will not install them.

                                          ---- CUDA_Getting_Started_Linux.pdf
```
接下来输入下列命令安装 `cuda`:
```
sudo apt-get update
sudo apt-get install cuda
```
安装完成后，再设置环境
```
export PATH=/usr/local/cuda-7.0/bin:$PATH    
export LD_LIBRARY_PATH=/usr/local/cuda-7.0/lib64:$LD_LIBRARY_PATH    
```

另外，我发现上面的 `export` 操作在我电脑上不起作用。所以我直接把 `lib64` 里的库文件软连接到了 `/usr/local/lib/` 下 

--------------------------------

## 五、安装 cudnn

从官方网站下载 `cudnn` 后解压。得到的文件是 `.h` 和 `.so` 文件。所以，直接把他们拷贝到 `/usr/local/include` 和  `/usr/local/lib/` 下就好了。

---------------------------------------------------


## 六、安装 anaconda
强烈推荐使用 `anaconda` 的 `python`，它里面集成了很多包，省去了很多麻烦。如果有 edu 邮箱的话，还可以获得 `accelerate anaconda`，在矩阵运算的时候，可以启用并行计算，速度快很多。
```
./Anaconda-2.3.0-Linux-x86_64.sh
conda update conda
conda install accelerate
conda install iopro
```
接下来拷贝 `anaconda` 的许可文件到用户主目录

`mv license_academic_20150611072013.txt ~/.continuum`

然后升级 `ipython`, 如果不用 `ipython`，那就跳过下面这一步:

```
conda update ipython
conda update ipython-notebook
```
------------------------------

## 七、安装 Opencv 2.4.11
喜欢 `opencv 3.0` 的，可以选择安装 `opencv 3.0`。这里我继续使用 `Opencv 2.4.11`。
`Opencv` 的安装过程较繁琐，切网上已经有了大量的安装教程，所以这里不赘述了。这里提供一个编译 `Opencv` 程序的脚本:
```
echo "compiling $1"
if [[ $1 == *.c ]]
then
    gcc -g `pkg-config --cflags opencv`  -o `dirname $1`/`basename $1 .c` $1 `pkg-config --libs opencv`;
elif [[ $1 == *.cpp ]]
then
    g++ -g `pkg-config --cflags opencv` -std=c++11 -std=gnu++11 -o `dirname $1`/`basename $1 .cpp` $1 `pkg-config --libs opencv`;
else
    echo "Please compile only .c or .cpp files"
fi
echo "Output file => ${1%.*}"
```
讲上述代码保存为一个 `xxx.sh` 文件。然后在终端里给该文件开启可执行权限 `sudo chmod 777 xxx.sh`
接下来在 `.bashrc` 中建立一个 `alias` 来指向 `xxx.sh`
```
subl ~/.bashrc
```
在 `.bashrc` 中键入:
```
alias opencv="/path/to/xx.sh"
```
以后要编译 `opencv` 程序的时候，只需要在终端里输入 `opencv xxx.cpp` 即可。无需在敲入繁琐的 `pkg-config` 前后缀。例如, 直接在终端里键入 `opencv` 命令，会提示:
```
compiling 
Please compile only .c or .cpp files
Output file => 
```

----------------------

## 八、安装 Matlab
安装过程略过。这里说下安装后做的事:
提供两种方法实现在 `terminal` 中启动 `matlab`
1) 讲 `matlab` 的可执行程序加入到系统的环境变量中
export PATH="/path/to/matlab"

2) 与 `opencv` 相同，在 `.bashrc` 中建立一个 `matlab` alias
```
alias matlab='/path/to/matlab'
```
这样做的优势是，可让 `matlab` 调用一些指定的库文件。因为，`matlab` 在启动时，会优先读取自带的 `opencv` 库，而不读取系统中安装好的 `opencv 2.4.11` 库。
在这种情况下做 `RCNN` 的实验，就可能遇到问题。所以我在 `Mac OSX 10.10` 下，做了如下 alias，`ubuntu` 下可以照葫芦画瓢
```
alias rcnn="DYLD_INSERT_LIBRARIES=/usr/local/lib/libopencv_highgui.2.4.dylib:/usr/local/lib/libtiff.5.dylib /Applications/MATLAB_R2014b.app/bin/matlab"
```

以后要做 `Rcnn` 实验的时候，只需要在终端里输入 `rcnn` 就可以启动 `matlab` 并优先读取自己安装的 `opencv 2.4.11` 库。
----------------------------

## 九、安装 Caffe
待续


