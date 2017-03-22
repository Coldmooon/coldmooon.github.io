---
layout:     post
title:      "在 fb.resnet.torch 中使用自己的数据集"
subtitle:   "Train on custorm datasets in fb.resnet.torch"
date:       2017-03-20 15:36:00
author:     "Coldmooon"
header-img: "img/home-bg.jpg"
---

## 一、前言
本文介绍如何在 `fb.resnet.torch` 中使用自己的数据集。方法有两种：

1) 直接读取图片: [Fine-tuning on a custom dataset](https://github.com/facebook/fb.resnet.torch/blob/master/pretrained/README.md#fine-tuning-on-a-custom-dataset)

> Your images don't need to be pre-processed or packaged in a database, but you need to arrange them so that your dataset contains a train and a val directory, which each contain sub-directories for every label. For example:
>
```
train/<label1>/<image.jpg>
train/<label2>/<image.jpg>
val/<label1>/<image.jpg>
val/<label2>/<image.jpg>
```
> You can then use the included ImageNet data loader with your dataset and train with the -resetClassifer and -nClasses options:

2) 把数据集制作成 `*.t7` 文件。`fb.resnet.torch` 中的 CIFAR 例子就使用 `t7` 文件。

本文介绍第二种方法。

## 二、生成 t7 数据集文件
将数据集存储为 `fb.resnet.torch` 识别的 `t7` 文件非常简单。只要将 图片一次性全部读取到一个变量中，然后保存这个变量即可。变量的结构为:

```
th> a = torch.load('mnist-new.t7')
th> print(a)
{
  train : 
    {
      data : DoubleTensor - size: 12000x1x28x28
      labels : DoubleTensor - size: 12000x1
    }
  val : 
    {
      data : DoubleTensor - size: 50000x1x28x28
      labels : DoubleTensor - size: 50000x1
    }
}
```

举个例子: 

```
-- 训练集
train_img = torch.rand(12000,1,28,28)
label_train = torch.ones(12000)

-- 测试集
test_img = torch.rand(50000,1,28,28)
label_test = torch.ones(50000)

trainData = { data = train_img, labels = label_train }
valData = { data = test_img, labels = label_test}

torch.save('./dataset.t7', { train = trainData, val = valData })
```
这样就把我们的图片存储为可被 `fb.resnet.torch` 识别的 `t7` 文件了。然后可以把刚才生成的数据集读进来，看看格式是否正确。

```
dataset = torch.load('./dataset.t7')
print(dataset)
```

## 三、修改 fb.resnet.torch 的代码

因为新的数据集是 `28 * 28 * 1` 大小的黑白图片；而默认的 `CIFAR` 数据集是 `32 * 32 * 3` 的彩色图片。所以，部分代码 (例如数据增广) 需要简单修改。具体操作如下:

#### 为新数据集增加命令行选项

以 `mnist-rot-12k` 数据集为例。编辑 `opts.lua` 文件，在 `-dataset` 字段中增加新的数据集名称 `mnist-rot-12k`: 

```
- cmd:option('-dataset', 'imagenet', 'Options: imagenet | cifar10 | cifar100')
+ cmd:option('-dataset', 'imagenet', 'Options: imagenet | cifar10 | cifar100 | mnist-rot-12k')
```
然后，为该数据集设定一些默认的参数: 

```
...
elseif opt.dataset == 'cifar100' then
    -- Default shortcutType = A and nEpochs=164
    opt.shortcutType = opt.shortcutType == '' and 'A' or opt.shortcutType
    opt.nEpochs = opt.nEpochs == 0 and 164 or opt.nEpochs
    
+ elseif opt.dataset == 'mnist-rot-12k' then
+     opt.shortcutType = opt.shortcutType == '' and 'B' or opt.shortcutType
+     opt.nEpochs = opt.nEpochs == 0 and 164 or opt.nEpochs

  else
     cmd:error('unknown dataset: ' .. opt.dataset)
  end
``` 

#### 编写 get(), size(), preprocess() 函数

根据官方指南，需要为新数据集增加三个函数 `get(), size(), preprocess()`，[见此](https://github.com/facebook/fb.resnet.torch/tree/master/datasets#datasetlua)。方便起见，我直接复制 `dataset/cifar10.lua` 的代码。**然后把所有 `Cifar` 字段替换为 `Mnist`，并修改数据增广的方式。其他部分无需修改。**

```
local t = require 'datasets/transforms'

local M = {}
local MnistDataset = torch.class('resnet.MnistDataset', M)

function MnistDataset:__init(imageInfo, opt, split)
   assert(imageInfo[split], split)
   self.imageInfo = imageInfo[split]
   self.split = split
end

function MnistDataset:get(i)
   local image = self.imageInfo.data[i]:float()
   local label = self.imageInfo.labels[i]

   return {
      input = image,
      target = label,
   }
end

function MnistDataset:size()
   return self.imageInfo.data:size(1)
end

-- Computed from entire MNIST training set
local meanstd = {
   mean = {125.3, 123.0, 113.9},
   std  = {63.0,  62.1,  66.7},
}

function MnistDataset:preprocess()
   if self.split == 'train' then
      return t.Compose{
         -- 这里把减均值去掉了，因为减均值代码是给 3 通道设计的。
         -- 如果要减均值，必须修改 transforms.lua 中的代码。
         -- t.ColorNormalize(meanstd),
         t.HorizontalFlip(0.5),
         t.RandomCrop(32, 4),
      }
   elseif self.split == 'val' then
      return t.Compose{
         -- 同样去掉了减均值。顺手加了另外两种增广。
         -- t.ColorNormalize(meanstd),
         t.HorizontalFlip(0.5),
         t.RandomCrop(32, 4),
      }
   else
      error('invalid split: ' .. self.split)
   end
end

return M.MnistDataset
```

#### 修改数据增广代码，使其可以处理单通道图像。

编辑 `datasets/transforms.lua`，找到随机剪裁函数，`function M.RandomCrop(size, padding)`: 

```
- local temp = input.new(3, input:size(2) + 2*padding, input:size(3) + 2*padding)
+ local temp = input.new(1, input:size(2) + 2*padding, input:size(3) + 2*padding)
```

然后修改减均值代码, 把 `for = 1,3` 循环去掉:

```
function M.ColorNormalize(meanstd)
   return function(img)
      img = img:clone()
      // 去掉循环即可，这里我就不改了，因为我生成数据集的时候就已经减均值了。
      for i=1,3 do
         img[i]:add(-meanstd.mean[i])
         img[i]:div(meanstd.std[i])
      end
      return img
   end
end
```

至此，新数据集添加完毕。数据增广部分需要跟自己的数据集酌情处理。

## 四、运行

稍微修改下 `resnet.lua` 网络结构，让其可以跑单通道图像。简单起见，直接修改 `cifar10` 部分的代码。

```
...
- elseif opt.dataset == 'cifar10' then
+ elseif opt.dataset == 'cifar10' or 'mnist-rot-12k' then

-- The ResNet CIFAR-10 model
-  model:add(Convolution(3,16,3,3,1,1,1,1))
+  model:add(Convolution(1,16,3,3,1,1,1,1))
...
```

然后在命令行中输入下面的命令就能训练了。

```
th main.lua -dataset mnist-rot-12k -nGPU 2 -batchSize 128 -depth 20
```