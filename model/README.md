# 这是一个基于keras原生数据集的图像识别测试


## 环境

预先准备： Windows环境、conda命令可用(自行查阅网络文档)

为了避免环境冲突，我的建议是新建一个python环境（此处命名为tensorflow）

创建新的python环境：`conda create -n tensorflow python=3.7`

使用该环境：`conda activate tensorflow`

 其余按照`setup.txt`依次安装最新版本

## 快速开始

使用` python load_trainning.py `（注意需要用conda先激活tensorflow环境，且注意文件目录）

或者以任何形式导入`image.ipynb`到 jupyter 使用jupyter运行（到网上查阅jupyter-python环境配置）

## 模型预测

（前置条件，拥有一个已经训练好的模型（.h5文件））

运行`python predict.py`  

按照文件中注释选择想要预测的图片（试着自己拍一张？）

## 已经训练的网络适用

共十类：

`  'airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck' `

## 继续训练和迁移训练

模型严格遵循keras标准，可用查询keras官方文档对模型结构进行修改

`Self-train.py` 中有一个简单的示例

## 神经网络结构详细解释

懒懒的作者决定之后再写

## WHAT'S MORE?

这是一份不完善的文档，后续将继续修改

-- ALICE