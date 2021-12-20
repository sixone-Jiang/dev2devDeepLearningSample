# 端到端的图像识别样例(Flask)

## 环境

* tensorflow2.7
* pillow
* Flask
* numpy、pandas、requests

ps： 该测试使用了GPU、如果不需要请设置model里的GPU调用为false

## 快速开始

* 服务端 

  ```bash
  python flasker.py
  ```

* 客户端

  请求发起代码在client中、按照Jupyter中代码顺序运行即可

## 神经网络模型

* 简介： 使用了cifar10数据集合，模型由 3层卷积（Conv2D）、两层池化（maxpooling）、一层全连接构成

* 模型代码放在model/ 下，拥有独立的README文档

* model/predict中代码是为Flask准备的接口

* 模型可预测标签共十类：

  ```
  'airplane', 'automobile', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck'
  ```

## 特别注意

* 如果没有GPU使用需求请将代码中含有

  ```python
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
      # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
      try:
          tf.config.experimental.set_virtual_device_configuration(
              gpus[0],
              [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*2)])
          logical_gpus = tf.config.experimental.list_logical_devices('GPU')
          print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
          print(e)
  ```

  替换为:

  ```python
  tf.device('/CPU:0')
  ```

  

