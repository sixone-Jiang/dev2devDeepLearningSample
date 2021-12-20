import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

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

# 下载数据
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# 建议使用打乱数据集顺序

# 将像素的值标准化至0到1的区间内。
train_images, test_images = train_images / 255.0, test_images / 255.0

# 数据标签
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 读取模型, 请自行导入同类框架的模型进行继续训练(或作为训练前置)
my_load_model = models.load_model("model_image.h5")

# 训练模型
my_load_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = my_load_model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

test_loss, test_acc = my_load_model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)

# 保存模型
my_load_model.save("model2.h5")

# 评估
test_loss, test_acc = my_load_model.evaluate(test_images,  test_labels, verbose=2)

# 预测
y = my_load_model.predict(test_images)

# 查看预测结果(test_images[0]真值为 'frog' 在ipynb文件中可以展示)
print(class_names[np.argmax(y[0])])