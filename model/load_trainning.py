import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

# 下载数据
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# 将像素的值标准化至0到1的区间内。
train_images, test_images = train_images / 255.0, test_images / 255.0

# 数据验证
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 构建卷积神经网络模型 (CNN，使用Keras.model进行快速构建)
model = models.Sequential()
# 卷积层1, 使用relu激活函数，输入类型为 width=32， height=32，通道数 = 3
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# 池化层1，使用maxpool方法
model.add(layers.MaxPooling2D((2, 2)))
# 卷积层2
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# 池化层2
model.add(layers.MaxPooling2D((2, 2)))
# 卷积层3
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# 过渡层，数据一维化
model.add(layers.Flatten())
# 全连接层
model.add(layers.Dense(64, activation='relu'))
# 输出层
model.add(layers.Dense(10))

# 训练模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

# 评估
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)

# 保存模型
model.save("model_image.h5")

# 预测
y = model.predict(test_images)

# 查看预测结果(test_image真值为 'frog' 在ipynb文件中可以展示)
print(class_names[np.argmax(y[0])])