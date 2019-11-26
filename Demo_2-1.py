from tensorflow.python.keras.datasets import mnist
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()
# print(train_images.shape)
# print(train_labels.shape)
# print(test_images.shape)
# print(test_labels.shape)
# print(train_labels)
# digit = train_images[8,1:14,1:14]       #实现图片的裁剪
digit = train_images[8]
import matplotlib.pyplot as plt
plt.imshow(digit,cmap=plt.cm.binary)
plt.show()