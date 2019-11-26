#加载数据集
from tensorflow.python.keras.datasets import reuters
data_path = 'D:\\data\\reuters.npz'
(train_data,train_labels),(test_data,test_labels) = reuters.load_data(path=data_path,num_words=10000)
print(train_data.shape)
#将索引解码为新闻文本
word_index = reuters.get_word_index()
reverse_word_index = dict([(value,key)for (key,value) in word_index.items()])
decode_newswise = ' '.join([reverse_word_index.get(i-3,'?')for i in train_data[0]])
# print(decode_newswise)

#编码数据
import  numpy as ny
def vectorize_sequences(sequences ,dimension = 10000):
    results = ny.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence] = 1.
    return results

#将训练数据向量化
x_train = vectorize_sequences(train_data)
#将测试数据向量化
x_test = vectorize_sequences(test_data)

#将标签数据向量化
def to_one_hot(labels,dimension = 46):
    results = ny.zeros((len(labels),dimension))
    for i ,label in enumerate(labels):
        results[i,label] = 1.
    return  results

#将训练标签向量化
one_hot_train_labels = to_one_hot(train_labels)
#将测试标签向量化
one_hot_test_labels = to_one_hot(test_labels)

#下面使用keras中内置的方法来向量化训练标签和测试标签
from tensorflow.python.keras.utils.np_utils import to_categorical
one_hot_train_labels_1 = to_categorical(train_labels)
one_hot_test_labels_1 = to_categorical(test_labels)
# print("下面是自定义函数和keras内置方法对训练标签向量化和测试标签向量化的判断")
# print('训练标签向量化：',one_hot_train_labels == one_hot_train_labels_1)
# 以上结果：[[ True  True  True ...  True  True  True]
# print('测试标签向量化：',one_hot_test_labels == one_hot_test_labels_1)
#以上结果：[[ True  True  True ...  True  True  True]

#留出验证集
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

#构建网络
from tensorflow.python.keras import layers,models
model = models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(46,activation='softmax'))

#编译模型
model.compile(
    optimizer='rmsprop',
    loss= 'categorical_crossentropy',
    metrics=['acc']
)

#训练模型
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=15,
    batch_size=512,
    validation_data=(x_val,y_val)
)

results = model.evaluate(x_test,one_hot_test_labels)
print(results)