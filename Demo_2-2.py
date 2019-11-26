from tensorflow.python.keras import models,layers,optimizers
network = models.Sequential()
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10,activation='softmax'))
network.compile(optimizer = optimizers.RMSprop(lr=0.001),
               loss = 'mse',
               metrics = ['accuracy'])
