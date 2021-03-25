import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
from PIL import Image
from numpy import asarray


def create_dataset(img_folder):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    i = 0
    max_train = 40000
    for file in os.listdir(img_folder):
        image_path = os.path.join(img_folder, file)
        img = Image.open(image_path)
        number = file.split('.jpg')[0]
        if i < max_train:
            x_train.append(asarray(img))
            y_train.append(asarray(number[-1]))
        else:
            x_test.append(asarray(img))
            y_test.append(asarray(number[-1]))
        i += 1
    return (asarray(x_train), asarray(y_train)), \
           (asarray(x_test), asarray(y_test))


(x_train, y_train), (x_test, y_test) = create_dataset('train1')
# the data, split between train and test sets
num_classes = 10

print(x_train.shape, y_train.shape)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

batch_size = 128
num_classes = 10
epochs = 10

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])

hist = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
print("The model has successfully trained")

model.save('breakathon.h5')
print("Saving the model as breakathon.h5")
