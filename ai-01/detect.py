from keras.models import load_model
import os
from keras.preprocessing.image import load_img
import numpy as np

model = load_model('breakathon.h5')
results = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0,
           '6': 0, '7': 0, '8': 0, '9': 0}

def predict_digit(img):
    #resize image to 28x28 pixels
    img = img.resize((28,28))
    #convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)
    #reshaping to support our model input and normalizing
    img = img.reshape(1,28,28,1)
    img = img/255.0
    #predicting the class
    res = model.predict([img])[0]
    return np.argmax(res), max(res)

img_folder = 'test1'

for file in os.listdir(img_folder):
    image_path = os.path.join(img_folder, file)
    img = load_img(image_path)
    digit, acc = predict_digit(img)
    results[str(digit)] = results[str(digit)] + 1

res = ""
for i in results.keys():
    res += str(results[i]) + '-'

print(res[:len(res)-1])
