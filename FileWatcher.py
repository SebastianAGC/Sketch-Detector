import os
import shutil
from PIL import Image
from NeuralNetwork import *
import time

path = "/Users/sebas/Documents/UVG/2019/Primer Semestre/Inteligencia Artificial/Proyecto2/Sketch-Detector/Sketch"
cats = {
    0: 'Circle',
    1: 'Egg',
    2: 'Face',
    3: 'House',
    4: 'Mickey',
    5: 'Question',
    6: 'Sad',
    7: 'Square',
    8: 'Tree',
    9: 'Triangle'
}

while True:
    if os.path.exists(path):
        if os.path.isfile(path + "/pending/sketch.bmp"):
            im = Image.open(os.path.join(path+"/pending/", 'sketch.bmp')).convert('I')
            p = np.array(im)
            p = 255 - p
            img = []
            for item in p:
                for i in item:
                    img.append(i)
            img = np.array(img)
            img = img / 255.0
            #shutil.move(path+'/pending/sketch.bmp', path+'/analyzed/sketch.bmp')
            nn = NeuralNetwork()
            data = np.load("Weights/nn.npy", allow_pickle=True)
            nn.weights1 = data[0]
            nn.weights2 = data[1]
            nn.bias1 = data[2]
            nn.bias2 = data[3]
            res = nn.feedforward(img.reshape(784, 1))
            res1 = np.argmax(res)

            sort = np.argsort(res.reshape(1,10))[0]

            print("1.", cats[sort[-1]], " Accuracy: ", res[sort[-1]]*100, ". 2.", cats[sort[-2]], " Accuracy: ", res[sort[-2]]*100, ". 3.", cats[sort[-3]], " Accuracy: ", res[sort[-3]]*100)
            #print("Resultado: ", cats[res1], ". Accuracy: ", res[res1])
            time.sleep(1)


