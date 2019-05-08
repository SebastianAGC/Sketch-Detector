from NeuralNetwork import *
from sklearn.model_selection import train_test_split
import numpy as np
import time

path = "/Users/sebas/Documents/UVG/2019/Primer Semestre/Inteligencia Artificial/Proyecto2/Sketch-Detector/Sketch/pending"

print("Loading data...")
circles = np.load("/Users/sebas/Documents/UVG/2019/Primer Semestre/Inteligencia Artificial/Proyecto2/data/circle.npy")
egg = np.load("/Users/sebas/Documents/UVG/2019/Primer Semestre/Inteligencia Artificial/Proyecto2/data/egg.npy")
face = np.load("/Users/sebas/Documents/UVG/2019/Primer Semestre/Inteligencia Artificial/Proyecto2/data/face.npy")
house = np.load("/Users/sebas/Documents/UVG/2019/Primer Semestre/Inteligencia Artificial/Proyecto2/data/house.npy")
mickey = np.load("/Users/sebas/Documents/UVG/2019/Primer Semestre/Inteligencia Artificial/Proyecto2/data/mickey.npy")
question = np.load("/Users/sebas/Documents/UVG/2019/Primer Semestre/Inteligencia Artificial/Proyecto2/data/question.npy")
sad = np.load("/Users/sebas/Documents/UVG/2019/Primer Semestre/Inteligencia Artificial/Proyecto2/data/sad.npy")
square = np.load("/Users/sebas/Documents/UVG/2019/Primer Semestre/Inteligencia Artificial/Proyecto2/data/square.npy")
tree = np.load("/Users/sebas/Documents/UVG/2019/Primer Semestre/Inteligencia Artificial/Proyecto2/data/tree.npy")
triangle = np.load("/Users/sebas/Documents/UVG/2019/Primer Semestre/Inteligencia Artificial/Proyecto2/data/triangle.npy")

circles_y  = np.array([1,0,0,0,0,0,0,0,0,0])
egg_y      = np.array([0,1,0,0,0,0,0,0,0,0])
face_y     = np.array([0,0,1,0,0,0,0,0,0,0])
house_y    = np.array([0,0,0,1,0,0,0,0,0,0])
mickey_y   = np.array([0,0,0,0,1,0,0,0,0,0])
question_y = np.array([0,0,0,0,0,1,0,0,0,0])
sad_y      = np.array([0,0,0,0,0,0,1,0,0,0])
square_y   = np.array([0,0,0,0,0,0,0,1,0,0])
tree_y     = np.array([0,0,0,0,0,0,0,0,1,0])
triangle_y = np.array([0,0,0,0,0,0,0,0,0,1])

data = []

for item in circles:
    data.append((circles_y, item))

for item in egg:
    data.append((egg_y, item))

for item in face:
    data.append((face_y, item))

for item in house:
    data.append((house_y, item))

for item in mickey:
    data.append((mickey_y, item))

for item in question:
    data.append((question_y, item))

for item in sad:
    data.append((sad_y, item))

for item in square:
    data.append((square_y, item))

for item in tree:
    data.append((tree_y, item))

for item in triangle:
    data.append((triangle_y, item))


data_train, data_test = train_test_split(data, test_size=0.20, random_state=42)
print("Data loaded.")
print("Starting training...")

start = time.time()
cont = 0
while cont < len(data_train):
    nn = NeuralNetwork(np.array(data_train[cont][1]), data_train[cont][0])
    nn.feedforward()
    nn.backprop()
    if(cont == len(data_train)-1):
        np.save("Weights/weights1", nn.weights1)
        np.save("Weights/weights2", nn.weights2)
    cont = cont + 1

print("Training complete.")
end = time.time()
print("Finished in ", end - start, " seconds.")
