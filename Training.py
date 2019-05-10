from NeuralNetwork import *
from sklearn.model_selection import train_test_split
import numpy as np
import time

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

circles_y  = np.array([1,0,0,0,0,0,0,0,0,0]).reshape(10, 1)
egg_y      = np.array([0,1,0,0,0,0,0,0,0,0]).reshape(10, 1)
face_y     = np.array([0,0,1,0,0,0,0,0,0,0]).reshape(10, 1)
house_y    = np.array([0,0,0,1,0,0,0,0,0,0]).reshape(10, 1)
mickey_y   = np.array([0,0,0,0,1,0,0,0,0,0]).reshape(10, 1)
question_y = np.array([0,0,0,0,0,1,0,0,0,0]).reshape(10, 1)
sad_y      = np.array([0,0,0,0,0,0,1,0,0,0]).reshape(10, 1)
square_y   = np.array([0,0,0,0,0,0,0,1,0,0]).reshape(10, 1)
tree_y     = np.array([0,0,0,0,0,0,0,0,1,0]).reshape(10, 1)
triangle_y = np.array([0,0,0,0,0,0,0,0,0,1]).reshape(10, 1)

data = []
i = 0
amount = 8000

while i < amount and i < len(circles):
    data.append(((circles[i] / 255).reshape(784, 1), circles_y))
    i += 1
i = 0
while i < amount and i < len(egg):
    data.append(((egg[i] / 255.0).reshape(784, 1), egg_y))
    i += 1
i = 0
while i < amount and i < len(face):
    data.append(((face[i] / 255.0).reshape(784, 1), face_y))
    i += 1
i = 0
while i < amount and i < len(house):
    data.append(((house[i] / 255.0).reshape(784, 1), house_y))
    i += 1
i = 0
while i < amount and i < len(mickey):
    data.append(((mickey[i] / 255.0).reshape(784, 1), mickey_y))
    i += 1
i = 0
while i < amount and i < len(question):
    data.append(((question[i] / 255.0).reshape(784, 1), question_y))
    i += 1
i = 0
while i < amount and i < len(sad):
    data.append(((sad[i] / 255.0).reshape(784, 1), sad_y))
    i += 1
i = 0
while i < amount and i < len(square):
    data.append(((square[i] / 255.0).reshape(784, 1), square_y))
    i += 1
i = 0
while i < amount and i < len(tree):
    data.append(((tree[i] / 255.0).reshape(784, 1), tree_y))
    i += 1
i = 0
while i < amount and i < len(triangle):
    data.append(((triangle[i] / 255.0).reshape(784, 1), triangle_y))
    i += 1
print("Data loaded.")


####Training
print("Starting training...")
data_train, data_test = train_test_split(data, test_size=0.20, random_state=42)
data_cross, data_test = train_test_split(data_test, test_size=0.50, random_state=42)
net = NeuralNetwork()
net.gradientDescent(data_train, data_cross, data_test, 3.0, 100)
data = [
    net.weights1,
    net.weights2,
    net.bias1,
    net.bias2
]
print("Saving train")
np.save("Weights/nn.npy", data)
print("Training complete.")





