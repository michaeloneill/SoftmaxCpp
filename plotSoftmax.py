import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

# plot the first 100 images 

file = open("outputDigits", "r")

images = np.loadtxt(file).reshape(100, 28, 28)

fig = plt.figure("First 100 digits")

for x in range(10):
    for y in range(10):
        ax = fig.add_subplot(10, 10, 10*x+y+1) # filled column-wise
        ax.matshow(images[10*x+y], cmap = cm.binary)
        plt.xticks([])
        plt.yticks([])
plt.show()

file.close()

# plot the cost histories

file = open("outputCostHistory", "r")

costHistory = np.loadtxt(file)

fig = plt.figure("Cost History")

plt.plot(costHistory)
plt.xlabel("# iterations")
plt.ylabel("cost")
plt.show()

file.close()

# plot the learning curves

file = open("outputLogisticLC", "r")

data = np.loadtxt(file)

plt.figure('learning curves')
plt.plot(data[:, 0], data[:, 1], 'r', label='training score')
plt.plot(data[:, 0], data[:, 2], 'b', label='validation score')
plt.xlabel('proportion training samples used')
plt.ylabel('score')
plt.legend()
plt.show()

file.close()

#plot validation curve for Lambda

file = open("outputLogisticLamVal", "r")

data = np.loadtxt(file)

plt.figure('Lambda validation')
plt.plot(data[:, 0], data[:, 1], 'r', label='training score')
plt.plot(data[:, 0], data[:, 2], 'b', label='validation score')
plt.xlabel('lambda')
plt.ylabel('score')
plt.legend()
plt.show()

file.close()

file = open("outputLogisticAlphaVal", "r")

data = np.loadtxt(file)

plt.figure('Alpha Validation Curves')
plt.plot(data[:, 0], data[:, 1], 'r', label='training score')
plt.plot(data[:, 0], data[:, 2], 'b', label='validation score')
plt.xlabel('alpha')
plt.ylabel('score')
plt.legend()
plt.show()

file.close()



