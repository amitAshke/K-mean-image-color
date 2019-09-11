import load
import init_centroids
import numpy
import matplotlib.pyplot as plt

def lossCalculation(color1, color2):
    return numpy.linalg.norm(color1 - color2) ** 2

def closestColor(centroids, pixelColor):
    minLoss = lossCalculation(centroids[0], pixelColor)
    selectedColor = centroids[0]
    for index in range(1, len(centroids)):
        newLoss = lossCalculation(centroids[index], pixelColor)
        if minLoss > newLoss:
            minLoss = newLoss
            selectedColor = centroids[index]
    return selectedColor

def smallestLoss(centroids, pixelColor):
    minLoss = lossCalculation(centroids[0], pixelColor)
    for index in range(1, len(centroids)):
        newLoss = lossCalculation(centroids[index], pixelColor)
        if minLoss > newLoss:
            minLoss = newLoss
    return minLoss

def updateData(centroids, pixelColor, algorithmData):
    closest = closestColor(centroids, pixelColor)
    for index in range(len(centroids)):
        if closest[0] == algorithmData[index][0][0] and closest[1] == algorithmData[index][0][1] and closest[2] == algorithmData[index][0][2]:
            algorithmData[index][1] += 1
            algorithmData[index][2] += pixelColor

def updateCentroids(centroids, algorithmData):
    for index in range(len(centroids)):
        for i in range(3):
            if algorithmData[index][1] != 0:
                centroids[index][i] = algorithmData[index][2][i] / algorithmData[index][1]

def printData(centroids, algorithmData, k):
    # averageLoss = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for iteration in range(0, 11):

        print("iter ", iteration, sep="", end=": ")
        print(*centroids, sep=", ", end="\n")

        # averageLoss[iteration] = 0

        for row in range(0, load.img_size[0]):
            for col in range(0, load.img_size[1]):
                updateData(centroids, load.A_norm[row][col], algorithmData)
        #         averageLoss[iteration] += smallestLoss(centroids, load.A_norm[row][col])
        #
        # averageLoss[iteration] = averageLoss[iteration] / (load.img_size[0] * load.img_size[1])
        # print("average loss: ", averageLoss[iteration], sep="", end="\n")

        if iteration == 10:
            break

        updateCentroids(centroids, algorithmData)

    # plt.figure()
    # plt.plot(averageLoss)
    # plt.title("k = " + str(k))
    # plt.ylabel("avaragel loss")
    # plt.xlabel("iteration")
    # plt.show()

numpy.set_printoptions(precision=2)

for k in [2, 4, 8, 16]:
    centroids = init_centroids.init_centroids(load.A_norm, k)
    algorithmData = []

    for centroid in centroids:
        algorithmData.append([centroid, 0, [0, 0, 0]])

    print("k=", k, sep="", end=":\n")
    printData(centroids, algorithmData, k)

    imageCopy = load.A_norm
    if (k == 16):
        for row in range(0, load.img_size[0]):
            for col in range(0, load.img_size[1]):
                imageCopy[row][col] = closestColor(centroids, load.A_norm[row][col])
        load.plt.imshow(imageCopy)
        load.plt.grid(False)
        load.plt.show()