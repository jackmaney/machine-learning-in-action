import numpy as np
import operator


def createBasicDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ["A", "A", "B", "B"]
    return group, labels


def euclideanDistances(inputArr, dataSet):
    return np.linalg.norm(dataSet - inputArr, axis=1)


def classify0(inputArr, dataSet, labels, k, distanceFunction=euclideanDistances):
    distances = distanceFunction(inputArr, dataSet)

    sortedDistIndices = distances.argsort()[0:k]
    classCount = {}

    for index in sortedDistIndices:
        classCount[labels[index]] = classCount.get(labels[index], 0) + 1

    sortedClassCount = sorted(classCount.iteritems(),
                              key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]
