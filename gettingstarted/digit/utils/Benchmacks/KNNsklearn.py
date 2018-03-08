__author__ = 'Abner'
import csv
import numpy as np

def loadTrainData():
    l = []
    with open('../Data/train.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)
    l.remove(l[0])
    l = np.array(l)
    label = l[:,0]
    data = l[:,1:]
    return nomalizing(toInt(data)),toInt(label)

def toInt(array):
    array = np.mat(array)
    m,n = np.shape(array)
    newArray = np.zeros((m,n))
    for i in xrange(m):
        for j in xrange(n):
            newArray[i,j] = int(array[i,j])
    return newArray

def nomalizing(array):
    m,n = np.shape(array)
    for i in xrange(m):
        for j in xrange(n):
            if array[i,j]!=0:
                array[i,j]=1
    return array

def loadTestData():
    l = []

    with open('../Data/test.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)
    l.remove(l[0])
    data=np.array(l)
    return nomalizing(toInt(data))

from sklearn.neighbors import KNeighborsClassifier
def knnClassify(trainData,trainLabel,testData):
    print 'Training...'
    knnclf = KNeighborsClassifier()
    knnclf.fit(trainData,np.ravel(trainLabel))
    print 'Predicting...'
    testLabel = knnclf.predict(testData)

    testid = []
    for i in xrange(testData):
        testid.append(i+1)

    print 'Printing...'
    predictions_file = open("../Submissions/knn_summission.csv", "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["ImageId","Label"])
    open_file_object.writerows(zip(testid,testLabel))
    predictions_file.close()

from sklearn import svm
def svcClassify(trainData,trainLabel,testData):
    svcclf = svm.SVC(C=5.0)
    svcclf.fit(trainData,np.ravel(trainLabel))
    testLabel = svcclf.predict(testData)

    testid = []
    for i in xrange(testData):
        testid.append(i+1)

    predictions_file = open("../Submissions/svm_summission.csv", "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["ImageId","Label"])
    open_file_object.writerows(zip(testid,testLabel))
    predictions_file.close()


from sklearn.naive_bayes import GaussianNB
def GaussianNBClassify(trainData,trainLabel,testData):
    nbClf = GaussianNB()
    nbClf.fit(trainData,np.ravel(trainLabel))
    testLabel = nbClf.predict(testData)

    testid = []
    for i in xrange(testData):
        testid.append(i+1)

    predictions_file = open("../Submissions/GauNB_summission.csv", "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["ImageId","Label"])
    open_file_object.writerows(zip(testid,testLabel))
    predictions_file.close()

if __name__ == '__main__':
    trainData,trainLabel = loadTrainData()
    testData = loadTestData()

    knnClassify(trainData,trainLabel,testData)