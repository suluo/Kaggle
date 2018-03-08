__author__ = 'Abner'

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import csv
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

def nomalizing(array):
    m,n = np.shape(array)
    for i in xrange(m):
        for j in xrange(n):
            if array[i,j]!=0:
                array[i,j]=1
    return array

#after nomalizing :0.94257 no nomalizing:0.94457
def main():
    train_df = pd.read_csv('../Data/train.csv',header=0)
    trainlabel = train_df.ix[:,0].values
    traindata = train_df.ix[:,1:].values
    test_df = pd.read_csv('../Data/test.csv',header=0)
    testdata = test_df.values
    testid = test_df.index.astype(int)

    print 'Training...'
    rfClf = RandomForestClassifier()
    #rfClf = KNeighborsClassifier()
    #rfClf = svm.SVC(C=5.0)
    rfClf = rfClf.fit(traindata,np.ravel(trainlabel))

    print 'Predicting...'
    testlabel = rfClf.predict(testdata).astype(int)

    print 'Printing...'
    predictions_file = open("../Submissions/svm_summission.csv", "wb")
    output_file_object = csv.writer(predictions_file)
    output_file_object.writerow(["ImageId","Label"])
    output_file_object.writerows(zip(testid+1,testlabel))
    predictions_file.close()


if __name__ == '__main__':
    main()

