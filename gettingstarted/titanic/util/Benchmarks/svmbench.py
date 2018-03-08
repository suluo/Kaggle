__author__ = 'Abner'

import pandas as pd
import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

def data_pretreat(filename):

    df = pd.read_csv(filename,header=0)

    df['Gender'] = df['Sex'].map({'female':0,'male':1}).astype(int)

    if len(df.Embarked[df.Embarked.isnull()])>0:
        df.Embarked[df.Embarked.isnull()] = df.Embarked.dropna().mode().values
    Ports = list(sorted(enumerate(np.unique(df['Embarked']))))
    Ports_dict = {name:i for i,name in Ports}
    df.Embarked = df.Embarked.map(lambda x:Ports_dict[x]).astype(int)

    median_age = df['Age'].dropna().median()
    if len(df.Age[df.Age.isnull()])>0:
        df.loc[(df.Age.isnull()),'Age'] = median_age

    if len(df.Fare[df.Fare.isnull()])>0:
        median_fare = np.zeros(3)
        for f in range(3):
            median_fare[f] = df[df.Pclass == f+1]['Fare'].dropna().median()
        for f in range(3):
            df.loc[(df.Fare.isnull())&(df.Pclass == f+1),'Fare'] = median_fare[f]

    ids = df['PassengerId'].values

    df = df.drop(['Name','Sex','Ticket','Cabin','PassengerId'],axis=1)
    data = df.values

    #print df.dtypes
    #print df.info()

    return ids,data


def main():
    train_id,train_data = data_pretreat('../Data/train.csv')
    test_id,test_data = data_pretreat('../Data/test.csv')

    print 'Training...'
    svmmodel = RandomForestClassifier(n_estimators = 100)
    svmmodel = svmmodel.fit(train_data[0:,1:],train_data[0:,0])

    print 'Predicting...'
    output = svmmodel.predict(test_data).astype(int)

    predictions_file = open("../Submissions/svmsummission1.csv", "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["PassengerId","Survived"])
    open_file_object.writerows(zip(test_id, output))
    predictions_file.close()

if __name__ == '__main__':
    main()











