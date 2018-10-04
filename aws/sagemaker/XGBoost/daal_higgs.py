from daal.algorithms import gbt
from daal.algorithms.gbt.classification import prediction, training
from daal.algorithms import classifier

from daal.data_management import (
    FileDataSource, DataSourceIface, NumericTableIface, HomogenNumericTable,
    MergedNumericTable, BlockDescriptor, readOnly
)

import pandas as pd
import numpy as np
import time

dataset = pd.read_csv('https://s3.amazonaws.com/h2o-public-test-data/bigdata/laptop/higgs_head_2M.csv',header=None)
print(dataset.head())

train = dataset.head(1000000)
test = dataset.tail(1000000)

X_train = train.drop(0,axis=1)
X_test = test.drop(0,axis=1)

Y_train = train[0]
Y_test = test[0]

trainNT = HomogenNumericTable(X_train)
trainGroungTruth = HomogenNumericTable(np.reshape(Y_train.values,(-1,1)))

testNT = HomogenNumericTable(X_test)
testGroungTruth = HomogenNumericTable(np.reshape(Y_test.values,(-1,1)))

algorithm = training.Batch(2)
algorithm.parameter().maxIterations = 50
algorithm.parameter().shrinkage = 0.3
algorithm.parameter().maxTreeDepth = 6
algorithm.parameter().minObservationsInLeafNode = 1
algorithm.input.set(classifier.training.data, trainNT)
algorithm.input.set(classifier.training.labels, trainGroungTruth)

start = time.time()
trainingResult = algorithm.compute()
end = time.time()
print(end - start, 'elapsed')

model = trainingResult.get(classifier.training.model)

test_algorithm = prediction.Batch(2)
test_algorithm.input.setTable(classifier.prediction.data,  testNT)
test_algorithm.input.setModel(classifier.prediction.model, model)
predictionResult = test_algorithm.compute().get(classifier.prediction.prediction)

block = BlockDescriptor()
predictionResult.getBlockOfRows(0, predictionResult.getNumberOfRows(), readOnly, block)
predictions = block.getArray()

from sklearn.metrics import accuracy_score
print('Accuracy = ', accuracy_score(Y_test, predictions))

from sklearn.metrics import classification_report
print(classification_report(Y_test, predictions))