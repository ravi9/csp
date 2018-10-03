# file: kdtree_knn_dense_batch.py
#===============================================================================
# Copyright 2014-2018 Intel Corporation.
#
# This software and the related documents are Intel copyrighted  materials,  and
# your use of  them is  governed by the  express license  under which  they were
# provided to you (License).  Unless the License provides otherwise, you may not
# use, modify, copy, publish, distribute,  disclose or transmit this software or
# the related documents without Intel's prior written permission.
#
# This software and the related documents  are provided as  is,  with no express
# or implied  warranties,  other  than those  that are  expressly stated  in the
# License.
#===============================================================================

## <a name="DAAL-EXAMPLE-PY-KDTREE_KNN_DENSE_BATCH"></a>
## \example kdtree_knn_dense_batch.py


import pandas as pd	
import sys, os
sys.path.append(os.path.join(os.path.dirname(sys.executable),'share','pydaal_examples','examples','python','source'))
from customUtils import serialize, deserialize, getArrayFromNT
from time import time
import numpy as np
from daal.algorithms.kdtree_knn_classification import training, prediction
from daal.data_management import BlockDescriptor, readOnly
from daal.algorithms import classifier
from daal.data_management import (
    DataSourceIface, FileDataSource, HomogenNumericTable, MergedNumericTable, NumericTableIface

)
from daal.algorithms.classifier.quality_metric import multiclass_confusion_matrix
from daal.algorithms import multi_class_classifier
from daal.algorithms.classifier.quality_metric import binary_confusion_matrix
from daal.algorithms import classifier
from daal.algorithms import svm
from utils import printNumericTables,printNumericTable


# read the dowloaded raw data
trainDatasetFileName='/pat_tce/pvenkat2/largeData/covtypedata/covtype.data'
print("Reading raw data from {}".format(trainDatasetFileName))
raw = np.loadtxt(trainDatasetFileName, delimiter=',')

# split into train/test with a 90/10 split
np.random.seed(0)
np.random.shuffle(raw)
train_size = int(0.9 * raw.shape[0])
train_features = raw[:train_size, :-1]
train_labels = raw[:train_size, -1]
train_labels[train_labels==7]=0
test_features = raw[train_size:, :-1]
test_labels = raw[train_size:, -1]
test_labels[test_labels==7]=0
trainData=HomogenNumericTable(train_features)
trainGroundTruth=HomogenNumericTable(train_labels[:,np.newaxis])
testData=HomogenNumericTable(test_features)
groundTruthLabels=HomogenNumericTable(test_labels[:,np.newaxis])
print(np.unique(train_labels))
print('train_features shape = ', train_features.shape)
print('train_labels shape = ', train_labels.shape)

global trainingResult

# Create an algorithm object to train the KD-tree based kNN model
algorithm = training.Batch()

# Pass a training data set and dependent values to the algorithm
algorithm.input.set(classifier.training.data, trainData)
algorithm.input.set(classifier.training.labels, trainGroundTruth)

# Train the KD-tree based kNN model
print(getArrayFromNT(trainData,'head'))
print(getArrayFromNT(trainGroundTruth,'head'))

start = time()
trainingResult = algorithm.compute()
end = time()
print("Training time taken to compute: {}".format(end-start))

algorithmp = prediction.Batch(k=10)
# Pass the testing data set and trained model to the algorithm
algorithmp.input.setTable(classifier.prediction.data,  testData)
algorithmp.input.setModel(classifier.prediction.model, trainingResult.get(classifier.training.model))

# Compute prediction results
start=time()
predictionResult = algorithmp.compute()
end=time()
print("Prediction time taken to compute: {}".format(end-start))

def testModelQualityB():
    global predictedLabels, qualityMetricSetResult, groundTruthLabels

    # Retrieve predicted labels
    predictedLabels = predictionResult.get(classifier.prediction.prediction)

    # Create a quality metric set object to compute quality metrics of the SVM algorithm
    qualityMetricSet = svm.quality_metric_set.Batch()

    input = qualityMetricSet.getInputDataCollection().getInput(svm.quality_metric_set.confusionMatrix)

    input.set(binary_confusion_matrix.predictedLabels,   predictedLabels)
    input.set(binary_confusion_matrix.groundTruthLabels, groundTruthLabels)

    # Compute quality metrics and get the quality metrics
    # returns ResultCollection class from svm.quality_metric_set
    qualityMetricSetResult = qualityMetricSet.compute()
	
def testModelQualityM():
    global predictedLabels, qualityMetricSetResult

    # Retrieve predicted labels
    predictedLabels = predictionResult.get(classifier.prediction.prediction)

    # Create a quality metric set object to compute quality metrics of the multi-class classifier algorithm
    qualityMetricSet = multi_class_classifier.quality_metric_set.Batch(7)
    input = qualityMetricSet.getInputDataCollection().getInput(multi_class_classifier.quality_metric_set.confusionMatrix)

    input.set(multiclass_confusion_matrix.predictedLabels,   predictedLabels)
    input.set(multiclass_confusion_matrix.groundTruthLabels, groundTruthLabels)

    # Compute quality metrics and get the quality metrics
    # returns ResultCollection class from daal.algorithms.multi_class_classifier.quality_metric_set
    qualityMetricSetResult = qualityMetricSet.compute()


def printResultsB():

    # Print the classification results
    printNumericTables(
        groundTruthLabels, predictedLabels,
        "Ground truth", "Classification results",
        "SVM classification results (first 20 observations):", 20, interval=15, flt64=False
    )

    # Print the quality metrics
    qualityMetricResult = qualityMetricSetResult.getResult(svm.quality_metric_set.confusionMatrix)
    printNumericTable(qualityMetricResult.get(binary_confusion_matrix.confusionMatrix), "Confusion matrix:")

    block = BlockDescriptor()
    qualityMetricsTable = qualityMetricResult.get(binary_confusion_matrix.binaryMetrics)
    qualityMetricsTable.getBlockOfRows(0, 1, readOnly, block)
    qualityMetricsData = block.getArray().flatten()
    print("Accuracy:      {0:.3f}".format(qualityMetricsData[binary_confusion_matrix.accuracy]))
    print("Precision:     {0:.3f}".format(qualityMetricsData[binary_confusion_matrix.precision]))
    print("Recall:        {0:.3f}".format(qualityMetricsData[binary_confusion_matrix.recall]))
    print("F-score:       {0:.3f}".format(qualityMetricsData[binary_confusion_matrix.fscore]))
    print("Specificity:   {0:.3f}".format(qualityMetricsData[binary_confusion_matrix.specificity]))
    print("AUC:           {0:.3f}".format(qualityMetricsData[binary_confusion_matrix.AUC]))
    qualityMetricsTable.releaseBlockOfRows(block)
def printResultsM():

    # Print the classification results
    printNumericTables(
        groundTruthLabels, predictedLabels,
        "Ground truth", "Classification results",
        "SVM classification results (first 20 observations):", 20, interval=15, flt64=False
    )
    # Print the quality metrics
    qualityMetricResult = qualityMetricSetResult.getResult(multi_class_classifier.quality_metric_set.confusionMatrix)
    printNumericTable(qualityMetricResult.get(multiclass_confusion_matrix.confusionMatrix), "Confusion matrix:")

    block = BlockDescriptor()
    qualityMetricsTable = qualityMetricResult.get(multiclass_confusion_matrix.multiClassMetrics)
    qualityMetricsTable.getBlockOfRows(0, 1, readOnly, block)
    qualityMetricsData = block.getArray().flatten()
    print("Average accuracy: {0:.3f}".format(qualityMetricsData[multiclass_confusion_matrix.averageAccuracy]))
    print("Error rate:       {0:.3f}".format(qualityMetricsData[multiclass_confusion_matrix.errorRate]))
    print("Micro precision:  {0:.3f}".format(qualityMetricsData[multiclass_confusion_matrix.microPrecision]))
    print("Micro recall:     {0:.3f}".format(qualityMetricsData[multiclass_confusion_matrix.microRecall]))
    print("Micro F-score:    {0:.3f}".format(qualityMetricsData[multiclass_confusion_matrix.microFscore]))
    print("Macro precision:  {0:.3f}".format(qualityMetricsData[multiclass_confusion_matrix.macroPrecision]))
    print("Macro recall:     {0:.3f}".format(qualityMetricsData[multiclass_confusion_matrix.macroRecall]))
    print("Macro F-score:    {0:.3f}".format(qualityMetricsData[multiclass_confusion_matrix.macroFscore]))
    qualityMetricsTable.releaseBlockOfRows(block)	
	
testModelQualityM()
printResultsM()