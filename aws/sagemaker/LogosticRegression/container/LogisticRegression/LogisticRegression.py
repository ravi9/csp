#!/usr/bin/python
# -*- coding: utf-8 -*-

#import daal.algorithms.kmeans.init
#from daal.algorithms import kmeans
from daal.algorithms import logistic_regression
from daal.algorithms.logistic_regression import prediction, training
from daal.algorithms import classifier

from daal.data_management import InputDataArchive, OutputDataArchive
from daal.data_management import Compressor_Zlib, Decompressor_Zlib, \
    level9, DecompressionStream, CompressionStream, HomogenNumericTable
from daal.data_management import BlockDescriptor, HomogenNumericTable, BlockDescriptor_Float32, readOnly, readWrite
#from utils import printNumericTable
import numpy as np
from numpy import float32, float64, int32
import warnings

class LogisticRegression:

    '''
....Constructor to set Kmeans compute parameters
....'''

    def __init__(
        self,
        nClasses,
        dtype=float64,
        penaltyL1=0,
        penaltyL2=0,
        ):
        """\n\t\tnClusters: default: None\n\t\t\tnumber of centroids to compute\n\t\tmaxIterations: default: 300\n\t\t\tmaximum number of iterations \n\t\tinitialCentroidMethod: default: \xe2\x80\x99defaultDense' \n\t\t\tInitial centroid assignment method. Refer here for other available methods\n\t\t method: default: 'defaultDense'\n\t\t\tfinal centroid computation mode. Refer here for other available methods\t \n\t\toversamplingFactor: default: 0.5\n\t\t\tapplicable only if initialCentroidMethod is \xe2\x80\x98parallelPlusDense\xe2\x80\x99, \xe2\x80\x98parallelPlusCSR\xe2\x80\x99\n\t\t\tA fraction of nClusters in each of nRounds of parallel K-Means++.\n\t\t\tL=nClusters*oversamplingFactor points are sampled in a round\n\t\tnRounds: default: 5\n\t\t\tapplicable only if initialCentroidMethod is \xe2\x80\x98parallelPlusDense\xe2\x80\x99, \xe2\x80\x98parallelPlusCSR\xe2\x80\x99\n\t\t\tThe number of rounds for parallel K-Means++. (L*nRounds) must be greater than nClusters.\n\t\taccuracyThreshold: default: 0.0001\n\t\t\tThe threshold for termination of the algorithm.\n\t\tgamma: default:1.0\n\t\t\tThe weight to be used in distance calculation for binary categorical features.\n\t\tdistanceType: default: 'euclidean'\n\t\t\tThe measure of closeness between points being clustered.\n\t\tassignFlag: default: True\n\t\t\tFlag that enables cluster assignments for clustered data points.\n\t\t"""

        self.dtype = dtype
        self.nClasses = nClasses
        self.penaltyL1 = penaltyL1
        self.penaltyL2 = penaltyL2

    def train(self, data, labels):
        algorithm = training.Batch(self.nClasses)

        # Pass the training data set and dependent values to the algorithm
        algorithm.input.set(classifier.training.data, data)
        algorithm.input.set(classifier.training.labels, labels)
        algorithm.parameter().penaltyL1=self.penaltyL1;
        algorithm.parameter().penaltyL2=self.penaltyL2;
        self.trainingResult = algorithm.compute()
        #self.betaResult = self.result.get(classifier.training.model).getBeta()
        return self

    def predict(self, data, model):
        algorithm = prediction.Batch(self.nClasses)
        # Pass the testing data set and trained model to the algorithm
        algorithm.input.setTable(classifier.prediction.data,  data)
        algorithm.input.setModel(classifier.prediction.model, model)
        #algorithm.parameter().resultsToCompute |= logistic_regression.prediction.computeClassesProbabilities | logistic_regression.prediction.computeClassesLogProbabilities
        predictionResult = algorithm.compute()
        self.prediction = predictionResult.get(classifier.prediction.prediction)
        #self.probabilities = predictionResult.get(classifier.prediction.probabilities)
        #self.logProbabilities = predictionResult.get(classifier.prediction.logProbabilities)
        return self

    def serializeTrainingResult(self):
        #  Create a data archive to serialize the numeric table
        dataArch = InputDataArchive()
        #  Serialize the numeric table into the data archive
        self.trainingResult.serialize(dataArch)
        #  Get the length of the serialized data in bytes
        length = dataArch.getSizeOfArchive()
        #  Store the serialized data in an array
        buffer = np.zeros(length, dtype=np.ubyte)
        dataArch.copyArchiveToArray(buffer)

        return buffer

    def deserializeTrainingResult(self, buffer):
        #  Create a data archive to deserialize the numeric table
        dataArch = OutputDataArchive(buffer)
        #  Create a numeric table object
        self.trainingResult = training.Result()
        #  Deserialize the numeric table from the data archive
        self.trainingResult.deserialize(dataArch)

        return self.trainingResult

#    def compress(self, arrayData):
#        compressor = Compressor_Zlib()
#        compressor.parameter.gzHeader = True
#        compressor.parameter.level = level9
#        comprStream = CompressionStream(compressor)
#        comprStream.push_back(arrayData)
#        compressedData = np.empty(comprStream.getCompressedDataSize(),
#                                  dtype=np.uint8)
#        comprStream.copyCompressedArray(compressedData)
#        return compressedData
#
#    def decompress(self, arrayData):
#        decompressor = Decompressor_Zlib()
#        decompressor.parameter.gzHeader = True
#
#        # Create a stream for decompression
#
#        deComprStream = DecompressionStream(decompressor)
#
#        # Write the compressed data to the decompression stream and decompress it
#
#        deComprStream.push_back(arrayData)
#
#        # Allocate memory to store the decompressed data
#
#        bufferArray = np.empty(deComprStream.getDecompressedDataSize(),
#                               dtype=np.uint8)
#
#        # Store the decompressed data
#
#        deComprStream.copyDecompressedArray(bufferArray)
#        return bufferArray
#
#    # -------------------
#    # ***Serialization***
#    # -------------------
#
#    def serialize(
#        self,
#        data,
#        fileName=None,
#        useCompression=False,
#        ):
#        buffArrObjName = (str(type(data)).split()[1].split('>')[0]
#                          + '()').replace("'", '')
#        dataArch = InputDataArchive()
#        data.serialize(dataArch)
#        length = dataArch.getSizeOfArchive()
#        bufferArray = np.zeros(length, dtype=np.ubyte)
#        dataArch.copyArchiveToArray(bufferArray)
#        if useCompression == True:
#            if fileName != None:
#                if len(fileName.rsplit('.', 1)) == 2:
#                    fileName = fileName.rsplit('.', 1)[0]
#                compressedData = Kmeans.compress(self, bufferArray)
#                np.save(fileName, compressedData)
#            else:
#                comBufferArray = Kmeans.compress(self, bufferArray)
#                serialObjectDict = {'Array Object': comBufferArray,
#                                    'Object Information': buffArrObjName}
#                return serialObjectDict
#        else:
#            if fileName != None:
#                if len(fileName.rsplit('.', 1)) == 2:
#                    fileName = fileName.rsplit('.', 1)[0]
#                np.save(fileName, bufferArray)
#            else:
#                serialObjectDict = {'Array Object': bufferArray,
#                                    'Object Information': buffArrObjName}
#                return serialObjectDict
#        infoFile = open(fileName + '.txt', 'w')
#        infoFile.write(buffArrObjName)
#        infoFile.close()
#
#    # ---------------------
#    # ***Deserialization***
#    # ---------------------
#
#    def deserialize(
#        self,
#        serialObjectDict=None,
#        fileName=None,
#        useCompression=False,
#        ):
#        import daal
#        if fileName != None and serialObjectDict == None:
#            bufferArray = np.load(fileName)
#            buffArrObjName = open(fileName.rsplit('.', 1)[0] + '.txt',
#                                  'r').read()
#        elif fileName == None and any(serialObjectDict):
#            bufferArray = serialObjectDict['Array Object']
#            buffArrObjName = serialObjectDict['Object Information']
#        else:
#            warnings.warn('Expecting "bufferArray" or "fileName" argument, NOT both'
#                          )
#            raise SystemExit
#        if useCompression == True:
#            bufferArray = Kmeans.decompress(self, bufferArray)
#        dataArch = OutputDataArchive(bufferArray)
#        try:
#            deSerialObj = eval(buffArrObjName)
#        except AttributeError:
#            deSerialObj = HomogenNumericTable()
#        deSerialObj.deserialize(dataArch)
#        return deSerialObj
