#!/usr/bin/python
# -*- coding: utf-8 -*-

import daal.algorithms.kmeans.init
from daal.algorithms import kmeans
from daal.data_management import InputDataArchive, OutputDataArchive
from daal.data_management import Compressor_Zlib, Decompressor_Zlib, \
    level9, DecompressionStream, CompressionStream, HomogenNumericTable
from daal.data_management import BlockDescriptor, HomogenNumericTable, BlockDescriptor_Float32, readOnly, readWrite
#from utils import printNumericTable
import numpy as np
from numpy import float32, float64, int32
import warnings


class Kmeans:

    '''
....Constructor to set Kmeans compute parameters
....'''

    def __init__(
        self,
        nClusters,
        maxIterations=300,
        initialCentroidMethod='defaultDense',
        method='defaultDense',
        oversamplingFactor=0.5,
        nRounds=5,
        accuracyThreshold=0.0001,
        gamma=1.0,
        distanceType='euclidean',
        assignFlag=True,
        dtype=float64,
        ):
        """\n\t\tnClusters: default: None\n\t\t\tnumber of centroids to compute\n\t\tmaxIterations: default: 300\n\t\t\tmaximum number of iterations \n\t\tinitialCentroidMethod: default: \xe2\x80\x99defaultDense' \n\t\t\tInitial centroid assignment method. Refer here for other available methods\n\t\t method: default: 'defaultDense'\n\t\t\tfinal centroid computation mode. Refer here for other available methods\t \n\t\toversamplingFactor: default: 0.5\n\t\t\tapplicable only if initialCentroidMethod is \xe2\x80\x98parallelPlusDense\xe2\x80\x99, \xe2\x80\x98parallelPlusCSR\xe2\x80\x99\n\t\t\tA fraction of nClusters in each of nRounds of parallel K-Means++.\n\t\t\tL=nClusters*oversamplingFactor points are sampled in a round\n\t\tnRounds: default: 5\n\t\t\tapplicable only if initialCentroidMethod is \xe2\x80\x98parallelPlusDense\xe2\x80\x99, \xe2\x80\x98parallelPlusCSR\xe2\x80\x99\n\t\t\tThe number of rounds for parallel K-Means++. (L*nRounds) must be greater than nClusters.\n\t\taccuracyThreshold: default: 0.0001\n\t\t\tThe threshold for termination of the algorithm.\n\t\tgamma: default:1.0\n\t\t\tThe weight to be used in distance calculation for binary categorical features.\n\t\tdistanceType: default: 'euclidean'\n\t\t\tThe measure of closeness between points being clustered.\n\t\tassignFlag: default: True\n\t\t\tFlag that enables cluster assignments for clustered data points.\n\t\t"""

        self.nClusters = nClusters
        self.initialCentroidMethod = initialCentroidMethod
        self.oversamplingFactor = oversamplingFactor
        self.nRounds = nRounds
        self.method = method
        self.maxIterations = maxIterations
        self.accuracyThreshold = accuracyThreshold
        self.gamma = gamma
        self.distanceType = distanceType
        self.assignFlag = assignFlag
        self.dtype = dtype

    def compute(self, data):
        if self.method == 'lloydCSR':
            self.method = kmeans.lloydCSR
        elif self.method == 'defaultDense':
            self.method = kmeans.lloydDense
        if self.initialCentroidMethod == 'defaultDense':
            initMethod = kmeans.init.deterministicDense
        elif self.initialCentroidMethod == 'deterministicCSR':
            initMethod = kmeans.init.deterministicCSR
        elif self.initialCentroidMethod == 'randomDense':
            initMethod = kmeans.init.randomDense
        elif self.initialCentroidMethod == 'randomCSR':
            initMethod = kmeans.init.randomCSR
        elif self.initialCentroidMethod == 'plusPlusDense':
            initMethod = kmeans.init.plusPlusDense
        elif self.initialCentroidMethod == 'plusPlusCSR':
            initMethod = kmeans.init.plusPlusCSR
        elif self.initialCentroidMethod == 'parallelPlusDense':
            initMethod = kmeans.init.parallelPlusDense
        elif self.initialCentroidMethod == 'parallelPlusCSR ':
            initMethod = kmeans.init.parallelPlusCSR
        initAlg = kmeans.init.Batch(self.nClusters, method=initMethod,
                                    oversamplingFactor=self.oversamplingFactor,
                                    nRounds=self.nRounds,
                                    dtype=self.dtype)
        initAlg.input.set(kmeans.init.data, data)
        res = initAlg.compute()
        InitialCentroidsResult = res.get(kmeans.init.centroids)
        algorithm = kmeans.Batch(
            self.nClusters,
            self.maxIterations,
            method=self.method,
            accuracyThreshold=self.accuracyThreshold,
            gamma=self.gamma,
            distanceType=self.distanceType,
            assignFlag=self.assignFlag,
            )
        algorithm.input.set(kmeans.data, data)
        algorithm.input.set(kmeans.inputCentroids,
                            InitialCentroidsResult)
        res = algorithm.compute()
        if self.assignFlag != False:
            self.clusterAssignments = res.get(kmeans.assignments)
        self.centroidResults = res.get(kmeans.centroids)
        self.objectiveFunction = res.get(kmeans.objectiveFunction)
        return self

    def predict(self, centroidResults, data):
        algorithm = kmeans.Batch(
            self.nClusters,
            0,
            method=self.method,
            accuracyThreshold=self.accuracyThreshold,
            gamma=self.gamma,
            distanceType=self.distanceType,
            assignFlag=True,
            )
        algorithm.input.set(kmeans.data, data)
        algorithm.input.set(kmeans.inputCentroids, centroidResults)
        res = algorithm.compute()
        return res.get(kmeans.assignments)
    
   
    def compress(self, arrayData):
        compressor = Compressor_Zlib()
        compressor.parameter.gzHeader = True
        compressor.parameter.level = level9
        comprStream = CompressionStream(compressor)
        comprStream.push_back(arrayData)
        compressedData = np.empty(comprStream.getCompressedDataSize(),
                                  dtype=np.uint8)
        comprStream.copyCompressedArray(compressedData)
        return compressedData

    def decompress(self, arrayData):
        decompressor = Decompressor_Zlib()
        decompressor.parameter.gzHeader = True

        # Create a stream for decompression

        deComprStream = DecompressionStream(decompressor)

        # Write the compressed data to the decompression stream and decompress it

        deComprStream.push_back(arrayData)

        # Allocate memory to store the decompressed data

        bufferArray = np.empty(deComprStream.getDecompressedDataSize(),
                               dtype=np.uint8)

        # Store the decompressed data

        deComprStream.copyDecompressedArray(bufferArray)
        return bufferArray

    # -------------------
    # ***Serialization***
    # -------------------

    def serialize(
        self,
        data,
        fileName=None,
        useCompression=False,
        ):
        buffArrObjName = (str(type(data)).split()[1].split('>')[0]
                          + '()').replace("'", '')
        dataArch = InputDataArchive()
        data.serialize(dataArch)
        length = dataArch.getSizeOfArchive()
        bufferArray = np.zeros(length, dtype=np.ubyte)
        dataArch.copyArchiveToArray(bufferArray)
        if useCompression == True:
            if fileName != None:
                if len(fileName.rsplit('.', 1)) == 2:
                    fileName = fileName.rsplit('.', 1)[0]
                compressedData = Kmeans.compress(self, bufferArray)
                np.save(fileName, compressedData)
            else:
                comBufferArray = Kmeans.compress(self, bufferArray)
                serialObjectDict = {'Array Object': comBufferArray,
                                    'Object Information': buffArrObjName}
                return serialObjectDict
        else:
            if fileName != None:
                if len(fileName.rsplit('.', 1)) == 2:
                    fileName = fileName.rsplit('.', 1)[0]
                np.save(fileName, bufferArray)
            else:
                serialObjectDict = {'Array Object': bufferArray,
                                    'Object Information': buffArrObjName}
                return serialObjectDict
        infoFile = open(fileName + '.txt', 'w')
        infoFile.write(buffArrObjName)
        infoFile.close()

    # ---------------------
    # ***Deserialization***
    # ---------------------

    def deserialize(
        self,
        serialObjectDict=None,
        fileName=None,
        useCompression=False,
        ):
        import daal
        if fileName != None and serialObjectDict == None:
            bufferArray = np.load(fileName)
            buffArrObjName = open(fileName.rsplit('.', 1)[0] + '.txt',
                                  'r').read()
        elif fileName == None and any(serialObjectDict):
            bufferArray = serialObjectDict['Array Object']
            buffArrObjName = serialObjectDict['Object Information']
        else:
            warnings.warn('Expecting "bufferArray" or "fileName" argument, NOT both'
                          )
            raise SystemExit
        if useCompression == True:
            bufferArray = Kmeans.decompress(self, bufferArray)
        dataArch = OutputDataArchive(bufferArray)
        try:
            deSerialObj = eval(buffArrObjName)
        except AttributeError:
            deSerialObj = HomogenNumericTable()
        deSerialObj.deserialize(dataArch)
        return deSerialObj
