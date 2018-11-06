#!/usr/bin/python
# -*- coding: utf-8 -*-

import daal4py as d4p
import numpy as np

class LogisticRegression:

    '''
....Constructor to set Kmeans compute parameters
....'''

    def __init__(
        self,
        nClasses,
        dtype="float",
        penaltyL1=0,
        penaltyL2=0,
        interceptFlag=True,
        optSolverParam = {}
        ):
        """\n\t\tnClusters: default: None\n\t\t\tnumber of centroids to compute\n\t\tmaxIterations: default: 300\n\t\t\tmaximum number of iterations \n\t\tinitialCentroidMethod: default: \xe2\x80\x99defaultDense' \n\t\t\tInitial centroid assignment method. Refer here for other available methods\n\t\t method: default: 'defaultDense'\n\t\t\tfinal centroid computation mode. Refer here for other available methods\t \n\t\toversamplingFactor: default: 0.5\n\t\t\tapplicable only if initialCentroidMethod is \xe2\x80\x98parallelPlusDense\xe2\x80\x99, \xe2\x80\x98parallelPlusCSR\xe2\x80\x99\n\t\t\tA fraction of nClusters in each of nRounds of parallel K-Means++.\n\t\t\tL=nClusters*oversamplingFactor points are sampled in a round\n\t\tnRounds: default: 5\n\t\t\tapplicable only if initialCentroidMethod is \xe2\x80\x98parallelPlusDense\xe2\x80\x99, \xe2\x80\x98parallelPlusCSR\xe2\x80\x99\n\t\t\tThe number of rounds for parallel K-Means++. (L*nRounds) must be greater than nClusters.\n\t\taccuracyThreshold: default: 0.0001\n\t\t\tThe threshold for termination of the algorithm.\n\t\tgamma: default:1.0\n\t\t\tThe weight to be used in distance calculation for binary categorical features.\n\t\tdistanceType: default: 'euclidean'\n\t\t\tThe measure of closeness between points being clustered.\n\t\tassignFlag: default: True\n\t\t\tFlag that enables cluster assignments for clustered data points.\n\t\t"""

        self.dtype = dtype
        self.nClasses = nClasses
        self.penaltyL1 = penaltyL1
        self.penaltyL2 = penaltyL2
        self.interceptFlag = interceptFlag
        self.optSolverParam = optSolverParam

    def train(self, train_data, train_labels):
        #self.betaResult = self.result.get(classifier.training.model).getBeta()
        optSolver = None
        #create a solver
        if self.optSolverParam['name'] == 'sgd':
            lrs = np.array([[self.optSolverParam['learningRate']]], dtype=np.double)
            batchSize = self.optSolverParam['batchSize']
            method = self.optSolverParam["method"]
            if method == "defaultDense":
                batchSize = 1
            optSolver = d4p.optimization_solver_sgd(function = None, learningRateSequence = lrs,
                                                    method = method,
                                                    accuracyThreshold = self.optSolverParam['accuracyThreshold'],
                                                    nIterations = self.optSolverParam['maxIterations'],
                                                    batchSize = batchSize
                                                    )
        if self.optSolverParam['name'] == 'lbfgs':
            sls = np.array([[self.optSolverParam['stepLength']]], dtype=np.double)
            optSolver = d4p.optimization_solver_lbfgs(function = None,
                                                      stepLengthSequence=sls,
                                                      accuracyThreshold = self.optSolverParam['accuracyThreshold'],
                                                      nIterations = self.optSolverParam['maxIterations'],
                                                      batchSize = self.optSolverParam['batchSize'],
                                                      correctionPairBatchSize = self.optSolverParam['correctionPairBatchSize'],
                                                      L = self.optSolverParam['L']
                                                      )
        if self.optSolverParam['name'] == 'adagrad':
            lr = np.array([[self.optSolverParam['learningRate']]], dtype=np.double)
            optSolver = d4p.optimization_solver_adagrad(function = None,
                                                        learningRate=lr,
                                                        accuracyThreshold = self.optSolverParam['accuracyThreshold'],
                                                        nIterations = self.optSolverParam['maxIterations'],
                                                        batchSize = self.optSolverParam['batchSize']
                                                        )
            
        train_alg = d4p.logistic_regression_training(nClasses      = self.nClasses,
                                                     penaltyL1     = self.penaltyL1,
                                                     penaltyL2     = self.penaltyL2,
                                                     interceptFlag = self.interceptFlag,
                                                     fptype        = self.dtype,
                                                     optimizationSolver = optSolver
                                                     )
        self.trainingResult = train_alg.compute(train_data, train_labels)

        return self

    def predict(self, predict_data, model):
        #self.probabilities = predictionResult.get(classifier.prediction.probabilities)
        #self.logProbabilities = predictionResult.get(classifier.prediction.logProbabilities)
        # set parameters and compute predictions
        predict_alg = d4p.logistic_regression_prediction(nClasses=self.nClasses,
                                                         resultsToCompute="computeClassesLabels")
        predict_result = predict_alg.compute(predict_data, model)
        self.prediction = predict_result.prediction
        return self
