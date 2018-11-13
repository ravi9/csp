#!/usr/bin/python
# -*- coding: utf-8 -*-

import daal4py as d4p
import numpy as np

class LogisticRegression:

    '''
....Constructor to set LogisticRegression compute parameters
....'''

    def __init__(
        self,
        nClasses,
        dtype="float",
        penaltyL1=0,
        penaltyL2=0,
        interceptFlag=True,
        resultsToCompute="computeClassesLabels",
        optSolverParam = {}
        ):

        self.dtype = dtype
        self.nClasses = nClasses
        self.penaltyL1 = penaltyL1
        self.penaltyL2 = penaltyL2
        self.interceptFlag = interceptFlag
        self.optSolverParam = optSolverParam
        self.resultsToCompute = resultsToCompute

    def train(self, train_data, train_labels):
        #self.betaResult = self.result.get(classifier.training.model).getBeta()
        dtype = (np.float64 if self.dtype == "double" else np.float32)
        optSolver = None
        #create a solver
        if self.optSolverParam['solverName'] == 'sgd':
            lrs = np.array([[self.optSolverParam['solverLearningRate']]], dtype=dtype)
            batchSize_ = int(self.optSolverParam['solverBatchSize'])
            method = self.optSolverParam["solverMethod"]
            if method == "defaultDense":
                batchSize_ = 1
            optSolver = d4p.optimization_solver_sgd(function = None, learningRateSequence = lrs,
                                                    method = method,
                                                    accuracyThreshold = dtype(self.optSolverParam['solverAccuracyThreshold']),
                                                    nIterations = int(self.optSolverParam['solverMaxIterations']),
                                                    batchSize = batchSize_
                                                    )
        if self.optSolverParam['solverName'] == 'lbfgs':
            sls = np.array([[self.optSolverParam['solverStepLength']]], dtype=dtype)
            optSolver = d4p.optimization_solver_lbfgs(function = None,
                                                      stepLengthSequence=sls,
                                                      accuracyThreshold = dtype(self.optSolverParam['solverAccuracyThreshold']),
                                                      nIterations = int(self.optSolverParam['solverMaxIterations']),
                                                      batchSize = int(self.optSolverParam['solverBatchSize']),
                                                      correctionPairBatchSize = int(self.optSolverParam['solverCorrectionPairBatchSize']),
                                                      L = int(self.optSolverParam['L'])
                                                      )
        if self.optSolverParam['solverName'] == 'adagrad':
            lr = np.array([[self.optSolverParam['solverLearningRate']]], dtype=dtype)
            optSolver = d4p.optimization_solver_adagrad(function = None,
                                                        learningRate=lr,
                                                        accuracyThreshold = dtype(self.optSolverParam['solverAccuracyThreshold']),
                                                        nIterations = int(self.optSolverParam['solverMaxIterations']),
                                                        batchSize = int(self.optSolverParam['solverBatchSize'])
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
        predict_alg = d4p.logistic_regression_prediction(fptype = self.dtype, nClasses=self.nClasses,
                                                         resultsToCompute = self.resultsToCompute)
        predict_result = predict_alg.compute(predict_data, model)
        self.prediction = predict_result.prediction
        self.probabilities = predict_result.probabilities
        return self
