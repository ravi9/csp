# file: LogisticRegression.py
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
                                                      L = int(self.optSolverParam['solverL'])
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
        # set parameters and compute predictions
        predict_alg = d4p.logistic_regression_prediction(fptype = self.dtype, nClasses=self.nClasses,
                                                         resultsToCompute = self.resultsToCompute)
        predict_result = predict_alg.compute(predict_data, model)
        self.prediction = predict_result.prediction
        self.probabilities = predict_result.probabilities
        return self
