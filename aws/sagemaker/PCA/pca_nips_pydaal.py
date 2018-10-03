import pandas as pd
dataset = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00371/NIPS_1987-2015.csv', index_col=0)
import numpy as np
import timeit

matrx = dataset.transpose().apply(lambda x: (x-x.mean())/x.std()).as_matrix()

from daal.algorithms import pca
import daal.algorithms.pca.transform as pca_transform
from daal.data_management import FileDataSource, DataSourceIface, HomogenNumericTable, BlockDescriptor, readOnly
algorithm = pca.Batch(fptype=np.float32)
algorithm.input.setDataset(pca.data, HomogenNumericTable(matrx))
algorithm.parameter.resultsToCompute = pca.mean | pca.variance | pca.eigenvalue
algorithm.parameter.nComponents = 11462
begin = timeit.default_timer()
result = algorithm.compute()
end = timeit.default_timer()
print('elapsed', end - begin)

block = BlockDescriptor()
result.get(pca.eigenvalues).getBlockOfRows(0, result.get(pca.eigenvalues).getNumberOfRows(), readOnly, block)
eigen_val = block.getArray()
print ('eigen_val', eigen_val)
result.get(pca.eigenvectors).getBlockOfRows(0, result.get(pca.eigenvectors).getNumberOfRows(), readOnly, block)
eigen_vec = block.getArray()
print ('eigen_vec', eigen_vec)
print(matrx[0])
tralgo = pca_transform.Batch(fptype=np.float32)
tralgo.parameter.nComponents=11462
tralgo.input.setTable(pca_transform.data, HomogenNumericTable(matrx))
tralgo.input.setTable(pca_transform.eigenvectors, result.get(pca.eigenvectors))
trres = tralgo.compute()
#printNumericTable(trres.get(pca.transform.transformedData), "Transformed data: ", 4)
block = BlockDescriptor()
trres.get(pca_transform.transformedData).getBlockOfRows(0, trres.get(pca_transform.transformedData).getNumberOfRows(), readOnly, block)
transformedData = block.getArray()
np.save('transformedData.npy', transformedData)
print ('transformedData', transformedData)
