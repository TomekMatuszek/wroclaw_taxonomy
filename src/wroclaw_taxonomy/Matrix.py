from __future__ import annotations
import geopandas as gpd
import numpy as np
import numpy.ma as ma
import warnings
from scipy.spatial.distance import cdist

class Matrix:
    def __init__(self, data:gpd.GeoDataFrame, columns:str | list[str], normalize:bool):
        for_matrix = data.loc[:,columns]
        if normalize == True:
            if any(item in ('lat', 'lon') for item in columns):
                warnings.warn('You are normalizing coordinate values. It may slightly change results.')
            for_matrix = for_matrix.apply(lambda x: (x-x.mean())/ x.std(), axis=0)
        
        distance_matrix = np.array(cdist(for_matrix, for_matrix, metric='euclidean'))
        self.matrix = distance_matrix
    
    def mask_matrix(self):
        return ma.masked_array(self.matrix, mask= self.matrix==0)
    
    def clear_matrix(self, data:gpd.GeoDataFrame, column:str):
        for i in range(0, data.shape[0]):
            cluster = data.loc[i, column]
            indexes = data.index[data[column] == cluster].tolist()
            self.matrix[i, indexes] = 0
            self.matrix[indexes, i] = 0
        return self.matrix