from __future__ import annotations
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

class Stats:
    def __init__(self, objects:gpd.GeoDataFrame = None, dendrite:gpd.GeoDataFrame = None):
        self.objects = objects
        self.dendrite = dendrite
    
    def refresh_results(self, objects:gpd.GeoDataFrame, dendrite:gpd.GeoDataFrame):
        self.objects = objects
        self.dendrite = dendrite
    
    def points_stats(self):
        if self.objects is not None:
            n_connections = self.objects.groupby('connections').fid.agg('count')
            return n_connections
        else:
            return None
    
    def dendrite_stats(self):
        stats = self.dendrite
        if stats is not None:
            stats['length'] = stats.geometry.length
            mean_length = stats.groupby('level').length.agg('mean')
            return mean_length
        else:
            return None