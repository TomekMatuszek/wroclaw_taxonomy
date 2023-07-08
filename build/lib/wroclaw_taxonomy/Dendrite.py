from __future__ import annotations
import math
import pandas as pd
import geopandas as gpd
import numpy as np
import numpy.ma as ma
import warnings
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from scipy.spatial.distance import cdist
import wroclaw_taxonomy
from wroclaw_taxonomy.Matrix import Matrix

class Dendrite:
    """
    A class used to generate dendrite based on Wroclaw taxonomy method from points.

    ----------

    Attributes
    ----------
    crs : int
        coordinate reference system identifier
    source_data : GeoDataFrame
        dataset provided by user
    data : GeoDataFrame
        dataset provided by user
    matrix : NDArray
        distance matrix
    n_levels : int
        number of levels of result dendrite
    dendrite : GeoDataFrame
        line layer representing created dendrite
    results : GeoDataFrame
        source data with new columns such as cluster ID or number of connections

    Methods
    ----------
    calculate(columns:list = ['lat', 'lon'], normalize:bool = False)
        calculates dendrite
    export_objects(out_file:str = 'dendrite_points.geojson')
        exports source data with new columns such as cluster ID or number of connections
    export_dendrite(out_file:str = 'dendrite.geojson')
        exports line layer with calculated dendrite
    """

    def __init__(self, src: str | gpd.GeoDataFrame):
        """
        Parameters
        ----------
        src : str | GeoDataFrame
            path to spatial file or GeoDataFrame object with loaded data
        """
        if isinstance(src, gpd.GeoDataFrame):
            data = src
        elif isinstance(src, str):
            data = gpd.read_file(src)
        else:
            raise TypeError('Argument in_file has to be either string or GeoDataFrame')

        # change coordinate reference system
        if data.crs.to_authority()[1] != '4326':
            data.to_crs(epsg=4326, inplace=True)

        # get UTM zone number and convert to this EPSG
        crs = self.__get_UTM_zone(data.total_bounds)
        data.to_crs(epsg=crs, inplace=True)
        self.crs = crs

        # create needed columns
        data['fid'] = [i for i in range(1, data.shape[0] + 1)]
        data['lat'] = data.centroid.y
        data['lon'] = data.centroid.x

        self._processed = False
        self.plot_style = {
            "markersize": 10,
            "cmap": 'jet',
            "line_color": '#222222',
            "object_color": '#ff0000'
        }
        self.data = data
        self.source_data = data.copy()
        self.stats = wroclaw_taxonomy.Stats()
    
    def __str__(self):
        if self._processed:
            return f'<Dendrite object:  {self.levels} levels>'
        else:
            return '<Dendrite object:  unprocessed>'
      
    def __get_UTM_zone(self, bounds:np.ndarray) -> int:
        if math.ceil((bounds[2] + 180) / 6) - math.ceil((bounds[0] + 180) / 6) > 1:
            return 3857
        else:
            zone = math.ceil(((bounds[2] + bounds[0]) / 2 + 180) / 6)
            if bounds[3] >= 0:
                crs = int("326" + str(zone))
            else:
                crs = int("327" + str(zone))
            return crs
    
    def __cluster_points(self, data:gpd.GeoDataFrame, lvl:int):
        for i in range(0, data.shape[0]):
            if lvl == 1:
                data.loc[data['cluster1'] == i + 1, 'cluster1'] = data.loc[i, 'cluster1']
            else:
                data.loc[data[f'cluster{lvl}'] == data.loc[i, f'cluster{lvl-1}'], f'cluster{lvl}'] = data.loc[i, f'cluster{lvl}']
        return data
    
    def __merge_clusters(self, data:gpd.GeoDataFrame, distance_matrix:Matrix):
        lvl = 2
        while data[f'cluster{lvl-1}'].unique().shape[0] > 1:
            # get nearest neighbour of cluster
            data[f'nearest{lvl}'] = np.argmin(distance_matrix.mask_matrix(), axis=1) + 1
            data[f'nearest{lvl}_dist'] = np.min(distance_matrix.mask_matrix(), axis=1)

            for i in data[f'cluster{lvl-1}'].unique():
                cluster = data.loc[data[f'cluster{lvl-1}'] == i, :]
                nearest = cluster.loc[cluster[f'nearest{lvl}_dist'] == np.min(cluster[f'nearest{lvl}_dist']), 'fid']
                data.loc[(data['fid'] != nearest.values[0]) & (data[f'cluster{lvl-1}'] == i), f'nearest{lvl}'] = -1

            # get id of nearest cluster
            for i in range(0, data.shape[0]):
                if data.loc[i, f'nearest{lvl}'] == -1:
                    continue
                else:
                    nearest_cluster = data.loc[data['fid'] == data.loc[i, f'nearest{lvl}'], f'cluster{lvl-1}']
                    data.loc[data[f'cluster{lvl-1}'] == data.loc[i, f'cluster{lvl-1}'], f'cluster{lvl}'] = nearest_cluster.values[0]

            # grouping clusters into bigger ones
            data = self.__cluster_points(data, lvl)
            # clearing distance matrix
            distance_matrix.clear_matrix(data, f'cluster{lvl}')
            lvl += 1
        return (data, lvl)
    
    def __create_connections(self, data:gpd.GeoDataFrame):
        for i in range(1, self.levels):
            for j in range(0, data.shape[0]):
                if data.loc[j, f'nearest{i}'] != -1:
                    data.loc[j, f'line{i}'] = LineString([data.loc[j, 'geometry'].centroid, data.loc[data['fid'] == data.loc[j, f'nearest{i}'], 'geometry'].values[0].centroid]).wkt
                else:
                    data.loc[j, f'line{i}'] = ''
        return data
    
    def __count_connections(self, data:gpd.GeoDataFrame):
        for i in range(0, data.shape[0]):
            to_ids = {x for lst in [data.loc[data[f'nearest{j}'] == i + 1, 'fid'].to_list() for j in range(1, self.levels)] for x in lst}
            from_ids = set([data.loc[i, f'nearest{j}'] for j in range(1, self.levels)]) - {-1}
            data.loc[i, 'connections'] =  len(to_ids | from_ids)
        return data
    
    def __create_dendrite(self, data:gpd.GeoDataFrame):
        dendrite = gpd.GeoDataFrame(columns=['cluster', 'level', 'geometry'], geometry='geometry')
        for i in range(1, self.levels):
            lines = data.loc[data[f'line{i}'] != '', ['fid', f'nearest{i}', f'cluster{i}', f'line{i}']]
            lines.rename(columns={f'cluster{i}':'cluster',f'nearest{i}':'nearest'}, inplace=True)
            lines['level'] = i
            lines['geometry'] = gpd.GeoSeries.from_wkt(lines[f'line{i}'])
            dendrite = pd.concat([dendrite, gpd.GeoDataFrame(lines[['fid', 'nearest', 'cluster', 'level', 'geometry']], geometry='geometry')])
        return dendrite
    
    def calculate(self, columns:list = ['lat', 'lon'], normalize:bool = False) -> Dendrite:
        """
        Method which calculates dendrite based on Wroclaw taxonomy.

        ----------

        Parameters
        ----------
        columns : list
            list of columns which should be used in computing Euclidean distance between points
        normalize : bool
            if True, values of chosen columns are normalized to (0, 1) range
        """
        data = self.data
        assert isinstance(columns, list), 'Argument columns has to be a list'
        
        # create distance matrix
        distance_matrix = Matrix(data, columns, normalize)
        # get nearest neighbours
        nn = np.argmin(distance_matrix.mask_matrix(), axis=1) + 1
        data['nearest1'] = nn
        data['cluster1'] = nn

        # grouping into clusters
        data = self.__cluster_points(data, 1)
        # clearing matrix
        distance_matrix.clear_matrix(data, 'cluster1')

        # repeating clustering until getting one big cluster
        data, lvl = self.__merge_clusters(data, distance_matrix)
        self.levels = lvl
        # linestrings for every connection level
        data = self.__create_connections(data)    
        # counting connections for every point
        data = self.__count_connections(data)
        # creating dendrite lines
        dendrite = self.__create_dendrite(data)
        
        #dendrite['length'] = dendrite.length
        #dendrite = dendrite[~((dendrite['length'] > (np.mean(dendrite.length) + 2 * np.std(dendrite.length))) & (dendrite['level'] > 2))]

        self.stats.refresh_results(data, dendrite)

        self.n_levels = lvl - 1
        self.dendrite = dendrite
        self.results = data
        self._processed = True
        return self
    
    def export_objects(self, out_file:str = 'dendrite_points.geojson') -> gpd.GeoDataFrame:
        """
        Exports source data with added columns to GeoJSON file.

        ----------

        Parameters
        ----------
        out_file : str
            path to output file
        """
        if self._processed is True:
            self.results.to_file(out_file, driver='GeoJSON', crs=self.crs)
            return self.results
        else:
            warnings.warn('Dendrite has not been calculated yet! Source data returned.')
            self.data.to_file(out_file, driver='GeoJSON', crs=self.crs)
            return self.data
    
    def export_dendrite(self, out_file:str = 'dendrite.geojson') -> gpd.GeoDataFrame:
        """
        Exports computed dendrite to GeoJSON file.

        ----------

        Parameters
        ----------
        out_file : str
            path to output file
        """
        if self._processed is True:
            self.dendrite.to_file(out_file, driver='GeoJSON', crs=self.crs)
            return self.dendrite
        else:
            warnings.warn('Dendrite has not been calculated yet!')
            return None
    
    def create_plotter(self) -> wroclaw_taxonomy.Plotter:
        """
        Creates Plotter instance from calculated dendrite.
        """
        plotter = wroclaw_taxonomy.Plotter(self)
        return plotter
 