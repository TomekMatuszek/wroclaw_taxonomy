import pandas as pd
import geopandas as gpd
import numpy as np
import numpy.ma as ma
import warnings
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from scipy.spatial.distance import cdist

class Dendrite:
    def __init__(self, src):
        if isinstance(src, gpd.GeoDataFrame):
            data = src
        elif isinstance(src, str):
            data = gpd.read_file(src, driver = 'GeoJSON')
        else:
            raise TypeError('Argument in_file has to be either string or GeoDataFrame')
        
        # change coordinate reference system
        if data.crs.to_authority()[1] != '4326':
            data.to_crs(epsg=4326, inplace=True)

        # get UTM zone number and convert to this EPSG
        zone = round((data.centroid.x[0] + 180) / 6)
        if data.centroid.y[0] >= 0:
            crs = int("326" + str(zone))
        else:
            crs = int("327" + str(zone))

        data.to_crs(epsg=crs, inplace=True)
        self.crs = crs

        # create needed columns
        data['fid'] = [i for i in range(1, data.shape[0] + 1)]
        data['lat'] = data.centroid.y
        data['lon'] = data.centroid.x

        self.data = data
        self.source_data = data.loc[:, data.columns]
    def __str__(self):
        pass
    def __clear_matrix(self, data, matrix, column):
        for i in range(0, data.shape[0]):
            cluster = data.loc[i, column]
            indexes = data.index[data[column] == cluster].tolist()
            matrix[i, indexes] = 0
            matrix[indexes, i] = 0
        return matrix
    def calculate(self, columns=['lat', 'lon'], normalize=False):
        data = self.data
        assert isinstance(columns, list), 'Argument columns has to be a list'
        # stworzenie macierzy odległości
        if normalize == True:
            if any(item in ('lat', 'lon') for item in columns):
                warnings.warn('You are normalizing coordinate values. It may slightly change results.')
            for_matrix = data.loc[:,columns].apply(lambda x: (x-x.mean())/ x.std(), axis=0)
        else:
            for_matrix = data.loc[:,columns]
        
        distance_matrix = np.array(cdist(for_matrix, for_matrix, metric='euclidean'))
        self.matrix = np.array(cdist(for_matrix, for_matrix, metric='euclidean'))
        
        # wyznaczenie najbliższych sąsiadów
        data['nearest1'] = np.argmin(ma.masked_array(distance_matrix, mask= distance_matrix==0), axis=1) + 1
        data['cluster1'] = np.argmin(ma.masked_array(distance_matrix, mask= distance_matrix==0), axis=1) + 1

        # zgrupowanie punktów w klastry
        for i in range(0, data.shape[0]):
            data.loc[data['cluster1'] == i + 1, 'cluster1'] = data.loc[i, 'cluster1']

        # czyszczenie macierzy ze stworzonych połączeń
        distance_matrix = self.__clear_matrix(data, distance_matrix, 'cluster1')

        # powtarzanie łączenia aż do otrzymania jednego klastra
        lvl = 2
        while data[f'cluster{lvl-1}'].unique().shape[0] > 1:
            # wyznaczenie najbliższego punktu wraz z odległością po maskowaniu macierzy z zer
            data[f'nearest{lvl}'] = np.argmin(ma.masked_array(distance_matrix, mask= distance_matrix==0), axis=1) + 1
            data[f'nearest{lvl}_dist'] = np.min(ma.masked_array(distance_matrix, mask= distance_matrix==0), axis=1)

            # pozostawienie tylko najbliższych odległości dla każdego klastra
            for i in data[f'cluster{lvl-1}'].unique():
                cluster = data.loc[data[f'cluster{lvl-1}'] == i, :]
                nearest = cluster.loc[cluster[f'nearest{lvl}_dist'] == np.min(cluster[f'nearest{lvl}_dist']), 'fid']
                data.loc[(data['fid'] != nearest.values[0]) & (data[f'cluster{lvl-1}'] == i), f'nearest{lvl}'] = -1

            # przyporządkowanie numeru klastra do którego należy najbliższy sąsiad
            for i in range(0, data.shape[0]):
                if data.loc[i, f'nearest{lvl}'] == -1:
                    continue
                else:
                    nearest_cluster = data.loc[data['fid'] == data.loc[i, f'nearest{lvl}'], f'cluster{lvl-1}']
                    data.loc[data[f'cluster{lvl-1}'] == data.loc[i, f'cluster{lvl-1}'], f'cluster{lvl}'] = nearest_cluster.values[0]

            # zgrupowanie klastrów w większe klastry
            for i in range(0, data.shape[0]):
                data.loc[data[f'cluster{lvl}'] == data.loc[i, f'cluster{lvl-1}'], f'cluster{lvl}'] = data.loc[i, f'cluster{lvl}']

            # czyszczenie macierzy
            distance_matrix = self.__clear_matrix(data, distance_matrix, f'cluster{lvl}')
            lvl += 1

        self.levels = lvl
        # tworzenie linestringów dla połączeń każdego rzędu
        for i in range(1, lvl):
            for j in range(0, data.shape[0]):
                if data.loc[j, f'nearest{i}'] != -1:
                    data.loc[j, f'line{i}'] = LineString([data.loc[j, 'geometry'].centroid, data.loc[data['fid'] == data.loc[j, f'nearest{i}'], 'geometry'].values[0].centroid]).wkt
                else:
                    data.loc[j, f'line{i}'] = ''
        
        for i in range(0, data.shape[0]):
            to_ids = {x for lst in [data.loc[data[f'nearest{j}'] == i + 1, 'fid'].to_list() for j in range(1, self.levels)] for x in lst}
            from_ids = set([data.loc[i, f'nearest{j}'] for j in range(1, self.levels)]) - {-1}
            data.loc[i, 'connections'] =  len(to_ids | from_ids)
        
        self.results = data
    def export_objects(self, out_file='dendrite_points.geojson'):
        results = self.results
        results.to_file(out_file, driver='GeoJSON', crs=self.crs)
        return results
    def export_dendrite(self, out_file='dendrite.geojson'):
        results = self.results
        dendrite = gpd.GeoDataFrame(columns=['cluster', 'level', 'geometry'], geometry='geometry')
        for i in range(1, self.levels):
            lines = results.loc[results[f'line{i}'] != '', ['fid', f'nearest{i}', f'cluster{i}', f'line{i}']]
            lines.rename(columns={f'cluster{i}':'cluster',f'nearest{i}':'nearest'}, inplace=True)
            lines['level'] = i
            lines['geometry'] = gpd.GeoSeries.from_wkt(lines[f'line{i}'])
            dendrite = pd.concat([dendrite, gpd.GeoDataFrame(lines[['fid', 'nearest', 'cluster', 'level', 'geometry']], geometry='geometry')])
        self.dendrite = dendrite
        dendrite.to_file(out_file, driver='GeoJSON', crs=self.crs)
        return dendrite
    def plot(self):
        dendrite = self.dendrite
        points = self.results
        fig, ax = plt.subplots(figsize = (10, 10))
        for lvl, lwd in zip(range(1, max(dendrite['level']) + 1), np.arange(0.5, 2 + (1.5 / (max(dendrite['level']) + 1)), (1.5 / (max(dendrite['level']) + 1)))):
            dendrite[dendrite['level'] == lvl].plot(ax=ax, color='#222222',  linewidth=lwd, zorder=5)

        points.plot(ax=ax, color='#ff0000', markersize=(points['connections'] - 0.75) * 2, zorder=10)
        plt.show()

