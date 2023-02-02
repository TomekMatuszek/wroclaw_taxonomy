import math
import pandas as pd
import geopandas as gpd
import numpy as np
import numpy.ma as ma
import warnings
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from scipy.spatial.distance import cdist
import imageio
import os

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
    set_style(style:dict = 'default')
        sets style of plots and animations created by this class
    plot(level:int = None, lines:bool = True, style:dict = None, show:bool = True)
        plots result dendrite and objects
    animate(out_file:str = 'dendrite.gif', frame_duration:int = 1, lines:bool = True, style:dict = None)
        created an animation presenting each step of dendrite creation
    """
    def __init__(self, src: str | gpd.GeoDataFrame):
        """
        Parameters
        ----------
        src : str | GeoDataFrame
            path to GeoJSON file or GeoDataFrame object with loaded data
        """
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
        crs = self.__get_UTM_zone(data.total_bounds)
        data.to_crs(epsg=crs, inplace=True)
        self.crs = crs

        # create needed columns
        data['fid'] = [i for i in range(1, data.shape[0] + 1)]
        data['lat'] = data.centroid.y
        data['lon'] = data.centroid.x

        self.plot_style = {
            "markersize": 10,
            "cmap": 'jet',
            "line_color": '#222222',
            "object_color": '#ff0000'
        }
        self.data = data
        self.source_data = data.copy()
    def __str__(self):
        pass
    def __clear_matrix(self, data, matrix, column):
        for i in range(0, data.shape[0]):
            cluster = data.loc[i, column]
            indexes = data.index[data[column] == cluster].tolist()
            matrix[i, indexes] = 0
            matrix[indexes, i] = 0
        return matrix
    def __get_UTM_zone(self, bounds):
        if math.ceil((bounds[2] + 180) / 6) - math.ceil((bounds[0] + 180) / 6) > 1:
            return 3857
        else:
            zone = math.ceil(((bounds[2] + bounds[0]) / 2 + 180) / 6)
            if bounds[3] >= 0:
                crs = int("326" + str(zone))
            else:
                crs = int("327" + str(zone))
            return crs
    def calculate(self, columns:list = ['lat', 'lon'], normalize:bool = False):
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
        if normalize == True:
            if any(item in ('lat', 'lon') for item in columns):
                warnings.warn('You are normalizing coordinate values. It may slightly change results.')
            for_matrix = data.loc[:,columns].apply(lambda x: (x-x.mean())/ x.std(), axis=0)
        else:
            for_matrix = data.loc[:,columns]
        
        distance_matrix = np.array(cdist(for_matrix, for_matrix, metric='euclidean'))
        self.matrix = np.copy(distance_matrix)
        
        # get nearest neighbours
        data['nearest1'] = np.argmin(ma.masked_array(distance_matrix, mask= distance_matrix==0), axis=1) + 1
        data['cluster1'] = np.argmin(ma.masked_array(distance_matrix, mask= distance_matrix==0), axis=1) + 1

        # grouping into clusters
        for i in range(0, data.shape[0]):
            data.loc[data['cluster1'] == i + 1, 'cluster1'] = data.loc[i, 'cluster1']

        # clearing matrix
        distance_matrix = self.__clear_matrix(data, distance_matrix, 'cluster1')

        # repeating clustering to get one big cluster
        lvl = 2
        while data[f'cluster{lvl-1}'].unique().shape[0] > 1:
            # get nearest neighbour of cluster
            data[f'nearest{lvl}'] = np.argmin(ma.masked_array(distance_matrix, mask= distance_matrix==0), axis=1) + 1
            data[f'nearest{lvl}_dist'] = np.min(ma.masked_array(distance_matrix, mask= distance_matrix==0), axis=1)

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
            for i in range(0, data.shape[0]):
                data.loc[data[f'cluster{lvl}'] == data.loc[i, f'cluster{lvl-1}'], f'cluster{lvl}'] = data.loc[i, f'cluster{lvl}']

            # clearing distance matrix
            distance_matrix = self.__clear_matrix(data, distance_matrix, f'cluster{lvl}')
            lvl += 1

        self.levels = lvl
        # linestrings for every connection level
        for i in range(1, lvl):
            for j in range(0, data.shape[0]):
                if data.loc[j, f'nearest{i}'] != -1:
                    data.loc[j, f'line{i}'] = LineString([data.loc[j, 'geometry'].centroid, data.loc[data['fid'] == data.loc[j, f'nearest{i}'], 'geometry'].values[0].centroid]).wkt
                else:
                    data.loc[j, f'line{i}'] = ''
        
        # counting connections for every point
        for i in range(0, data.shape[0]):
            to_ids = {x for lst in [data.loc[data[f'nearest{j}'] == i + 1, 'fid'].to_list() for j in range(1, self.levels)] for x in lst}
            from_ids = set([data.loc[i, f'nearest{j}'] for j in range(1, self.levels)]) - {-1}
            data.loc[i, 'connections'] =  len(to_ids | from_ids)

        # creating dendrite lines
        dendrite = gpd.GeoDataFrame(columns=['cluster', 'level', 'geometry'], geometry='geometry')
        for i in range(1, self.levels):
            lines = data.loc[data[f'line{i}'] != '', ['fid', f'nearest{i}', f'cluster{i}', f'line{i}']]
            lines.rename(columns={f'cluster{i}':'cluster',f'nearest{i}':'nearest'}, inplace=True)
            lines['level'] = i
            lines['geometry'] = gpd.GeoSeries.from_wkt(lines[f'line{i}'])
            dendrite = pd.concat([dendrite, gpd.GeoDataFrame(lines[['fid', 'nearest', 'cluster', 'level', 'geometry']], geometry='geometry')])
        
        #dendrite['length'] = dendrite.length
        #dendrite = dendrite[~((dendrite['length'] > (np.mean(dendrite.length) + 2 * np.std(dendrite.length))) & (dendrite['level'] > 2))]

        self.n_levels = lvl - 1
        self.dendrite = dendrite
        self.results = data
    def export_objects(self, out_file:str = 'dendrite_points.geojson') -> gpd.GeoDataFrame:
        """
        Exports source data with added columns to GeoJSON file.

        ----------

        Parameters
        ----------
        out_file : str
            path to output file
        """
        self.results.to_file(out_file, driver='GeoJSON', crs=self.crs)
        return self.results
    def export_dendrite(self, out_file:str = 'dendrite.geojson') -> gpd.GeoDataFrame:
        """
        Exports computed dendrite to GeoJSON file.

        ----------

        Parameters
        ----------
        out_file : str
            path to output file
        """
        self.dendrite.to_file(out_file, driver='GeoJSON', crs=self.crs)
        return self.dendrite
    def set_style(self, style:dict | str = 'default'):
        """
        Sets style for every map that will be generated later.

        ----------

        Parameters
        ----------
        style : dict
            dictionary containing style configuration of maps, e.g. markersize, cmap, line color and object color
        """
        if style != 'default':
            self.plot_style = style
        else:
            self.plot_style = {
                "markersize": 10,
                "cmap": 'jet',
                "line_color": '#222222',
                "object_color": '#ff0000'
            }
    def plot(self, level:int = None, lines:bool = True, style:dict = None, show:bool = True):
        """
        Displays map of computed dendrite and source objects.

        ----------

        Parameters
        ----------
        level : int
            only connections from smaller or equal level will be displayed on a map
        lines : bool
            if True, dendrite is plotted; if False, only source objects are plotted
        style : dict
            style configuration of map, e.g. markersize, cmap, line color and object color
        show : bool
            if True, map is displayed immediately; if False, map is returned and can be saved to variable
        """
        if style is None:
            style = self.plot_style
        else:
            style = self.plot_style | style
        dendrite = self.dendrite
        objects = self.results
        if level is not None:
            dendrite = dendrite[dendrite['level'] <= level]
        fig, ax = plt.subplots(figsize = (10, 10))
        if lines==True:
            for lvl, lwd in zip(range(1, max(dendrite['level']) + 1), np.arange(0.5, 2 + (1.5 / (max(dendrite['level']) + 1)), (1.5 / (max(dendrite['level']) + 1)))):
                dendrite[dendrite['level'] == lvl].plot(ax=ax, color=style["line_color"],  linewidth=lwd, zorder=5)

        if objects.geom_type[0] == 'Point' and level is not None:
            objects.plot(ax=ax, cmap=style["cmap"], markersize=style["markersize"],
            zorder=10, column=f'cluster{level}')
        elif objects.geom_type[0] == 'Point' and level is None:
            objects.plot(ax=ax, color=style["object_color"], zorder=10,
            markersize=(objects['connections'] - 0.75) * 2)
        elif objects.geom_type[0] == 'MultiPolygon' and level is not None:
            objects.plot(ax=ax, cmap=style["cmap"],
            zorder=1, column=f'cluster{level}')
        elif objects.geom_type[0] == 'MultiPolygon' and level is None:
            objects.plot(ax=ax, zorder=1,
            cmap='Reds', column='connections')
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        if show==True:
            plt.show()
        else:
            return fig

    def animate(self, out_file:str = 'dendrite.gif', frame_duration:int = 1, lines:bool = True, style:dict = None):
        """
        Creates GIF animation showing each step of creating dendrite (map of every level of dendrite connections).

        ----------

        Parameters
        ----------
        out_file : str
            path to output .gif file where animation will be saved
        frame_duration : int
            time of display of each frame in seconds
        lines : bool
            if True, dendrite is included; if False, only source objects are plotted
        style : dict
            style configuration of animation, e.g. markersize, cmap, line color and object color
        """
        dendrite = self.dendrite
        n_frames = np.max(dendrite["level"].unique())
        files = []
        frames = []
        for i in range(1, n_frames + 1):
            img = self.plot(level=i, lines=lines, style=style, show=False)
            plt.close()
            img.savefig(f'frame{i}.png')
            files.append(f'frame{i}.png')
            frames.append(imageio.imread(f'frame{i}.png'))
        
        imageio.mimsave(out_file, frames, duration=frame_duration)
        for file in files:
            os.remove(file)

        return f"GIF saved in {out_file}"


