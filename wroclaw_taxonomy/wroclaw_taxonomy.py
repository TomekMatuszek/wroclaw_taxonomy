import pandas as pd
import geopandas as gpd
import numpy as np
import numpy.ma as ma
from shapely.geometry import LineString
from scipy.spatial.distance import cdist

# funkcja czyszcząca macierz odległości
def clear_matrix(data, matrix, column):
    for i in range(0, data.shape[0]):
        cluster = data.loc[i, column]
        indexes = data.index[data[column] == cluster].tolist()
        matrix[i, indexes] = 0
        matrix[indexes, i] = 0

def create_dendrite(in_file, columns=['lat', 'lon'], out_file='dendrite.geojson', type='lines'):
    # wczytanie danych punktowych
    if isinstance(in_file, gpd.GeoDataFrame):
        data = in_file
    elif isinstance(in_file, str):
        data = gpd.read_file(in_file, driver = 'GeoJSON')
    else:
        raise TypeError('Argument in_file has to be either string or GeoDataFrame')
    
    # zmiana układu współrzędnych
    if data.crs.to_authority()[1] != '4326':
        data.to_crs(epsg=4326, inplace=True)
    zone = round((data.centroid.x[0] + 180) / 6)
    if data.centroid.y[0] >= 0:
        crs = int("326" + str(zone))
    else:
        crs = int("327" + str(zone))

    #print(data)
    data.to_crs(epsg=crs, inplace=True)
    data['fid'] = [i for i in range(1, data.shape[0] + 1)]
    data['lat'] = data.centroid.y
    data['lon'] = data.centroid.x
    print(data)

    assert isinstance(columns, list), 'Argument columns has to be a list'
    # stworzenie macierzy odległości
    #distance_matrix = np.array(data.geometry.apply(lambda x: data.distance(x).astype(np.int64)))
    distance_matrix = np.array(cdist(data.loc[:,columns], data.loc[:,columns], metric='euclidean'))
    print(distance_matrix)
    
    # wyznaczenie najbliższych sąsiadów
    data['nearest1'] = np.argmin(ma.masked_array(distance_matrix, mask= distance_matrix==0), axis=1) + 1
    data['cluster1'] = np.argmin(ma.masked_array(distance_matrix, mask= distance_matrix==0), axis=1) + 1

    # zgrupowanie punktów w klastry
    for i in range(0, data.shape[0]):
        data.loc[data['cluster1'] == i + 1, 'cluster1'] = data.loc[i, 'cluster1']

    # czyszczenie macierzy ze stworzonych połączeń
    clear_matrix(data, distance_matrix, 'cluster1')

    #print(data[data['naz_glowna'].isin(['Aleksandrów Kujawski', 'Ciechocinek', 'Nieszawa', 'Toruń', 'Chełmża'])])

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
        clear_matrix(data, distance_matrix, f'cluster{lvl}')
        lvl += 1

    # tworzenie linestringów dla połączeń każdego rzędu
    for i in range(1, lvl):
        for j in range(0, data.shape[0]):
            if data.loc[j, f'nearest{i}'] != -1:
                data.loc[j, f'line{i}'] = LineString([data.loc[j, 'geometry'].centroid, data.loc[data['fid'] == data.loc[j, f'nearest{i}'], 'geometry'].values[0].centroid]).wkt
            else:
                data.loc[j, f'line{i}'] = ''

    #print(data[data['naz_glowna'].isin(['Aleksandrów Kujawski', 'Ciechocinek', 'Nieszawa', 'Toruń', 'Chełmża', 'Skępe', 'Lipno'])])

    # tworzenie warstwy liniowej z połączeniami dla każdego poziomu
    dendrite = gpd.GeoDataFrame(columns=['cluster', 'level', 'geometry'], geometry='geometry')
    for i in range(1, lvl):
        lines = data.loc[data[f'line{i}'] != '', ['fid', f'nearest{i}', f'cluster{i}', f'line{i}']]
        lines.rename(columns={f'cluster{i}':'cluster',f'nearest{i}':'nearest'}, inplace=True)
        lines['level'] = i
        lines['geometry'] = gpd.GeoSeries.from_wkt(lines[f'line{i}'])
        dendrite = dendrite.append(gpd.GeoDataFrame(lines[['fid', 'nearest', 'cluster', 'level', 'geometry']], geometry='geometry'))
    
    # eksport warstwy wynikowej
    if type == 'lines':
        #print(dendrite)
        dendrite.to_file(out_file, driver='GeoJSON', crs=crs)
        return dendrite
    elif type == 'points':
        #print(data)
        # liczenie połączeń rozchodzących się z punktu
        for i in range(0, data.shape[0]):
            to_ids = {x for lst in [data.loc[data[f'nearest{j}'] == i + 1, 'fid'].to_list() for j in range(1, lvl)] for x in lst}
            from_ids = set([data.loc[i, f'nearest{j}'] for j in range(1, lvl)]) - {-1}
            data.loc[i, 'connections'] =  len(to_ids | from_ids)
        data.to_file(out_file, driver='GeoJSON', crs=crs)
        return data
