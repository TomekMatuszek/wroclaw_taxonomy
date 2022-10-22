import pandas as pd
import geopandas as gpd
import numpy as np
import numpy.ma as ma
from shapely.geometry import LineString

# funkcja czyszcząca macierz odległości
def clear_matrix(data, matrix, column):
    for i in range(0, data.shape[0]):
        cluster = data.loc[i, column]
        indexes = data.index[data[column] == cluster].tolist()
        matrix[i, indexes] = 0
        matrix[indexes, i] = 0

def create_dendrite(in_file, crs=4326, out_file='dendrite.geojson', type='lines'):
    # wczytanie danych punktowych i stworzenie macierzy
    data = gpd.read_file(in_file, driver = 'GeoJSON')
    print(data)
    data = data.to_crs(epsg=2180)
    distance_matrix = np.array(data.geometry.apply(lambda x: data.distance(x).astype(np.int64)))

    # wyznaczenie najbliższych sąsiadów
    data['nearest1'] = np.argmin(ma.masked_array(distance_matrix, mask= distance_matrix==0), axis=1) + 1
    data['cluster1'] = np.argmin(ma.masked_array(distance_matrix, mask= distance_matrix==0), axis=1) + 1

    # zgrupowanie punktów w klastry
    for i in range(0, data.shape[0]):
        data.loc[data['cluster1'] == i + 1, 'cluster1'] = data.loc[i, 'cluster1']

    # czyszczenie macierzy ze stworzonych połączeń
    clear_matrix(data, distance_matrix, 'cluster1')

    print(data[data['naz_glowna'].isin(['Aleksandrów Kujawski', 'Ciechocinek', 'Nieszawa', 'Toruń', 'Chełmża'])])

    # powtarzanie łączenia aż do otrzymania jednego klastra
    j = 2
    while data[f'cluster{j-1}'].unique().shape[0] > 1:
        # wyznaczenie najbliższego punktu wraz z odległością po maskowaniu macierzy z zer
        data[f'nearest{j}'] = np.argmin(ma.masked_array(distance_matrix, mask= distance_matrix==0), axis=1) + 1
        data[f'nearest{j}_dist'] = np.min(ma.masked_array(distance_matrix, mask= distance_matrix==0), axis=1)

        # pozostawienie tylko najbliższych odległości dla każdego klastra
        for i in data[f'cluster{j-1}'].unique():
            cluster = data.loc[data[f'cluster{j-1}'] == i, :]
            nearest = cluster.loc[cluster[f'nearest{j}_dist'] == np.min(cluster[f'nearest{j}_dist']), 'fid']
            data.loc[(data['fid'] != nearest.values[0]) & (data[f'cluster{j-1}'] == i), f'nearest{j}'] = -1

        # przyporządkowanie numeru klastra do którego należy najbliższy sąsiad
        for i in range(0, data.shape[0]):
            if data.loc[i, f'nearest{j}'] == -1:
                continue
            else:
                nearest_cluster = data.loc[data['fid'] == data.loc[i, f'nearest{j}'], f'cluster{j-1}']
                data.loc[data[f'cluster{j-1}'] == data.loc[i, f'cluster{j-1}'], f'cluster{j}'] = nearest_cluster.values[0]

        # zgrupowanie klastrów w większe klastry
        for i in range(0, data.shape[0]):
            data.loc[data[f'cluster{j}'] == data.loc[i, f'cluster{j-1}'], f'cluster{j}'] = data.loc[i, f'cluster{j}']

        # czyszczenie macierzy
        clear_matrix(data, distance_matrix, f'cluster{j}')
        j += 1

    # tworzenie linestringów dla połączeń każdego rzędu
    for i in range(1, 6):
        for j in range(0, data.shape[0]):
            if data.loc[j, f'nearest{i}'] != -1:
                data.loc[j, f'line{i}'] = LineString([data.loc[j, 'geometry'], data.loc[data['fid'] == data.loc[j, f'nearest{i}'], 'geometry'].values[0]]).wkt
            else:
                data.loc[j, f'line{i}'] = ''

    print(data[data['naz_glowna'].isin(['Aleksandrów Kujawski', 'Ciechocinek', 'Nieszawa', 'Toruń', 'Chełmża', 'Skępe', 'Lipno'])])
    #data.to_csv('wkt.csv')

    dendrite = gpd.GeoDataFrame(columns=['cluster', 'level', 'geometry'], geometry='geometry')
    for i in range(1, 6):
        lines = data.loc[data[f'line{i}'] != '', ['fid', f'nearest{i}', f'cluster{i}', f'line{i}']]
        lines.rename(columns={f'cluster{i}':'cluster',f'nearest{i}':'nearest'}, inplace=True)
        lines['level'] = i
        lines['geometry'] = gpd.GeoSeries.from_wkt(lines[f'line{i}'])
        dendrite = dendrite.append(gpd.GeoDataFrame(lines[['fid', 'nearest', 'cluster', 'level', 'geometry']], geometry='geometry'))
    
    if type == 'lines':
        print(dendrite)
        dendrite.to_file(out_file, driver='GeoJSON', crs=2180)
    elif type == 'points':
        print(data)
        for i in range(0, data.shape[0]):
            to = data.loc[data['nearest1'] == i + 1, :].shape[0] + data.loc[data['nearest2'] == i + 1, :].shape[0] + data.loc[data['nearest3'] == i + 1, :].shape[0] + data.loc[data['nearest4'] == i + 1, :].shape[0] + data.loc[data['nearest5'] == i + 1, :].shape[0]
            conns = 1 + (1 if data.loc[i, 'nearest2'] != -1 else 0) + (1 if data.loc[i, 'nearest3'] != -1 else 0) + (1 if data.loc[i, 'nearest4'] != -1 else 0) + (1 if data.loc[i, 'nearest5'] != -1 else 0)
            data.loc[i, 'connections'] =  to + conns
        print(data[data['naz_glowna'].isin(['Aleksandrów Kujawski', 'Ciechocinek', 'Nieszawa', 'Toruń', 'Chełmża', 'Skępe', 'Lipno'])])
        data.to_file(out_file, driver='GeoJSON', crs=2180)

create_dendrite(in_file='citiesPL.geojson', out_file='dendrite.geojson', type='lines')
#create_dendrite(in_file='citiesPL.geojson', out_file='dendrite_points.geojson', type='points')