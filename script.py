import pandas as pd
import geopandas as gpd
import numpy as np
import numpy.ma as ma
from shapely.geometry import LineString

# funkcja czyszcząca macierz odległości
def clear_matrix(column):
    for i in range(0, data.shape[0]):
        cluster = data.loc[i, column]
        indexes = data.index[data[column] == cluster].tolist()
        distance_matrix[i, indexes] = 0
        distance_matrix[indexes, i] = 0

# wczytanie danych punktowych i stworzenie macierzy
data = gpd.read_file('citiesPL.geojson', driver = 'GeoJSON')
print(data)
data = data.to_crs(epsg=2180)
distance_matrix = np.array(data.geometry.apply(lambda x: data.distance(x).astype(np.int64)))
print(distance_matrix)

# wyznaczenie najbliższych sąsiadów
data['nearest1'] = np.argmin(ma.masked_array(distance_matrix, mask= distance_matrix==0), axis=1) + 1
data['cluster1'] = np.argmin(ma.masked_array(distance_matrix, mask= distance_matrix==0), axis=1) + 1

# zgrupowanie punktów w klastry
for i in range(0, data.shape[0]):
    data.loc[data['cluster1'] == i + 1, 'cluster1'] = data.loc[i, 'cluster1']

# czyszczenie macierzy ze stworzonych połączeń
clear_matrix('cluster1')

print(data[data['naz_glowna'].isin(['Aleksandrów Kujawski', 'Ciechocinek', 'Nieszawa', 'Toruń', 'Chełmża'])])

# powtarzanie łączenia aż do otrzymania jednego klastra
j = 2
while data[f'cluster{j-1}'].unique().shape[0] > 1:
    # wyznaczenie najbliższego punktu wraz z odległością po maskowaniu macierzy z zer
    data[f'nearest{j}'] = np.argmin(ma.masked_array(distance_matrix, mask= distance_matrix==0), axis=1) + 1
    data[f'nearest{j}_dist'] = np.min(ma.masked_array(distance_matrix, mask= distance_matrix==0), axis=1)
    print(data[data['naz_glowna'].isin(['Aleksandrów Kujawski', 'Ciechocinek', 'Nieszawa', 'Toruń', 'Chełmża'])])

    # pozostawienie tylko najbliższych odległości dla każdego klastra
    for i in data[f'cluster{j-1}'].unique():
        cluster = data.loc[data[f'cluster{j-1}'] == i, :]
        nearest = cluster.loc[cluster[f'nearest{j}_dist'] == np.min(cluster[f'nearest{j}_dist']), 'fid']
        data.loc[(data['fid'] != nearest.values[0]) & (data[f'cluster{j-1}'] == i), f'nearest{j}'] = -1

    print(data[data['naz_glowna'].isin(['Aleksandrów Kujawski', 'Ciechocinek', 'Nieszawa', 'Toruń', 'Chełmża'])])

    # przyporządkowanie numeru klastra do którego należy najbliższy sąsiad
    for i in range(0, data.shape[0]):
        if data.loc[i, f'nearest{j}'] == -1:
            continue
        else:
            nearest_cluster = data.loc[data['fid'] == data.loc[i, f'nearest{j}'], f'cluster{j-1}']
            data.loc[data[f'cluster{j-1}'] == data.loc[i, f'cluster{j-1}'], f'cluster{j}'] = nearest_cluster.values[0]

    print(data[data['naz_glowna'].isin(['Aleksandrów Kujawski', 'Ciechocinek', 'Nieszawa', 'Toruń', 'Chełmża'])])

    # zgrupowanie klastrów w większe klastry
    for i in range(0, data.shape[0]):
        data.loc[data[f'cluster{j}'] == data.loc[i, f'cluster{j-1}'], f'cluster{j}'] = data.loc[i, f'cluster{j}']

    print(data[data['naz_glowna'].isin(['Aleksandrów Kujawski', 'Ciechocinek', 'Nieszawa', 'Toruń', 'Chełmża', 'Skępe', 'Lipno'])])

    # czyszczenie macierzy
    clear_matrix(f'cluster{j}')
    j += 1

print(data[data['naz_glowna'].isin(['Aleksandrów Kujawski', 'Ciechocinek', 'Nieszawa', 'Toruń', 'Chełmża', 'Skępe', 'Lipno'])])
#data.to_file('nearest.geojson', driver = 'GeoJSON')

# tworzenie linestringów dla połączeń każdego rzędu
for i in range(1, 6):
    for j in range(0, data.shape[0]):
        if data.loc[j, f'nearest{i}'] != -1:
            data.loc[j, f'line{i}'] = LineString([data.loc[j, 'geometry'], data.loc[data['fid'] == data.loc[j, f'nearest{i}'], 'geometry'].values[0]]).wkt
        else:
            data.loc[j, f'line{i}'] = ''

print(data[data['naz_glowna'].isin(['Aleksandrów Kujawski', 'Ciechocinek', 'Nieszawa', 'Toruń', 'Chełmża', 'Skępe', 'Lipno'])])
data.to_csv('wkt.csv')
