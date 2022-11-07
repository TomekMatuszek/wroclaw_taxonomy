import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from wroclaw_taxonomy import create_dendrite

cities = gpd.read_file('data/citiesPL_pop.geojson', driver = 'GeoJSON')
dendrite = create_dendrite(
    in_file=cities,
    #columns=['pop', 'lat', 'lon'],
    normalize=False,
    out_file='dendrite.geojson',
    type='lines'
)
fig, ax = plt.subplots()
for lvl, lwd in zip(range(1, max(dendrite['level']) + 1), np.arange(0.5, 2 + (1.5 / (max(dendrite['level']) + 1)), (1.5 / (max(dendrite['level']) + 1)))):
    print(f'{lvl}: {lwd}')
    dendrite[dendrite['level'] == lvl].plot(ax=ax, color='#222222',  linewidth=lwd)

plt.show()