import geopandas as gpd
from wroclaw_taxonomy import create_dendrite

cities = gpd.read_file('data/citiesPL_pop.geojson', driver = 'GeoJSON')
create_dendrite(
    in_file=cities,
    #columns=['pop', 'lat', 'lon'],
    out_file='dendrite.geojson',
    type='lines'
)