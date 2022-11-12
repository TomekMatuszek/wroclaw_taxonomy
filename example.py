from wroclaw_taxonomy.wroclaw_taxonomy import create_dendrite

create_dendrite(
    in_file='data/citiesPL_pop.geojson',
    #columns=['pop', 'lat', 'lon'],
    normalize=False,
    out_file='dendrite.geojson',
    type='lines'
)
create_dendrite(
    in_file='data/citiesPL_pop.geojson',
    #columns=['pop', 'lat', 'lon'],
    normalize=False,
    out_file='dendrite_points.geojson',
    type='points'
)