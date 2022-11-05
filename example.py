from wroclaw_taxonomy.wroclaw_taxonomy import create_dendrite

create_dendrite(
    in_file='citiesPL_pop.geojson',
    #columns=['pop', 'lat', 'lon'],
    out_file='dendrite.geojson',
    type='lines'
)
create_dendrite(
    in_file='citiesPL_pop.geojson',
    #columns=['pop', 'lat', 'lon'],
    out_file='dendrite_points.geojson',
    type='points'
)