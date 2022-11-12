from wroclaw_taxonomy.wroclaw_taxonomy import create_dendrite

create_dendrite(
    in_file='data/powiaty_dane.geojson',
    columns=['bezrobocie-TABLICA_[%]', 'stan ludności-TABLICA_[osoba]', 'wynagrodzenia-TABLICA_[%]'],
    normalize=True,
    out_file='dendrite_points2.geojson',
    type='points'
)