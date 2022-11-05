# wroclaw-taxonomy package

## About

This package consists of one function called `create_dendrite` which groupes set of points into clusters based on Wroclaw taxonomy method.

## Installation

You can install this package by using below pip install command in terminal:

`pip install git+https://github.com/TomekMatuszek/wroclaw-taxonomy.git`

## Examples

```python
from wroclaw_taxonomy import create_dendrite

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
```