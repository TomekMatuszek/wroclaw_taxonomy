# wroclaw-taxonomy package

## About

This package consists of one function called `create_dendrite` which groupes set of points into clusters based on Wroclaw taxonomy method.

## Installation

You can install this package by using below pip install command in terminal:

`pip install git+https://github.com/TomekMatuszek/wroclaw-taxonomy.git`

## Examples

```python
from wroclaw_taxonomy import Dendrite

dendrite = Dendrite(src='data/citiesPL_pop.geojson')
dendrite.calculate(columns=['lat', 'lon'], normalize=False)

dendrite.export_objects(out_file='dendrite_points.geojson')
dendrite.export_dendrite(out_file='dendrite.geojson')

dendrite.plot()
```

[img/dendrite.png](img/dendrite.png)