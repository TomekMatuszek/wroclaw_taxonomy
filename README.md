# wroclaw_taxonomy package

## About

This package consists of one class called `Dendrite` which enables user to create dendrite out of set of points based on Wroclaw taxonomy method.
It uses euclidean distance to group objects into possibly similar groups.
In the next steps, each group is merged with the other closest group until the whole dataset is combined into one coherent dendrite.

User can export results to GeoJSON file or visualise them using `plot()` and `animate()`.

## Installation

You can install this package from PyPI by typing in terminal:

`pip install wroclaw_taxonomy`

Or download it directly from source repo:

`pip install git+https://github.com/TomekMatuszek/wroclaw_taxonomy.git`

## Example

Basic workflow:

```python
from wroclaw_taxonomy import Dendrite

dendrite = Dendrite(src='data/citiesPL_pop.geojson')
dendrite.calculate(columns=['lat', 'lon'], normalize=False)

dendrite.export_objects(out_file='dendrite_points.geojson')
dendrite.export_dendrite(out_file='dendrite.geojson')

dendrite.plot()
```

![](https://github.com/TomekMatuszek/wroclaw_taxonomy/blob/35c8045b73ee65029bdb1d9afc5ed75f6a6e136c/img/dendrite.png)

Customizing plot:

```python
dendrite.plot(
    level=1, lines=True,
    style = {
        "markersize": 20,
        "cmap": 'gist_rainbow',
        "line_color": '#000000',
        "object_color": '#0000ff'
    }
)
```

![](https://github.com/TomekMatuszek/wroclaw_taxonomy/blob/35c8045b73ee65029bdb1d9afc5ed75f6a6e136c/img/dendrite_custom.png)

Animation showing every stage of the dendrite creation:

```python
dendrite.animate(out_file='dendrite.gif', frame_duration=1)
```

![](https://github.com/TomekMatuszek/wroclaw_taxonomy/blob/35c8045b73ee65029bdb1d9afc5ed75f6a6e136c/img/dendrite.gif)

More examples can be found in file [examples.ipynb](https://github.com/TomekMatuszek/wroclaw_taxonomy/blob/main/examples.ipynb)