# StratigraPy

StratigraPy is a Python package for stratigraphic modeling based on [Landlab](https://github.com/landlab/landlab). Similarly to Landlab, StratigraPy favors flexibility for prototyping and teaching rather than speed.

StratigraPy is still experimental and under heavy development.

[![Click to watch the video](https://img.youtube.com/vi/THp7vKp5ha4/maxresdefault.jpg)](https://www.youtube.com/watch?v=THp7vKp5ha4)

## Installation

You can directly install StratigraPy from pip:

    pip install stratigrapy

Or from GitHub using pip:

    pip install git+https://github.com/grongier/stratigrapy.git

To run the Jupyter notebook in [examples](examples) you will also need tqdm, cmocean, pyvista, trame, trame-vtk, trame-vuetify, and jupyter, which you can install from pip too:

    pip install tqdm cmocean pyvista trame trame-vtk trame-vuetify jupyter

You can also install everything using conda and the `environment.yml` file included in this repository:

    conda env create -f environment.yml

## Usage

StratigraPy follows Landlab's API and you can check [Landlab's extensive documentation](https://landlab.csdms.io) to get an idea of its features. Here's a very basic example of setting up and running a model:

```
import numpy as np
import matplotlib.pyplot as plt
from stratigrapy import RasterModelGrid
from stratigrapy.components import GravityDrivenRouter

# Create the grid with two sediment classes
grid = RasterModelGrid((50, 50),
                       xy_spacing=(500., 500.),
                       number_of_classes=2,
                       initial_allocation=1020,
                       number_of_layers_to_fuse=20,
                       number_of_top_layers=20)
grid.set_closed_boundaries_at_grid_edges(True, True, True, False)

# Define the initial topography
elevation = grid.add_zeros('topographic__elevation', at='node', clobber=True)
elevation[grid.y_of_node > 15000.] = 100.

# Define the bathymetry
bathymetry = grid.add_zeros('bathymetric__depth', at='node', clobber=True)

# Define a diffusion component
gdr = GravityDrivenRouter(grid, diffusivity_cont=[2e-1, 2e-2])

# Run the simulation
for i in range(1000):
    gdr.run_one_step(100.)
    grid.stacked_layers.fuse(time=np.mean)
grid.stacked_layers.fuse(finalize=True, time=np.mean)

# Plot a slice through the domain with sediments and bedrock
fig, ax = plt.subplots(figsize=(8, 4))
pc = grid.plot_layers(ax, 'composition', i_class=1, mask_wedges=True, cmap='pink', zorder=2)
fig.colorbar(pc[0], ax=ax, label='Fraction of the second sediment class')
raster_y = grid.y_of_node[grid.core_nodes].reshape(grid.cell_grid_shape)[:, 24]
raster_z = grid.at_node['topographic__elevation'][grid.core_nodes].reshape(grid.cell_grid_shape)[:, 24]
ymin, ymax = ax.get_ylim()
ax.fill_between(raster_y, raster_z, ymin, color='#d9d9d9', zorder=1)
ax.set(xlabel='y (m)', ylabel='z (m)', ylim=(ymin, ymax));
```

You can find more complete examples in the folder [examples](examples).

## Citation

If you use StratigraPy in your research, please cite the original article(s) describing the method(s) you used (see the docstrings for the references). An acknowledgment of StratigraPy's use is always appreciated.

## Credits

This software was written by:

| [Guillaume Rongier](https://github.com/grongier) <br>[![ORCID Badge](https://img.shields.io/badge/ORCID-A6CE39?logo=orcid&logoColor=fff&style=flat-square)](https://orcid.org/0000-0002-5910-6868)</br> |
| :---: |

## License

Copyright notice: Technische Universiteit Delft hereby disclaims all copyright interest in the program StratigraPy written by the Author(s). Prof.dr.ir. S.G.J. Aarninkhof, Dean of the Faculty of Civil Engineering and Geosciences

&#169; 2025, Guillaume Rongier

This work is licensed under a MIT OSS licence, see [LICENSE](LICENSE) for more information.
