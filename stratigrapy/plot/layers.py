"""Plotting layers"""

# MIT License

# Copyright (c) 2025 Guillaume Rongier

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import warnings
import numpy as np

from ..utils import reshape_to_match
from .cfuncs import mask_layer


################################################################################
# Base functions

def _get_variable_values(grid, var):
    """
    Gets the values for a variable of a StackedLayers.
    """
    if isinstance(var, np.ndarray):
        if len(var) == grid.number_of_nodes:
            return var[grid.core_nodes]
        elif var.ndim > 1 and var.shape[1] == grid.number_of_nodes:
            return var[:, grid.core_nodes]
        else:
            return var
    elif var == 'dz' or var == 'thickness':
        return grid.stacked_layers['_dz']
    elif var == 'composition':
        return grid.stacked_layers.composition
    elif var == 'most_frequent_class' or var == 'most_frequent':
        return grid.stacked_layers.most_frequent_class
    else:
        return grid.stacked_layers[var]


def _select_sublayers(layers, i, axis=2, indices=None):
    """
    Gets the values of a subselection of a stack of layers.
    """
    if layers.ndim == 1 or axis >= layers.ndim:
        return layers
    elif isinstance(i, int):
        slices = tuple(slice(None) if j != axis else i for j in range(layers.ndim))
        return layers[slices]
    elif callable(i):
        return i(layers, axis=axis)
    elif i == 'middle':
        slices = tuple(slice(None) if j != axis else layers.shape[axis]//2 for j in range(layers.ndim))
        return layers[slices]
    elif i == 'top' and indices is not None:
        return np.take_along_axis(layers, reshape_to_match(indices, layers.shape), axis=axis)[0]
    else:
        return None


def _get_layer_values(grid, var, i_layer, i_class):
    """
    Gets the values of a layer.
    """
    return _select_sublayers(_select_sublayers(_get_variable_values(grid, var),
                                               i_layer,
                                               axis=0,
                                               indices=grid.stacked_layers.surface_index),
                             i_class,
                             axis=1)


def _get_layer_elevation(grid):
    """
    Gets the elevation of a StackedLayers.
    """
    z = grid.at_node['topographic__elevation'][grid.core_nodes] - grid.stacked_layers.z

    return np.concatenate([z, grid.at_node['topographic__elevation'][grid.core_nodes][np.newaxis]])


def _plot_horizontal_slice(ax, grid, var, i_layer, i_class, vmin, vmax, shading, **kwargs):
    """
    Plots a horizontal slice through a StackedLayers on a RasterModelGrid.
    """
    x = grid.x_of_node[grid.core_nodes].reshape(grid.cell_grid_shape)
    y = grid.y_of_node[grid.core_nodes].reshape(grid.cell_grid_shape)
    values = _get_layer_values(grid, var, i_layer, i_class).reshape(grid.cell_grid_shape)

    vmin = None if 'norm' in kwargs else vmin
    vmax = None if 'norm' in kwargs else vmax

    return ax.pcolormesh(x, y, values, vmin=vmin, vmax=vmax, shading=shading, **kwargs)


def _plot_vertical_slice(ax, grid, var, i_x, i_y, i_class, vmin, vmax, shading, **kwargs):
    """
    Plots a vertical slice through a StackedLayers on a RasterModelGrid.
    """
    x = grid.x_of_node[grid.core_nodes].reshape(grid.cell_grid_shape)
    y = grid.y_of_node[grid.core_nodes].reshape(grid.cell_grid_shape)
    z = _get_layer_elevation(grid).reshape(-1, *grid.cell_grid_shape)
    dz = _select_sublayers(_get_variable_values(grid, 'dz'), np.sum, axis=2).reshape(-1, *grid.cell_grid_shape)
    if (var == '_dz' or var == 'dz' or var == 'thickness') and i_class == np.sum:
        values = dz
    else:
        values = _select_sublayers(_get_variable_values(grid, var), i_class, axis=2).reshape(-1, *grid.cell_grid_shape)

    if i_x is None:
        i_x = slice(None)
        if i_y == 'middle':
            i_y = grid.cell_grid_shape[0]//2
        coords = np.tile(x[i_y, i_x], (2, 1))
    elif i_y is None:
        i_y = slice(None)
        if i_x == 'middle':
            i_x = grid.cell_grid_shape[1]//2
        coords = np.tile(y[i_y, i_x], (2, 1))

    vmin = np.nanmin(values[..., i_y, i_x]) if vmin is None and 'norm' not in kwargs else vmin
    vmax = np.nanmax(values[..., i_y, i_x]) if vmax is None and 'norm' not in kwargs else vmax

    c = []
    for i in range(len(values)):
        layer = values[i, i_y, i_x].astype(float)
        mask_layer(layer, dz[i, i_y, i_x])
        ci = ax.pcolormesh(coords,
                           z[i:i + 2, i_y, i_x],
                           np.tile(layer, (2, 1)),
                           shading=shading,
                           vmin=vmin,
                           vmax=vmax,
                           **kwargs)
        c.append(ci)

    return c


################################################################################
# Plot

def plot_layers(
    ax,
    grid,
    var,
    i_x='middle',
    i_y=None,
    i_layer=None,
    i_class=None,
    vmin=None,
    vmax=None,
    shading='gouraud',
    **kwargs,
):
    """Plot a horizontal or vertical slice through a stack of layers on a raster
    grid.

    Visualization follows the tie-centered approach described by Tetzlaff (2023);
    channel erosion and deposition is not properly handled in the vertical cross-
    sections (see figure 12 of Tetzlaff (2023)).

    Parameters
    ----------
    ax : matplotlib axes object
        The axis of a figure on which to plot the layer(s).
    grid : RasterModelGrid
        The grid containing the StackedLayers to plot.
    var : str or array-like
        The variable to plot as color, which can be on the StackedLayers or as
        a separate array. The keywords 'dz' or 'thickness' can be used to plot
        the layer thickness; the keyword 'composition' can be used to plot the
        proportion of each lithology.
    i_x : int, str, or callable, optional
        The index at which to slice vertically the layers along the grid's x
        axis. The keyword 'middle' will slice at the middle of the axis. A
        callable will merge the all the slices along the axis together (e.g.,
        numpy.mean or numpy.sum). A slice can only be made along a single axis,
        so `i_x` is incompatible with `i_y` and `i_layer`.
    i_y : int, str, or callable, optional
        The index at which to slice vertically the layers along the grid's y
        axis. The keyword 'middle' will slice at the middle of the axis. A
        callable will merge the all the slices along the axis together (e.g.,
        numpy.mean or numpy.sum). A slice can only be made along a single axis,
        so `i_y` is incompatible with `i_x` and `i_layer`.
    i_layer : int, str, or callable, optional
        The index at which to slice horizontally the layers, i.e., to plot a
        given layer in map view. The keyword 'middle' will slice at the middle
        of the axis; the keyword 'top' will show the values at the topographic
        surface. A callable will merge the all the slices along the axis together
        (e.g., numpy.mean or numpy.sum). A slice can only be made  along a single
        axis, so `i_layer` is incompatible with `i_x` and `i_y`.
    i_class : int or callable, optional
        The index to select the lithology to plot. A callable will merge the all
        the lithologies together (e.g., numpy.mean or numpy.sum).
    vmin : float, optional
        The minimum value to use when plotting the variable. If None, uses the
        minimum value of the variable.
    vmax : float, optional
        The maximum value to use when plotting the variable. If None, uses the
        maximum value of the variable.
    shading : str
        The fill style for the quadrilateral, which can be 'flat', 'nearest',
        'gouraud', or 'auto'. By default, it reproduces the tie-centered
        scheme of Tetzlaff (2023) by using 'gouraud'.
    **kwargs : `matplotlib.axes.Axes.pcolormesh` properties
        Other keyword arguments are pcolormesh properties, see
        `matplotlib.axes.Axes.pcolormesh` for a list of valid properties.

    Returns
    -------
    c : matplotlib.collections.QuadMesh or list
        The mesh(es) that constitute(s) the plot.

    Reference
    ---------
    Tetzlaff, D. (2023)
        Stratigraphic forward modeling software package for research and education
        https://arxiv.org/abs/2302.05272
    """
    if i_x is not None and i_y is not None:
        warnings.warn("Both `i_x` and `i_y` are not None, but a slice can only be made along a single axis. `i_y` will be turned to None.")
        i_y = None
    if i_x is not None and i_layer is not None:
        warnings.warn("Both `i_x` and `i_layer` are not None, but a slice can only be made along a single axis. `i_layer` will be turned to None.")
        i_layer = None
    if i_y is not None and i_layer is not None:
        warnings.warn("Both `i_y` and `i_layer` are not None, but a slice can only be made along a single axis. `i_layer` will be turned to None.")
        i_layer = None

    if i_x is not None or i_y is not None:
        return _plot_vertical_slice(ax, grid, var, i_x, i_y, i_class, vmin, vmax, shading, **kwargs)
    elif i_layer is not None:
        return _plot_horizontal_slice(ax, grid, var, i_layer, i_class, vmin, vmax, shading, **kwargs)
    else:
        raise Exception("At least one of `i_x`, `i_y`, or `i_layer` must not be None.")


class RasterModelGridLayerPlotterMixIn:
    """MixIn that provides layer plotting functionality to a raster grid.

    Inhert from this class to provide a ModelDataFields object with the method
    function, ``plot_layers``, that plots a slice through a StackedLayers.
    """

    def plot_layers(
        self,
        ax,
        var,
        i_x='middle',
        i_y=None,
        i_layer=None,
        i_class=None,
        vmin=None,
        vmax=None,
        shading='gouraud',
        **kwargs,
    ):
        """Plot a horizontal or vertical slice through a StackedLayers. This is
        a wrapper for `plot.plot_layers`, and its uses the same keywords.

        Visualization follows the tie-centered approach described by Tetzlaff (2023);
        channel erosion and deposition is not properly handled in the vertical cross-
        sections (see figure 12 of Tetzlaff (2023)).

        Parameters
        ----------
        ax : matplotlib axes object
            The axis of a figure on which to plot the layer(s).
        grid : RasterModelGrid
            The grid containing the StackedLayers to plot.
        var : str or array-like
            The variable to plot as color, which can be on the StackedLayers or as
            a separate array. The keywords 'dz' or 'thickness' can be used to plot
            the layer thickness; the keyword 'composition' can be used to plot the
            proportion of each lithology.
        i_x : int, str, or callable, optional
            The index at which to slice vertically the layers along the grid's x
            axis. The keyword 'middle' will slice at the middle of the axis. A
            callable will merge the all the slices along the axis together (e.g.,
            numpy.mean or numpy.sum). A slice can only be made along a single axis,
            so `i_x` is incompatible with `i_y` and `i_layer`.
        i_y : int, str, or callable, optional
            The index at which to slice vertically the layers along the grid's y
            axis. The keyword 'middle' will slice at the middle of the axis. A
            callable will merge the all the slices along the axis together (e.g.,
            numpy.mean or numpy.sum). A slice can only be made along a single axis,
            so `i_y` is incompatible with `i_x` and `i_layer`.
        i_layer : int, str, or callable, optional
            The index at which to slice horizontally the layers, i.e., to plot a
            given layer in map view. The keyword 'middle' will slice at the middle
            of the axis; the keyword 'top' will show the values at the topographic
            surface. A callable will merge the all the slices along the axis together
            (e.g., numpy.mean or numpy.sum). A slice can only be made  along a single
            axis, so `i_layer` is incompatible with `i_x` and `i_y`.
        i_class : int or callable, optional
            The index to select the lithology to plot. A callable will merge the all
            the lithologies together (e.g., numpy.mean or numpy.sum).
        vmin : float, optional
            The minimum value to use when plotting the variable. If None, uses the
            minimum value of the variable.
        vmax : float, optional
            The maximum value to use when plotting the variable. If None, uses the
            maximum value of the variable.
        shading : str
            The fill style for the quadrilateral, which can be 'flat', 'nearest',
            'gouraud', or 'auto'. By default, it reproduces the tie-centered
            scheme of Tetzlaff (2023) by using 'gouraud'.
        **kwargs : `matplotlib.axes.Axes.pcolormesh` properties
            Other keyword arguments are pcolormesh properties, see
            `matplotlib.axes.Axes.pcolormesh` for a list of valid properties.

        Returns
        -------
        c : matplotlib.collections.QuadMesh or list
            Mesh(es) that constitute(s) the plot.

        Reference
        ---------
        Tetzlaff, D. (2023)
            Stratigraphic forward modeling software package for research and education
            https://arxiv.org/abs/2302.05272

        See Also
        --------
        stratigrapy.plot.plot_layers
        """
        return plot_layers(ax, self, var, i_x, i_y, i_layer, i_class, vmin, vmax, shading, **kwargs)


def extract_tie_centered_layers(grid, var, i_class=None, axis=2, fill_nan=False):
    """Extract the layers coordinates and values of a variable for plotting, in
    particular with PyVista.

    Visualization follows the tie-centered approach described by Tetzlaff (2023);
    channel erosion and deposition is not properly handled in the vertical cross-
    sections (see figure 12 of Tetzlaff (2023)).

    Parameters
    ----------
    grid : RasterModelGrid
        The grid containing the StackedLayers to plot.
    var : str or array-like
        The variable to plot as color, which can be on the StackedLayers or as
        a separate array. The keywords 'dz' or 'thickness' can be used to plot
        the layer thickness; the keyword 'composition' can be used to plot the
        proportion of each lithology.
    i_class : int or callable, optional
        The index to select the lithology to plot. A callable will merge the all
        the lithologies together (e.g., numpy.mean or numpy.sum).
    axis : int
        The axis along which to post-process the layers to manage pinching-out
        cells. By default, the x-axis is used.
    fill_nan: bool
        If True, fills the NaN values of a layer with the values of the layer
        below.

    Returns
    -------
    x : ndarray
        The coordinates along the x-axis of the 3D grid making the layers.
    y : ndarray
        The coordinates along the y-axis of the 3D grid making the layers.
    z : ndarray
        The coordinates along the z-axis of the 3D grid making the layers.
    layers: ndarray
        The values of the variable `var` based on a tie-centered scheme. This
        array has the same size as `x`, `y`, and `z` along its last two
        dimensions (the horizontal dimensions y and x), but one element less
        along its first dimension (the vertical dimension z).

    Reference
    ---------
    Tetzlaff, D. (2023)
        Stratigraphic forward modeling software package for research and education
        https://arxiv.org/abs/2302.05272
    """
    z = _get_layer_elevation(grid).reshape(-1, *grid.cell_grid_shape)
    y = grid.y_of_node[grid.core_nodes].reshape(grid.cell_grid_shape)
    y = np.tile(y, (len(z), 1, 1))
    x = grid.x_of_node[grid.core_nodes].reshape(grid.cell_grid_shape)
    x = np.tile(x, (len(z), 1, 1))
    dz = _select_sublayers(_get_variable_values(grid, 'dz'), np.sum, axis=2).reshape(-1, *grid.cell_grid_shape)
    if (var == '_dz' or var == 'dz' or var == 'thickness') and i_class == np.sum:
        layers = dz
    else:
        layers = _select_sublayers(_get_variable_values(grid, var), i_class, axis=2).reshape(-1, *grid.cell_grid_shape)

    for l in range(layers.shape[0]):
        for c in range(layers.shape[axis]):
            slices = tuple(slice(None) if i != axis else c for i in range(1, layers.ndim))
            mask_layer(layers[l, *slices], dz[l, *slices])
            if fill_nan == True and l > 0:
                layers[l, np.isnan(layers[l])] = layers[l - 1, np.isnan(layers[l])]

    return x, y, z, layers
