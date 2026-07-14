"""Plotting layers"""

# MIT License

# Copyright (c) 2025-2026 Guillaume Rongier

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


def _get_interface_values(values):
    """
    Gets the averages of the tie values of consecutive layers onto their shared
    interface, for node-centered display.
    """
    values_bottom = values.copy()
    values_top = values.copy()
    below, above = values[:-1], values[1:]
    interface = np.where(
        np.isnan(below),
        above,
        np.where(np.isnan(above), below, 0.5 * (below + above)),
    )
    values_bottom[1:] = interface
    values_top[:-1] = interface

    return values_bottom, values_top


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
    elif var == "dz" or var == "thickness":
        return grid.stacked_layers["_dz"]
    elif var == "composition":
        return grid.stacked_layers.composition
    elif var == "most_frequent_class" or var == "most_frequent":
        return grid.stacked_layers.most_frequent_class
    else:
        return grid.stacked_layers[var]


def _select_sublayers(layers, i, axis=2, dz=None, indices=None):
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
    elif i == "middle":
        slices = tuple(
            slice(None) if j != axis else layers.shape[axis] // 2
            for j in range(layers.ndim)
        )
        return layers[slices]
    elif i == "weighted_mean" and dz is not None:
        thickness = np.sum(dz, axis=axis, keepdims=True)
        composition = np.zeros_like(dz)
        np.divide(dz, thickness, out=composition, where=thickness > 0.0)
        return np.sum(layers * composition, axis=axis)
    elif i == "top" and indices is not None:
        return np.take_along_axis(
            layers, reshape_to_match(indices, layers.shape), axis=axis
        )[0]
    else:
        return None


def _get_layer_field(grid, var, i_class, mask_value):
    """
    Gets the values of a variable and the layer thicknesses of a StackedLayers.
    """
    _dz = _get_variable_values(grid, "dz")
    dz = _select_sublayers(_dz, np.sum, axis=2).reshape(-1, *grid.cell_grid_shape)
    if (
        isinstance(var, np.ndarray) == False
        and (var == "_dz" or var == "dz" or var == "thickness")
        and i_class == np.sum
    ):
        values = dz.astype(float)
    else:
        values = np.array(
            _select_sublayers(_get_variable_values(grid, var), i_class, axis=2, dz=_dz),
            dtype=float,
        ).reshape(-1, *grid.cell_grid_shape)
    if mask_value is not None:
        values[values == mask_value] = np.nan

    return values, dz


def _get_layer_elevation(grid):
    """
    Gets the elevation of a StackedLayers.
    """
    z = grid.at_node["topographic__elevation"][grid.core_nodes] - grid.stacked_layers.z

    return np.concatenate(
        [z, grid.at_node["topographic__elevation"][grid.core_nodes][np.newaxis]]
    )


def _get_depositional_elevation(grid):
    """
    Gets the absolute elevation of each layer's depositional base and top in a
    StackedLayers.
    """
    layers = grid.stacked_layers
    base = grid.at_node["topographic__elevation"][grid.core_nodes] - layers.thickness
    return base + layers.z_bottom, base + layers.z_top


################################################################################
# 2D plotting


def _get_layer_values(grid, var, i_layer, i_class):
    """
    Gets the values of a layer.
    """
    # surface_index is an index into the allocated stacks (it includes the
    # offset of the first layer), while the arrays sliced from the StackedLayers
    # start at the first layer, so the offset must be removed
    surface_index = grid.stacked_layers.surface_index - grid.stacked_layers._first_layer
    if var == "composition" and isinstance(i_layer, (float, np.floating)):
        layers = grid.stacked_layers.get_superficial_composition(i_layer)
    else:
        layers = _select_sublayers(
            _get_variable_values(grid, var), i_layer, axis=0, indices=surface_index
        )

    if i_class == "weighted_mean":
        dz = _select_sublayers(
            _get_variable_values(grid, "_dz"),
            i_layer,
            axis=0,
            indices=surface_index,
        )
    else:
        dz = None

    return _select_sublayers(layers, i_class, axis=1, dz=dz)


def _plot_horizontal_slice(
    ax, grid, var, i_layer, i_class, centering, mask_value, vmin, vmax, **kwargs
):
    """
    Plots a horizontal slice through a StackedLayers on a RasterModelGrid.
    """
    x = grid.x_of_node[grid.core_nodes].reshape(grid.cell_grid_shape)
    y = grid.y_of_node[grid.core_nodes].reshape(grid.cell_grid_shape)
    values = _get_layer_values(grid, var, i_layer, i_class).reshape(
        grid.cell_grid_shape
    )
    if mask_value is not None:
        values[values == mask_value] = np.nan

    vmin = None if "norm" in kwargs else vmin
    vmax = None if "norm" in kwargs else vmax
    shading = kwargs.pop("shading", "nearest" if centering == "cell" else "gouraud")

    return ax.pcolormesh(x, y, values, vmin=vmin, vmax=vmax, shading=shading, **kwargs)


def _plot_vertical_slice(
    ax,
    grid,
    var,
    i_x,
    i_y,
    i_class,
    centering,
    mask_wedges,
    mask_null_layers,
    mask_value,
    vmin,
    vmax,
    **kwargs,
):
    """
    Plots a vertical slice through a StackedLayers on a RasterModelGrid as a
    single quadrilateral mesh.

    In the quadrilateral mesh, every layer owns its own bottom and top row of
    vertices, so the color stays discontinuous across layer interfaces even
    though consecutive rows coincide; the zero-height bands between two
    layers are invisible.
    """
    if "shading" in kwargs:
        warnings.warn(
            "`shading` does not apply to vertical cross-sections; use `centering` instead."
        )
        kwargs.pop("shading")

    x = grid.x_of_node[grid.core_nodes].reshape(grid.cell_grid_shape)
    y = grid.y_of_node[grid.core_nodes].reshape(grid.cell_grid_shape)
    values, dz = _get_layer_field(grid, var, i_class, mask_value)

    if i_x is None:
        i_x = slice(None)
        if i_y == "middle":
            i_y = grid.cell_grid_shape[0] // 2
        section = x[i_y, i_x]
    elif i_y is None:
        i_y = slice(None)
        if i_x == "middle":
            i_x = grid.cell_grid_shape[1] // 2
        section = y[i_y, i_x]

    vmin = (
        np.nanmin(values[..., i_y, i_x])
        if vmin is None and "norm" not in kwargs
        else vmin
    )
    vmax = (
        np.nanmax(values[..., i_y, i_x])
        if vmax is None and "norm" not in kwargs
        else vmax
    )

    layer_values = np.ascontiguousarray(values[:, i_y, i_x])
    layer_dz = dz[:, i_y, i_x]
    if len(layer_values) == 0:
        warnings.warn("The slice contains no layer to plot.")
        return None
    for i in range(len(layer_values)):
        mask_layer(layer_values[i], layer_dz[i], mask_wedges, mask_null_layers)

    interfaces = _get_layer_elevation(grid).reshape(-1, *grid.cell_grid_shape)[
        :, i_y, i_x
    ]
    n_layers, n_columns = layer_values.shape
    if centering == "cell":
        # Match the 'nearest' shading of the horizontal slices: each tie's
        # value fills, at constant color, the part of the section closer to
        # its column than to any other tie, so the cell boundaries sit at the
        # midpoints between ties
        boundaries = 0.5 * (section[:-1] + section[1:])
        columns = np.unique(np.concatenate([section, boundaries]))
        interfaces = np.stack([np.interp(columns, section, ifc) for ifc in interfaces])
        x_rows = np.tile(columns, (2 * n_layers, 1))
        z_rows = np.repeat(interfaces, 2, axis=0)[1:-1]
        nearest = np.searchsorted(boundaries, 0.5 * (columns[:-1] + columns[1:]))
        colors = np.full((2 * n_layers - 1, len(columns) - 1), np.nan)
        colors[::2] = layer_values[:, nearest]
        return ax.pcolormesh(
            x_rows, z_rows, colors, vmin=vmin, vmax=vmax, shading="flat", **kwargs
        )
    x_rows = np.tile(section, (2 * n_layers, 1))
    # Duplicating each internal interface gives each layer its own two rows;
    # the bands between two layers have a null height and stay invisible
    z_rows = np.repeat(interfaces, 2, axis=0)[1:-1]
    if centering == "node":
        values_bottom, values_top = _get_interface_values(layer_values)
    else:
        values_bottom, values_top = layer_values, layer_values
    colors = np.empty((2 * n_layers, n_columns))
    colors[0::2] = values_bottom
    colors[1::2] = values_top
    return ax.pcolormesh(
        x_rows, z_rows, colors, vmin=vmin, vmax=vmax, shading="gouraud", **kwargs
    )


def plot_layers(
    ax,
    grid,
    var,
    i_x="middle",
    i_y=None,
    i_layer=None,
    i_class=None,
    centering="tie",
    mask_wedges=False,
    mask_null_layers=True,
    mask_value=None,
    vmin=None,
    vmax=None,
    **kwargs,
):
    """Plot a horizontal or vertical slice through a stack of layers on a raster
    grid.

    Visualization follows the tie-centered approach described by Tetzlaff (2023),
    with the option to switch to a cell- or node-centered scheme; layer erosion
    is not properly handled in the vertical cross-sections (see figure 12 of
    Tetzlaff (2023)).

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
        proportion of each lithology; the keywords 'most_frequent_class' or
        'most_frequent' can be to plot the most frequent lithology.
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
    i_layer : int, float, str, or callable, optional
        The index at which to slice horizontally the layers, i.e., to plot a
        given layer in map view. The keyword 'middle' will slice at the middle
        of the axis; the keyword 'top' will show the values at the topographic
        surface. A callable will merge the all the slices along the axis together
        (e.g., numpy.mean or numpy.sum). A float is only supported together with
        the 'composition' variable, where it is read as a thickness from the
        surface: the map then shows the superficial composition of that top
        interval, accumulated across layers (see
        `StackedLayers.get_superficial_composition`), with `i_class` selecting the
        lithology. A slice can only be made along a single axis, so `i_layer` is
        incompatible with `i_x` and `i_y`.
    i_class : int or callable, optional
        The index to select the lithology to plot. A callable will merge the all
        the lithologies together (e.g., numpy.mean or numpy.sum). Using
        'weighted_mean' will average the lithologies weighted by their respective
        thicknesses.
    centering : str, optional
        The scheme used to color the layers:
            - 'tie' (default) assigns the values to vertical lines at the grid
              nodes, so the color is interpolated horizontally but stays constant
              vertically within a layer.
            - 'cell' fills the cell around each tie with the tie's constant
              value, the cell boundaries sitting at the midpoints between ties.
            - 'node' assigns interface-averaged values to the layer boundaries,
              so the color is also interpolated vertically across interfaces.
        In map view a tie is seen end-on, so 'tie' and 'node' coincide for horizontal
        slices: the value sits at the node and is interpolated in between, while 'cell'
        fills the cell around each node with a constant value.
    mask_wedges : bool, optional
        If True, wedges, where a layer pinches out to a null thickness, are
        displayed with the value of the non-null node. This is useful when
        displaying the composition of a given class for instance, and avoid
        perturbing the visualization with a pointless null composition.
    mask_null_layers : bool, optional
        If True, nodes of null thickness that are surrounded with null-thickness
        nodes are turned to NaN. This avoid perturbing the display of the
        stratigraphy, where null-thickness layers can still be visible.
    mask_value : float, optional
        A value of the variable being plotted to mask, i.e., to turn to NaN.
    vmin : float, optional
        The minimum value to use when plotting the variable. If None, uses the
        minimum value of the variable.
    vmax : float, optional
        The maximum value to use when plotting the variable. If None, uses the
        maximum value of the variable.
    **kwargs : `matplotlib.axes.Axes.pcolormesh` properties
        Other keyword arguments are passed to `pcolormesh`.

    Returns
    -------
    c : matplotlib.collections.Collection or None
        The mesh that constitutes the plot, or None when there is no layer to plot.

    Reference
    ---------
    Tetzlaff, D. (2023)
        Stratigraphic forward modeling software package for research and education
        https://arxiv.org/abs/2302.05272
    """
    # TODO: One reason `mask_wedges` is required is that sediment composition in
    # the stratigraphy is tracked as thicknesses, so that complete erosion of a
    # layer lead to an null composition. But not sure what is the best way
    # forward here
    if centering not in ("tie", "cell", "node"):
        raise ValueError(
            f"`centering` must be 'tie', 'cell', or 'node', not {centering!r}."
        )
    if i_x is not None and i_y is not None:
        warnings.warn(
            "Both `i_x` and `i_y` are not None, but a slice can only be made along a single axis. `i_y` will be turned to None."
        )
        i_y = None
    if i_x is not None and i_layer is not None:
        warnings.warn(
            "Both `i_x` and `i_layer` are not None, but a slice can only be made along a single axis. `i_layer` will be turned to None."
        )
        i_layer = None
    if i_y is not None and i_layer is not None:
        warnings.warn(
            "Both `i_y` and `i_layer` are not None, but a slice can only be made along a single axis. `i_layer` will be turned to None."
        )
        i_layer = None

    if i_x is not None or i_y is not None:
        return _plot_vertical_slice(
            ax,
            grid,
            var,
            i_x,
            i_y,
            i_class,
            centering,
            mask_wedges,
            mask_null_layers,
            mask_value,
            vmin,
            vmax,
            **kwargs,
        )
    elif i_layer is not None:
        return _plot_horizontal_slice(
            ax, grid, var, i_layer, i_class, centering, mask_value, vmin, vmax, **kwargs
        )
    else:
        raise Exception("At least one of `i_x`, `i_y`, or `i_layer` must not be None.")


class RasterModelGridLayerPlotterMixIn:
    """MixIn that provides layer plotting functionality to a raster grid.

    Inherit from this class to provide a ModelDataFields object with the method
    function, ``plot_layers``, that plots a slice through a StackedLayers.
    """

    def plot_layers(
        self,
        ax,
        var,
        i_x="middle",
        i_y=None,
        i_layer=None,
        i_class=None,
        centering="tie",
        mask_wedges=False,
        mask_null_layers=True,
        mask_value=None,
        vmin=None,
        vmax=None,
        **kwargs,
    ):
        """Plot a horizontal or vertical slice through a StackedLayers. This is
        a wrapper for `plot.plot_layers`, and its uses the same keywords.

        Visualization follows the tie-centered approach described by Tetzlaff (2023),
        with the option to switch to a cell- or node-centered scheme; layer erosion
        is not properly handled in the vertical cross-sections (see figure 12 of
        Tetzlaff (2023)).

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
            proportion of each lithology; the keywords 'most_frequent_class' or
            'most_frequent' can be to plot the most frequent lithology.
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
        i_layer : int, float, str, or callable, optional
            The index at which to slice horizontally the layers, i.e., to plot a
            given layer in map view. The keyword 'middle' will slice at the middle
            of the axis; the keyword 'top' will show the values at the topographic
            surface. A callable will merge the all the slices along the axis together
            (e.g., numpy.mean or numpy.sum). A float is only supported together with
            the 'composition' variable, where it is read as a thickness from the
            surface: the map then shows the superficial composition of that top
            interval, accumulated across layers (see
            `StackedLayers.get_superficial_composition`), with `i_class` selecting the
            lithology. A slice can only be made along a single axis, so `i_layer` is
            incompatible with `i_x` and `i_y`.
        i_class : int or callable, optional
            The index to select the lithology to plot. A callable will merge the all
            the lithologies together (e.g., numpy.mean or numpy.sum). Using
            'weighted_mean' will average the lithologies weighted by their respective
            thicknesses.
        centering : str, optional
            The scheme used to color the layers:
                - 'tie' (default) assigns the values to vertical lines at the grid nodes,
                  so the color is interpolated horizontally but stays constant vertically
                  within a layer.
                - 'cell' fills the cell around each tie with the tie's constant value,
                  the cell boundaries sitting at the midpoints between ties.
                - 'node' assigns interface-averaged values to the layer boundaries,
                  so the color is also interpolated vertically across interfaces.
            In map view a tie is seen end-on, so 'tie' and 'node' coincide for horizontal
            slices: the value sits at the node and is interpolated in between, while 'cell'
            fills the cell around each node with a constant value.
        mask_wedges : bool, optional
            If True, wedges, where a layer pinches out to a null thickness, are
            displayed with the value of the non-null node. This is useful when
            displaying the composition of a given class for instance, and avoid
            perturbing the visualization with a pointless null composition.
        mask_null_layers : bool, optional
            If True, nodes of null thickness that are surrounded with null-thickness
            nodes are turned to NaN. This avoid perturbing the display of the
            stratigraphy, where null-thickness layers can still be visible.
        mask_value : float, optional
            A value of the variable being plotted to mask, i.e., to turn to NaN.
        vmin : float, optional
            The minimum value to use when plotting the variable. If None, uses the
            minimum value of the variable.
        vmax : float, optional
            The maximum value to use when plotting the variable. If None, uses the
            maximum value of the variable.
        **kwargs : `matplotlib.axes.Axes.pcolormesh` properties
            Other keyword arguments are passed to `pcolormesh`.

        Returns
        -------
        c : matplotlib.collections.Collection or None
            The mesh that constitutes the plot, or None when there is no layer to plot.

        Reference
        ---------
        Tetzlaff, D. (2023)
            Stratigraphic forward modeling software package for research and education
            https://arxiv.org/abs/2302.05272

        See Also
        --------
        stratigrapy.plot.plot_layers
        """
        return plot_layers(
            ax,
            self,
            var,
            i_x,
            i_y,
            i_layer,
            i_class,
            centering,
            mask_wedges,
            mask_null_layers,
            mask_value,
            vmin,
            vmax,
            **kwargs,
        )


################################################################################
# 3D plotting


def _neighbor_stack(field, fill):
    """The 8-neighbor values of every tie of a plan grid, stacked on a new
    first axis in the order of `_PLAN_NEIGHBORS`, with `fill` outside the
    grid.
    """
    ny, nx = field.shape
    padded = np.pad(field, 1, constant_values=fill)

    return np.stack(
        [
            padded[1 + di : 1 + di + ny, 1 + dj : 1 + dj + nx]
            for di, dj in [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ]
        ]
    )


def _pick_nearest(candidates, eligible, weight, distance):
    """Selects, at every tie of a plan grid, the eligible candidate at the
    smallest physical distance, the largest `weight` (the layer thickness)
    breaking the ties deterministically. The candidates are stacked on the
    first axis, with one distance each.
    """
    distance = distance.reshape((-1,) + (1,) * (candidates.ndim - 1))
    score = np.where(eligible, distance, np.inf)
    nearest = np.min(score, axis=0)
    priority = np.where(score == nearest, weight, -1.0)
    pick = np.expand_dims(np.argmax(priority, axis=0), 0)
    picked = np.take_along_axis(candidates, pick, axis=0)[0]
    found = np.isfinite(nearest)

    return np.where(found, picked, np.nan), found


def _nearest_plan_neighbors(values, eligible, dz, spacing):
    """Value of the nearest eligible 8-neighbor at every tie of a plan grid:
    the 4-neighbors come before the diagonals by physical distance, and the
    thickest tie wins at equal distance.
    """
    dy, dx = spacing
    distance = np.array([dy, dy, dx, dx] + [np.hypot(dy, dx)] * 4)

    return _pick_nearest(
        _neighbor_stack(values, np.nan),
        _neighbor_stack(eligible, False),
        _neighbor_stack(dz, 0.0),
        distance,
    )


def _mask_plan_layers(values, dz, spacing, mask_wedges, mask_null_layers, fill_nan):
    """Applies the wedge and null-layer masking over the whole plan grid at
    once, the two-dimensional counterpart of the line-by-line `mask_layer` of
    the 2D sections. The rim of a wedge is any dead tie (null thickness)
    sharing a plan cell with a living tie, whichever the direction the layer
    pinches out along: `mask_wedges` recolors the rim with the value of its
    nearest living 8-neighbor, so the flanks of a wedge show the layer's own
    material all around it. A dead tie surrounded by dead ties only bounds
    no visible cell: `mask_null_layers` turns it to NaN. `fill_nan` then
    fills the remaining NaN of each layer with the values of the layer
    below.
    """
    from scipy.ndimage import binary_dilation

    dz = np.asarray(dz, float)
    dead = dz == 0.0
    rim = dead & binary_dilation(~dead, np.ones((1, 3, 3), bool))
    if mask_wedges == True:
        for k in np.flatnonzero(rim.any(axis=(1, 2))):
            picked = _nearest_plan_neighbors(values[k], ~dead[k], dz[k], spacing)[0]
            values[k][rim[k]] = picked[rim[k]]
    if mask_null_layers == True:
        values[dead & ~rim] = np.nan
    if fill_nan == True:
        for k in range(1, len(values)):
            missing = np.isnan(values[k])
            values[k][missing] = values[k - 1][missing]


def _interleave_midpoints(field):
    """Inserts the midpoint between consecutive entries along the last axis,
    by linear interpolation.
    """
    out = np.empty(field.shape[:-1] + (2 * field.shape[-1] - 1,))
    out[..., 0::2] = field
    out[..., 1::2] = 0.5 * (field[..., :-1] + field[..., 1:])
    return out


def _subdivide_plan_grid(field, shape):
    """Inserts the midpoints between the plan-grid nodes along both
    horizontal directions, with the columns flattened on the last axis of
    `field`. The surfaces are piecewise bilinear between the ties, so
    sampling them at the midpoints leaves the geometry unchanged; the
    midpoints become the cell boundaries of the cell-centered display, as in
    the 2D sections.
    """
    ny, nx = shape
    field = np.asarray(field, float).reshape(field.shape[:-1] + (ny, nx))
    field = _interleave_midpoints(field)
    field = np.swapaxes(_interleave_midpoints(np.swapaxes(field, -1, -2)), -1, -2)

    return field.reshape(field.shape[:-2] + (-1,))


def _nearest_tie_cells(values, dz, spacing, shape, mask_wedges):
    """Cell values on the midpoint-subdivided plan grid: each quadrant takes
    the value of its corner tie, the nearest tie to every point of the
    quadrant, except around the dead ties (null thickness), where a quadrant
    takes the value of the nearest living corner of its own plan cell — the
    edge-adjacent corners before the diagonal by physical distance, the
    thickest at equal distance — so each flank of a wedge shows the material
    it actually abuts, whichever the direction the layer pinches out along.
    Without `mask_wedges` the recorded values of the dead ties stay and only
    the missing ones take the nearest recorded corner. A dead tie in a fully
    dead plan cell keeps the tie fill of `_fill_dead_plan_ties`.
    """
    ny, nx = shape
    dz = np.asarray(dz, float)
    dead = dz == 0.0
    missing = ~np.isfinite(values) & dead
    filled = _fill_dead_plan_ties(values, dz, spacing)

    # Quadrant (I, J) has corner tie ((I + 1) // 2, (J + 1) // 2) and lies
    # in the plan cell (I // 2, J // 2), whose other row and column give the
    # three remaining corners
    i_tie = (np.arange(2 * (ny - 1)) + 1) // 2
    j_tie = (np.arange(2 * (nx - 1)) + 1) // 2
    i_other = 2 * (np.arange(2 * (ny - 1)) // 2) + 1 - i_tie
    j_other = 2 * (np.arange(2 * (nx - 1)) // 2) + 1 - j_tie
    ii, jj = i_tie[:, np.newaxis], j_tie
    oi, oj = i_other[:, np.newaxis], j_other

    cells = filled[:, ii, jj]
    candidates = np.stack([values[:, oi, jj], values[:, ii, oj], values[:, oi, oj]])
    weight = np.stack([dz[:, oi, jj], dz[:, ii, oj], dz[:, oi, oj]])
    if mask_wedges == True:
        eligible = weight > 0.0
        recolor = dead[:, ii, jj]
    else:
        eligible = np.isfinite(candidates)
        recolor = missing[:, ii, jj]
    dy, dx = spacing
    picked, found = _pick_nearest(
        candidates, eligible, weight, np.array([dy, dx, np.hypot(dy, dx)])
    )

    return np.where(recolor & found, picked, cells)


def _fill_dead_plan_ties(values, dz, spacing):
    """Plan-grid analog of `_fill_dead_ties`: replaces the missing values on
    the dead ties (null thickness) of each layer by the value of the nearest
    recorded tie.

    TODO: This suffers from the same limitations as the wedge masking in 2D,
    made worse by adding an extra dimension.
    """
    from scipy.ndimage import distance_transform_edt

    dz = np.asarray(dz, float)
    missing = ~np.isfinite(values) & (dz == 0.0)
    if not missing.any():
        return values
    values = values.copy()
    for k in np.flatnonzero(missing.any(axis=(1, 2))):
        recorded = np.isfinite(values[k])
        if not recorded.any():
            continue
        picked, found = _nearest_plan_neighbors(values[k], recorded, dz[k], spacing)
        fill = missing[k] & found
        values[k][fill] = picked[fill]
        left = missing[k] & ~found
        if left.any():
            nearest = distance_transform_edt(
                ~np.isfinite(values[k]), sampling=spacing, return_indices=True
            )[1]
            values[k][left] = values[k][nearest[0][left], nearest[1][left]]

    return values


def extract_layer_mesh(
    grid,
    var,
    i_class=None,
    centering="tie",
    mask_wedges=False,
    mask_null_layers=False,
    mask_value=None,
    fill_nan=False,
):
    """Extract the whole stratigraphy of a stack of layers on a raster grid as a
    mesh for 3D plotting, in particular with PyVista.

    Visualization follows the tie-centered approach described by Tetzlaff (2023),
    with the option to switch to a cell- or node-centered scheme; layer erosion
    is not properly handled in the vertical cross-sections (see figure 12 of
    Tetzlaff (2023)).

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
        the lithologies together (e.g., numpy.mean or numpy.sum). Using
        'weighted_mean' will average the lithologies weighted by their respective
        thicknesses.
    centering : str, optional
        The scheme used to color the layers:
             - 'tie' (default) assigns the values to vertical lines at the grid
               nodes, so the color is interpolated horizontally but stays constant
               vertically within a layer.
             - 'cell' fills the cells around each tie with the tie's constant
               value, returned as cell data instead of point data; the plan grid
               is subdivided once so the cell boundaries sit at the midpoints
               between ties, as in the horizontal slices, which quadruples the
               number of cells.
             - 'node' assigns interface-averaged values to the layer boundaries,
               so the color also grades vertically across interfaces.
    mask_wedges : bool, optional
        If True, the rim of the wedges, i.e., the ties of null thickness that
        share a plan cell with a tie where the layer lives, take the value of
        their nearest living neighbor, whichever the plan direction the layer
        pinches out along. With 'cell' centering the recoloring is done per
        quadrant: each quadrant of a rim tie takes the nearest living corner
        of its own plan cell, so each flank of a wedge shows the material it
        actually abuts even when the layer carries different values on each
        side of the pinch-out. This is useful when displaying the composition
        of a given class for instance, and avoids perturbing the
        visualization with a pointless null composition.
    mask_null_layers : bool, optional
        If True, ties of null thickness whose plan neighbors all have a null
        thickness too are turned to NaN. This avoid perturbing the display of
        the stratigraphy, where null-thickness layers can still be visible.
        Wherever the mesh keeps visible geometry against such a tie, the value
        of the nearest living tie of the layer shows instead, so the masking
        never leaves missing colors on visible cells.
    mask_value : float, optional
        A value of the variable being plotted to mask, i.e., to turn to NaN.
    fill_nan: bool
        If True, fills the NaN values of a layer with the values of the layer
        below.

    Returns
    -------
    x, y, z : ndarray of float, shape (nx, ny, nz)
        The coordinates of the points of the structured grid, ready for
        `pyvista.StructuredGrid(x, y, z)`. `nz` is ``2 * n_layers`` for
        'tie' centering (each layer owns its bottom and top sheet, so the
        color stays discontinuous across the interfaces) and ``n_layers + 1``
        otherwise; 'cell' centering subdivides the plan grid at the midpoints
        between ties.
    point_data : dict of ndarray, shape (n_points,)
        The arrays defined on the points, in VTK point order: 'value', the
        values of the variable `var` (unless ``centering='cell'``).
    cell_data : dict of ndarray, shape (n_cells,)
        The arrays defined on the cells, in VTK cell order: 'value', the
        values of the variable `var` (only when ``centering='cell'``).

    Reference
    ---------
    Tetzlaff, D. (2023)
        Stratigraphic forward modeling software package for research and education
        https://arxiv.org/abs/2302.05272

    Examples
    --------
    The structured grid is built and displayed with PyVista as::

        import pyvista as pv

        x, y, z, point_data, cell_data = extract_layer_mesh(
            grid, 'composition', i_class=1
        )
        mesh = pv.StructuredGrid(x, y, z)
        mesh.point_data.update(point_data)
        mesh.cell_data.update(cell_data)
        mesh.plot(scalars='value')
    """
    if centering not in ("tie", "cell", "node"):
        raise ValueError(
            f"`centering` must be 'tie', 'cell', or 'node', not {centering!r}."
        )

    shape = grid.cell_grid_shape
    values, dz = _get_layer_field(grid, var, i_class, mask_value)
    n_layers = len(values)
    if n_layers == 0:
        warnings.warn("The stack contains no layer to mesh.")
        return None
    _mask_plan_layers(
        values, dz, (grid.dy, grid.dx), mask_wedges, mask_null_layers, fill_nan
    )

    ny, nx = shape
    x = grid.x_of_node[grid.core_nodes]
    y = grid.y_of_node[grid.core_nodes]
    interfaces = _get_layer_elevation(grid)

    spacing = (grid.dy, grid.dx)

    point_data, cell_data = {}, {}
    if centering == "cell":
        cell_data["value"] = _nearest_tie_cells(
            values, dz, spacing, shape, mask_wedges
        ).reshape(-1)
        x = _subdivide_plan_grid(x, shape)
        y = _subdivide_plan_grid(y, shape)
        z = _subdivide_plan_grid(interfaces, shape)
        ny, nx = 2 * ny - 1, 2 * nx - 1
    elif centering == "node":
        z = interfaces
        values_bottom, values_top = _get_interface_values(values.reshape(n_layers, -1))
        values_bottom = _fill_dead_plan_ties(
            values_bottom.reshape(n_layers, ny, nx), dz, spacing
        ).reshape(n_layers, -1)
        values_top = _fill_dead_plan_ties(
            values_top.reshape(n_layers, ny, nx), dz, spacing
        ).reshape(n_layers, -1)
        point_data["value"] = np.concatenate([values_bottom[:1], values_top]).reshape(
            -1
        )
    else:
        # Duplicating each internal interface gives each layer its own two
        # sheets of points, so the color stays discontinuous across the
        # interfaces; the cells between two layers have a null height and
        # stay invisible
        z = np.repeat(interfaces, 2, axis=0)[1:-1]
        point_data["value"] = np.repeat(
            _fill_dead_plan_ties(values, dz, spacing).reshape(n_layers, -1),
            2,
            axis=0,
        ).reshape(-1)

    # The natural (nz, ny, nx) arrays are transposed to the (nx, ny, nz)
    # expected by pyvista.StructuredGrid, whose Fortran-ordered points then
    # match the C-ravelled data arrays
    z = z.reshape(len(z), ny, nx)
    x = np.broadcast_to(x.reshape(ny, nx), z.shape)
    y = np.broadcast_to(y.reshape(ny, nx), z.shape)

    return x.T, y.T, z.T, point_data, cell_data
