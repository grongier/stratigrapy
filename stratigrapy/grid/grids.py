"""Grids with stacked layers"""

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


import importlib
import os
from landlab.core.utils import get_functions_from_module, add_functions_to_class
from landlab import RasterModelGrid as _RasterModelGrid
from landlab import VoronoiDelaunayGrid as _VoronoiDelaunayGrid
from landlab import FramedVoronoiGrid as _FramedVoronoiGrid
from landlab import HexModelGrid as _HexModelGrid

from ..layers import StackedLayersMixIn
from ..plot.layers import RasterModelGridLayerPlotterMixIn


################################################################################
# Adding member function (from Landlab)


def add_module_functions_to_class(cls, module, pattern=None, exclude=None):
    """Add functions from a module to a class as methods.

    Parameters
    ----------
    cls : class
        A class.
    module : module
        An instance of a module.
    pattern : str, optional
        Only get functions whose name match a regular expression.
    exclude : str, optional
        Only get functions whose name exclude the regular expression.

    *Note* if both pattern and exclude are provided both conditions must be met.
    """
    (module, _) = os.path.splitext(os.path.basename(module))

    mod = importlib.import_module("." + module, package="stratigrapy.grid")

    funcs = get_functions_from_module(mod, pattern=pattern, exclude=exclude)
    add_functions_to_class(cls, funcs)


################################################################################
# Raster


class RasterModelGrid(
    StackedLayersMixIn, RasterModelGridLayerPlotterMixIn, _RasterModelGrid
):
    """A 2D uniform rectilinear grid."""

    def __init__(
        self,
        shape,
        xy_spacing=1.0,
        xy_of_lower_left=(0.0, 0.0),
        xy_of_reference=(0.0, 0.0),
        xy_axis_name=("x", "y"),
        xy_axis_units="-",
        bc=None,
        number_of_classes=1,
        initial_allocation=1,
        new_allocation=1,
        number_of_layers_to_fuse=1,
        number_of_top_layers=1,
        fuse_continuously=True,
        remove_empty_layers=False,
    ):
        """Create a 2D grid with equal spacing.

        Optionally takes numbers of rows and columns and cell size as
        inputs. If this are given, calls initialize() to set up the grid.
        At the moment, num_rows and num_cols MUST be specified. Both must be
        >=3 to allow correct automated setup of boundary conditions.

        Parameters
        ----------
        shape : tuple of int
            Shape of the grid in nodes as (nrows, ncols).
        xy_spacing : tuple or float, optional
            dx and dy spacing. Either provided as a float or a
            (dx, dy) tuple.
        xy_of_lower_left: tuple, optional
            (x, y) coordinates of the lower left corner.
        xy_of_reference : tuple, optional
            Coordinate value in projected space of the reference point,
            `xy_of_lower_left`. Default is (0., 0.)
        xy_axis_name: tuple of str
            Name to use for each axis.
        xy_axis_units: tuple of str, or str
            Units for coordinates of each axis.
        bc : dict, optional
            Edge boundary conditions.
        number_of_classes : int, optional
            Number of classes of material (e.g., different lithologies or grain
            sizes) making the layers of the StackedLayers.
        initial_allocation : int or array-like, optional
            Number of layers initially pre-allocated to speed up adding new layers.
            If an int, the layers are pre-allocated at the end; it an array-like,
            the first element corresponds to layers pre-allocated at the start,
            the second to layers pre-allocated at the end.
        new_allocation : int or array-like, optional
            Number of layers pre-allocated when adding new layers to speed up the
            process. If an array-like, different number of layers are pre-allocated
            at the start and at the end of the stacks.
        number_of_layers_to_fuse : int, optional
            Number of layers to merge together when calling `fuse`.
        number_of_top_layers : int, optional
            Number of layers at the top of the stack to ignore when calling `fuse`.
        fuse_continuously : bool, optional
            If True, fuse the layers as they are added; otherwise, wait until the
            number of unfused layers reaches `number_of_layers_to_fuse` (excluding
            the top layers) to fuse them.
        remove_empty_layers : bool, optional
            If True, remove the layers without any deposits at the top of the stack
            when adding a layer; otherwise, keep all the layers.

        Returns
        -------
        RasterModelGrid
            A newly-created grid.
        """
        _RasterModelGrid.__init__(
            self,
            shape,
            xy_spacing,
            xy_of_lower_left,
            xy_of_reference,
            xy_axis_name,
            xy_axis_units,
            bc,
        )
        StackedLayersMixIn.__init__(
            self,
            number_of_classes,
            initial_allocation,
            new_allocation,
            number_of_layers_to_fuse,
            number_of_top_layers,
            fuse_continuously,
            remove_empty_layers,
        )


add_module_functions_to_class(RasterModelGrid, "raster_divergence.py", pattern="calc_*")


################################################################################
# Voronoi


class VoronoiDelaunayGrid(StackedLayersMixIn, _VoronoiDelaunayGrid):
    """This inherited class implements an unstructured grid in which cells are
    Voronoi polygons and nodes are connected by a Delaunay triangulation. Uses
    scipy.spatial module to build the triangulation.

    Create an unstructured grid from points whose coordinates are given
    by the arrays *x*, *y*.
    """

    def __init__(
        self,
        x=None,
        y=None,
        reorient_links=True,
        xy_of_reference=(0.0, 0.0),
        xy_axis_name=("x", "y"),
        xy_axis_units="-",
        number_of_classes=1,
        initial_allocation=1,
        new_allocation=1,
        number_of_layers_to_fuse=1,
        number_of_top_layers=1,
        fuse_continuously=True,
        remove_empty_layers=False,
    ):
        """Create a Voronoi Delaunay grid from a set of points.

        Create an unstructured grid from points whose coordinates are given
        by the arrays *x*, *y*.

        Parameters
        ----------
        x : array_like
            x-coordinate of points
        y : array_like
            y-coordinate of points
        reorient_links (optional) : bool
            whether to point all links to the upper-right quadrant
        xy_of_reference : tuple, optional
            Coordinate value in projected space of (0., 0.)
            Default is (0., 0.)
        number_of_classes : int, optional
            Number of classes of material (e.g., different lithologies or grain
            sizes) making the layers of the StackedLayers.
        initial_allocation : int or array-like, optional
            Number of layers initially pre-allocated to speed up adding new layers.
            If an int, the layers are pre-allocated at the end; it an array-like,
            the first element corresponds to layers pre-allocated at the start,
            the second to layers pre-allocated at the end.
        new_allocation : int or array-like, optional
            Number of layers pre-allocated when adding new layers to speed up the
            process. If an array-like, different number of layers are pre-allocated
            at the start and at the end of the stacks.
        number_of_layers_to_fuse : int, optional
            Number of layers to merge together when calling `fuse`.
        number_of_top_layers : int, optional
            Number of layers at the top of the stack to ignore when calling `fuse`.
        fuse_continuously : bool, optional
            If True, fuse the layers as they are added; otherwise, wait until the
            number of unfused layers reaches `number_of_layers_to_fuse` (excluding
            the top layers) to fuse them.
        remove_empty_layers : bool, optional
            If True, remove the layers without any deposits at the top of the stack
            when adding a layer; otherwise, keep all the layers.

        Returns
        -------
        VoronoiDelaunayGrid
            A newly-created grid.
        """
        _VoronoiDelaunayGrid.__init__(
            self, x, y, reorient_links, xy_of_reference, xy_axis_name, xy_axis_units
        )
        StackedLayersMixIn.__init__(
            self,
            number_of_classes,
            initial_allocation,
            new_allocation,
            number_of_layers_to_fuse,
            number_of_top_layers,
            fuse_continuously,
            remove_empty_layers,
        )


add_module_functions_to_class(VoronoiDelaunayGrid, "divergence.py", pattern="calc_*")


class FramedVoronoiGrid(StackedLayersMixIn, _FramedVoronoiGrid):
    """A grid of Voronoi Delaunay cells with a structured perimeter layout.

    This inherited class implements a irregular 2D grid with Voronoi Delaunay cells and
    irregular patches. It is a special type of :class:`~.VoronoiDelaunayGrid` grid in which
    the initial set of points is arranged in a fixed lattice (e.g. like a
    :class:`~.RasterModelGrid`), named here "layout", and the core points are
    then moved a random distance from their initial positions, bounded by a user-supplied
    threshold.
    """

    def __init__(
        self,
        shape,
        xy_spacing=(1.0, 1.0),
        xy_of_lower_left=(0.0, 0.0),
        xy_min_spacing=(0.5, 0.5),
        seed=200,
        xy_of_reference=(0.0, 0.0),
        xy_axis_name=("x", "y"),
        xy_axis_units="-",
        number_of_classes=1,
        initial_allocation=1,
        new_allocation=1,
        number_of_layers_to_fuse=1,
        number_of_top_layers=1,
        fuse_continuously=True,
        remove_empty_layers=False,
    ):
        """Create a grid of voronoi cells with a structured perimeter.

        Create an irregular 2D grid with voronoi cells and triangular patches.
        It is a special type of :class:`~.VoronoiDelaunayGrid` in which the initial set
        of points is arranged in a regular lattice determined by the parameters
        *shape*, and *xy_spacing*. The coordinates of
        the core points are then randomly moved while the perimeter points
        remaining fixed, in a way determined by the parameters *xy_min_spacing*, and
        *seed*.

        Parameters
        ----------
        shape : tuple of int
            Number of rows and columns of nodes.
        xy_spacing : float or tuple of float, optional
            Node spacing along x and y coordinates. If ``float``, same spacing at *x* and *y*.
        xy_of_lower_left : tuple, optional
            Minimum *x*-of-node and *y*-of-node values. Depending on the grid,
            there may not be a node at this coordinate.
        xy_min_spacing: float or tuple of float, optional
            Final minimal spacing between nodes. Random moves of the core nodes
            from their initial positions cannot be above this threshold:
            ``(xy_spacing - xy_min_spacing) / 2``
            If ``float``, same minimal spacing for *x* and *y*.
        seed: int, optional
            Seed used to generate the random *x* and *y* moves.
            When set, controls the pseudo-randomness of moves to ensure
            reproducibility.
            When ``None``, the seed is random and the moves of coordinates are
            completely random.
        xy_of_reference : tuple, optional
            Coordinate value in projected space of the reference point,
            *xy_of_lower_left*.
        xy_axis_name: tuple of str, optional
            *x* and *y* axis names.
        xy_axis_units: str, optional
            *x* and *y* axis units.
        number_of_classes : int, optional
            Number of classes of material (e.g., different lithologies or grain
            sizes) making the layers of the StackedLayers.
        initial_allocation : int or array-like, optional
            Number of layers initially pre-allocated to speed up adding new layers.
            If an int, the layers are pre-allocated at the end; it an array-like,
            the first element corresponds to layers pre-allocated at the start,
            the second to layers pre-allocated at the end.
        new_allocation : int or array-like, optional
            Number of layers pre-allocated when adding new layers to speed up the
            process. If an array-like, different number of layers are pre-allocated
            at the start and at the end of the stacks.
        number_of_layers_to_fuse : int, optional
            Number of layers to merge together when calling `fuse`.
        number_of_top_layers : int, optional
            Number of layers at the top of the stack to ignore when calling `fuse`.
        fuse_continuously : bool, optional
            If True, fuse the layers as they are added; otherwise, wait until the
            number of unfused layers reaches `number_of_layers_to_fuse` (excluding
            the top layers) to fuse them.
        remove_empty_layers : bool, optional
            If True, remove the layers without any deposits at the top of the stack
            when adding a layer; otherwise, keep all the layers.

        Returns
        -------
        FramedVoronoiGrid
            A newly-created grid.
        """
        _FramedVoronoiGrid.__init__(
            self,
            shape,
            xy_spacing,
            xy_of_lower_left,
            xy_min_spacing,
            seed,
            xy_of_reference,
            xy_axis_name,
            xy_axis_units,
        )
        StackedLayersMixIn.__init__(
            self,
            number_of_classes,
            initial_allocation,
            new_allocation,
            number_of_layers_to_fuse,
            number_of_top_layers,
            fuse_continuously,
            remove_empty_layers,
        )


add_module_functions_to_class(FramedVoronoiGrid, "divergence.py", pattern="calc_*")


################################################################################
# Hex


class HexModelGrid(StackedLayersMixIn, _HexModelGrid):
    """A grid of hexagonal cells.

    This inherited class implements a regular 2D grid with hexagonal cells and
    triangular patches. It is a special type of VoronoiDelaunay grid in which
    the initial set of points is arranged in a triangular/hexagonal lattice.
    """

    def __init__(
        self,
        shape,
        spacing=1.0,
        xy_of_lower_left=(0.0, 0.0),
        orientation="horizontal",
        node_layout="hex",
        reorient_links=True,
        xy_of_reference=(0.0, 0.0),
        xy_axis_name=("x", "y"),
        xy_axis_units="-",
        number_of_classes=1,
        initial_allocation=1,
        new_allocation=1,
        number_of_layers_to_fuse=1,
        number_of_top_layers=1,
        fuse_continuously=True,
        remove_empty_layers=False,
    ):
        """Create a grid of hexagonal cells.

        Create a regular 2D grid with hexagonal cells and triangular patches.
        It is a special type of :class:`~.VoronoiModelGrid` in which the initial set
        of points is arranged in a triangular/hexagonal lattice.

        Parameters
        ----------
        shape : tuple of int
            Number of rows and columns of nodes.
        spacing : float, optional
            Node spacing.
        xy_of_lower_left : tuple of float, optional
            Minimum x-of-node and y-of-node values. Depending on the grid
            no node may be present at this coordinate. Default is (0., 0.).
        xy_of_reference : tuple of float, optional
            Coordinate value in projected space of the reference point,
            `xy_of_lower_left`. Default is (0., 0.)
        orientation : str, optional
            One of the 3 cardinal directions in the grid, either 'horizontal'
            (default) or 'vertical'
        node_layout : {"hex", "rect"}
            The grid layout of nodes.
        reorient_links : bool, optional
            Whether or not to re-orient all links to point between -45 deg
            and +135 deg clockwise from "north" (i.e., along y axis). default
            is True.
        number_of_classes : int, optional
            Number of classes of material (e.g., different lithologies or grain
            sizes) making the layers of the StackedLayers.
        initial_allocation : int or array-like, optional
            Number of layers initially pre-allocated to speed up adding new layers.
            If an int, the layers are pre-allocated at the end; it an array-like,
            the first element corresponds to layers pre-allocated at the start,
            the second to layers pre-allocated at the end.
        new_allocation : int or array-like, optional
            Number of layers pre-allocated when adding new layers to speed up the
            process. If an array-like, different number of layers are pre-allocated
            at the start and at the end of the stacks.
        number_of_layers_to_fuse : int, optional
            Number of layers to merge together when calling `fuse`.
        number_of_top_layers : int, optional
            Number of layers at the top of the stack to ignore when calling `fuse`.
        fuse_continuously : bool, optional
            If True, fuse the layers as they are added; otherwise, wait until the
            number of unfused layers reaches `number_of_layers_to_fuse` (excluding
            the top layers) to fuse them.
        remove_empty_layers : bool, optional
            If True, remove the layers without any deposits at the top of the stack
            when adding a layer; otherwise, keep all the layers.

        Returns
        -------
        HexModelGrid
            A newly-created grid.
        """
        _HexModelGrid.__init__(
            self,
            shape,
            spacing,
            xy_of_lower_left,
            orientation,
            node_layout,
            reorient_links,
            xy_of_reference,
            xy_axis_name,
            xy_axis_units,
        )
        StackedLayersMixIn.__init__(
            self,
            number_of_classes,
            initial_allocation,
            new_allocation,
            number_of_layers_to_fuse,
            number_of_top_layers,
            fuse_continuously,
            remove_empty_layers,
        )


add_module_functions_to_class(HexModelGrid, "divergence.py", pattern="calc_*")
