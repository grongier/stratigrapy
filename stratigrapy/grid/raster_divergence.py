"""Calculate multiple flux divergence on a raster grid"""

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


import numpy as np

from .cfuncs import _calc_mult_flux_div_at_node


################################################################################
# Flux divergence


def calc_mult_flux_div_at_node(grid, unit_flux, out=None):
    """Calculate the divergences of link-based fluxes at nodes.

    Given a flux per unit width across each face in the grid, calculate the net
    outflux (or influx, if negative) divided by cell area, at each node (zero
    or "out" value for nodes without cells).

    Parameters
    ----------
    grid : ModelGrid
        A ModelGrid.
    unit_flux : ndarray
        Flux per unit width along links for multiple classes treated in parallel
        (x number of links, y number of classes).

    Returns
    -------
    ndarray (x number of nodes, y number of classes)
        Flux divergences at nodes.
    """
    if unit_flux.shape[0] != grid.number_of_links:
        raise ValueError("Parameter unit_flux must be num links " "long")
    if out is None:
        out = np.zeros((grid.number_of_nodes, unit_flux.shape[1]))
    elif out.shape[0] != grid.number_of_nodes:
        raise ValueError("output buffer length mismatch with number of nodes")

    out.reshape(*grid.shape, unit_flux.shape[1])[:, (0, -1)] = 0.0
    out.reshape(*grid.shape, unit_flux.shape[1])[(0, -1), :] = 0.0

    _calc_mult_flux_div_at_node(grid.shape, (grid.dx, grid.dy), unit_flux, out)

    return out
