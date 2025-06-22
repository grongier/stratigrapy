"""Gravity-driven router"""

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

from .._base import _BaseRouter, _BaseDiffuser
from ..stream_power.cfuncs import calculate_sediment_influx


################################################################################
# Component


class GravityDrivenRouter(_BaseRouter, _BaseDiffuser):
    """Gravity-driven diffusion of a Landlab field in continental and marine domains
    based on a routing scheme.

    References
    ----------
    Rivenaes, J. C. (1992)
        Application of a dual‐lithology, depth‐dependent diffusion equation in stratigraphic simulation
        https://doi.org/10.1111/j.1365-2117.1992.tb00136.x
    Granjeon, D., & Joseph, P. (1999)
        Concepts and applications of a 3-D multiple lithology, diffusive model in stratigraphic modeling
        https://doi.org/10.2110/pec.99.62.0197
    """

    _name = "GravityDrivenRouter"

    def __init__(
        self,
        grid,
        diffusivity_cont=0.01,
        diffusivity_mar=0.001,
        wave_base=20.0,
        porosity=0.0,
        max_erosion_rate_sed=0.01,
        active_layer_rate_sed=None,
        bedrock_composition=1.0,
        max_erosion_rate_br=0.01,
        active_layer_rate_br=None,
        exponent_slope=1.0,
        fields_to_track=None,
    ):
        """
        Parameters
        ----------
        grid : ModelGrid
            A grid.
        diffusivity_cont : float or array-like (m/time)
            The diffusivity of the sediments over the continental domain for one
            or multiple lithologies.
        diffusivity_mar : float or array-like (m/time)
            The diffusivity of the sediments over the marine domain for one or
            multiple lithologies.
        wave_base : float (m)
            The wave base, below which weathering decreases exponentially.
        porosity : float or array-like (-)
            The porosity of the sediments at the time of deposition for one or
            multiple lithologies. When computing the active layer, this porosity
            is used unless the field 'sediment__porosity' is being tracked in
            the stratigraphy.
        max_erosion_rate_sed : float (m/time), optional
            The maximum erosion rate of the sediments. If None, all the sediments
            may be eroded in a single time step. The erosion rate defines the
            thickness of the active layer of the sediments if `active_layer_rate`
            is None.
        active_layer_rate_sed : float (m/time), optional
            The rate of formation of the active layer for sediments, which is used
            to determine the composition of the transported sediments. By default,
            it is set by the maximum erosion rate of the sediments.
        bedrock_composition : float or array-like (-)
            The composition of the material is added to the StackedLayers from
            the bedrock.
        max_erosion_rate_br : float (m/time)
            The maximum erosion rate of the bedrock. The erosion rate defines the
            thickness of the active layer of the bedrock if `active_layer_rate`
            is None.
        active_layer_rate_br : float (m/time), optional
            The rate of formation of the active layer for the bedrock, which is
            used to determine the composition of the transported sediments. By
            default, it is set by the maximum erosion rate of the bedrock.
        exponent_slope : float (-)
            The exponent for the slope.
        fields_to_track : str or array-like, optional
            The name of the fields at grid nodes to add to the StackedLayers at
            each iteration.
        """
        self._neighbors = grid.active_adjacent_nodes_at_node
        n_neighbors = self._neighbors.shape[1]

        super().__init__(grid=grid,
                               number_of_neighbors=n_neighbors,
                                diffusivity_cont=diffusivity_cont,
                                diffusivity_mar=diffusivity_mar,
                                wave_base=wave_base,
                                porosity=porosity,
                                max_erosion_rate_sed=max_erosion_rate_sed,
                                active_layer_rate_sed=active_layer_rate_sed,
                                bedrock_composition=bedrock_composition,
                                max_erosion_rate_br=max_erosion_rate_br,
                                active_layer_rate_br=active_layer_rate_br,
                                exponent_slope=exponent_slope,
                                fields_to_track=fields_to_track)

        # Field for the slopes
        n_nodes = grid.number_of_nodes
        self._link_lengths = grid.length_of_link
        self._links_to_neighbors = grid.links_at_node
        self._slope = np.zeros((n_nodes, n_neighbors, 1))

    def _calculate_slopes(self):
        """
        Calculates the slope between each node and its neighbors.
        """
        self._slope[..., 0] = (
            self._topography[:, np.newaxis] - self._topography[self._neighbors]
        ) / self._link_lengths[self._links_to_neighbors]
        self._slope[self._neighbors == -1] = 0.0
        self._slope[self._slope < 0.0] = 0.0

    def _calculate_sediment_outflux(self, dt):
        """
        Calculates the sediment outflux for multiple lithologies.
        """
        self._calculate_sediment_diffusivity()
        self._calculate_active_layer_composition(dt)
        self._calculate_slopes()

        self._sediment_outflux[:] = (
            self._K_sed
            * self._grid.cell_area_at_node[:, np.newaxis, np.newaxis]
            * self._active_layer_composition
            * self._slope**self._n
        )

    def run_one_step(self, dt, update_compatible=False, update=False):
        """Run the diffuser for one timestep, dt.

        Parameters
        ----------
        dt : float (time)
            The imposed timestep.
        update_compatible : bool, optional
            If False, create a new layer and deposit in that layer; otherwise,
            deposition occurs in the existing layer at the top of the stack only
            if the new layer is compatible with the existing layer.
        update : bool, optional
            If False, create a new layer and deposit in that layer; otherwise,
            deposition occurs in the existing layer.
        """
        cell_area = self._grid.cell_area_at_node[:, np.newaxis]

        self._calculate_sediment_outflux(dt)
        if self._max_erosion_rate_sed != self._active_layer_rate_sed or self.max_erosion_rate_br != self._active_layer_rate_br:
            self._calculate_active_layer(self._max_erosion_rate_sed*dt, self.max_erosion_rate_br*dt)
        self._max_sediment_outflux[:] = cell_area*self._active_layer[:, 0]/dt

        self._node_order = np.argsort(self._topography)
        self._sediment_influx[:] = 0.
        calculate_sediment_influx(
            self._node_order,
            self._neighbors,
            self._sediment_influx,
            self._sediment_outflux,
            self._max_sediment_outflux,
            dt,
        )

        self._apply_fluxes(dt, update_compatible, update)
