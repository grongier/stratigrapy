"""Gravity-driven diffuser"""

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

from .._base import _BaseDiffuser


################################################################################
# Component

class GravityDrivenDiffuser(_BaseDiffuser):
    """Gravity-driven diffusion of a Landlab field in continental and marine domains.

    References
    ----------
    Rivenaes, J. C. (1992)
        Application of a dual‐lithology, depth‐dependent diffusion equation in stratigraphic simulation
        https://doi.org/10.1111/j.1365-2117.1992.tb00136.x
    Granjeon, D., & Joseph, P. (1999)
        Concepts and applications of a 3-D multiple lithology, diffusive model in stratigraphic modeling
        https://doi.org/10.2110/pec.99.62.0197
    """

    _name = "GravityDrivenDiffuser"

    def __init__(
        self,
        grid,
        diffusivity_cont=0.01,
        diffusivity_mar=0.001,
        wave_base=20.,
        max_erosion_rate=0.01,
        active_layer_rate=None,
        exponent_slope=1.,
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
        max_erosion_rate : float (m/time), optional
            The maximum erosion rate of the sediments. If None, all the sediments
            may be eroded in a single time step. The erosion rate defines the
            thickness of the active layer if `active_layer_rate` is None.
        active_layer_rate : float or array-like (m/time), optional
            The rate of formation of the active layer, which is used to determine
            the composition of the transported sediments. By default, it is set
            by the maximum erosion rate.
        exponent_slope : float (-)
            The exponent for the slope.
        fields_to_track : str or array-like, optional
            The name of the fields at grid nodes to add to the StackedLayers at
            each iteration.
        """
        self._neighbors = grid.active_adjacent_nodes_at_node
        n_neighbors = self._neighbors.shape[1]

        super().__init__(grid, diffusivity_cont, diffusivity_mar, wave_base,
                         n_neighbors, max_erosion_rate, active_layer_rate,
                         exponent_slope, fields_to_track)

        # Field for the steepest slope
        n_nodes = grid.number_of_nodes
        self._link_lengths = grid.length_of_link
        self._links_to_neighbors = grid.links_at_node
        self._slope = np.zeros((n_nodes, n_neighbors, 1))

        # Fields for sediment fluxes
        n_sediments = self._K_sed.shape[2]
        self._total_sediment_outflux = np.zeros((n_nodes, 1, n_sediments))
        self._ratio = np.zeros((n_nodes, 1, n_sediments))

    def _calculate_slopes(self):
        """
        Calculates the slope between each node and its neighbors.
        """
        self._slope[..., 0] = (self._topography[:, np.newaxis] - self._topography[self._neighbors])/self._link_lengths[self._links_to_neighbors]
        self._slope[self._neighbors == -1] = 0.
        self._slope[self._slope < 0.] = 0.

    def _calculate_sediment_outflux(self, dt):
        """
        Calculates the sediment outflux for multiple lithologies.
        """
        cell_area = self._grid.cell_area_at_node[:, np.newaxis, np.newaxis]

        self._active_layer_composition[self._grid.core_nodes, 0] = self._stratigraphy.get_superficial_composition(self.active_layer_rate*dt)

        self._sediment_outflux[:] = self._K_sed * cell_area * self._active_layer_composition * self._slope**self.n

    def _threshold_sediment_outflux(self, dt):
        """
        Thresholds the sediment outflux to not go beyond the amount of sediments
        available for each lithology.
        """
        cell_area = self._grid.cell_area_at_node[:, np.newaxis]

        self._max_sediment_outflux[self._grid.core_nodes, 0] = cell_area[self._grid.core_nodes]*np.minimum(self.max_erosion_rate, self._stratigraphy.class_thickness/dt)
        self._total_sediment_outflux[:] = np.sum(self._sediment_outflux, axis=1, keepdims=True)
        self._ratio[:] = 0.
        np.divide(self._max_sediment_outflux, self._total_sediment_outflux, out=self._ratio, where=self._total_sediment_outflux > 0.)
        self._sediment_outflux[self._ratio[:, 0, 0] < 1.] *= self._ratio[self._ratio[:, 0, 0] < 1.]

    def _calculate_sediment_influx(self):
        """
        Calculate the influx of sediments based on the outflux and the steepest
        slope.
        """
        self._sediment_influx[:] = 0.
        np.add.at(self._sediment_influx, self._neighbors, self._sediment_outflux)

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
        self._calculate_slopes()
        self._calculate_sediment_diffusivity()
        self._calculate_sediment_outflux(dt)
        self._threshold_sediment_outflux(dt)

        self._calculate_sediment_influx()

        self._apply_fluxes(dt, update_compatible, update)
