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
from landlab import Component

from ...utils import convert_to_array, reshape_to_match, format_fields_to_track


################################################################################
# Component

class GravityDrivenDiffuser(Component):
    """Gravity-driven diffusion of a Landlab field in continental and marine domains.

    References
    ----------
    Granjeon, D. (1996)
        Modélisation stratigraphique déterministe: Conception et applications d'un modèle diffusif 3D multilithologique
        https://theses.hal.science/tel-00648827
    Granjeon, D., & Joseph, P. (1999)
        Concepts and applications of a 3-D multiple lithology, diffusive model in stratigraphic modeling
        https://doi.org/10.2110/pec.99.62.0197
    """

    _name = "GravityDrivenDiffuser"

    _unit_agnostic = True

    _info = {
        "flow__receiver_node": {
            "dtype": int,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Node array of receivers (node that receives flow from current node)",
        },
        "bathymetric__depth": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "The water depth under the sea",
        },
        "topographic__elevation": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Land surface topographic elevation",
        },
        "topographic__steepest_slope": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "The steepest *downhill* slope",
        },
    }

    def __init__(
        self,
        grid,
        diffusivity_cont=0.01,
        diffusivity_mar=0.001,
        wave_base=20.,
        max_erosion_rate=None,
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
            The maximum erosion rate of the sediments, which defines the
            thickness of the superficial layer of sediments that can be eroded
            at each time step. If None, all the sediments may be eroded in a
            single time step.
        exponent_slope : float (-)
            The exponent for the slope.
        fields_to_track : str or array-like, optional
            The name of the fields at grid nodes to add to the StackedLayers at
            each iteration.
        """
        super().__init__(grid)

        # Parameters
        n_nodes = grid.number_of_nodes
        self.K_cont = convert_to_array(diffusivity_cont)
        n_sediments = len(self.K_cont)
        self.K_mar = convert_to_array(diffusivity_mar)
        self.wave_base = wave_base
        self._K_sed = np.zeros((n_nodes, n_sediments))
        self.max_erosion_rate = np.inf if max_erosion_rate is None else max_erosion_rate
        self.n = exponent_slope
        self.fields_to_track = format_fields_to_track(fields_to_track)

        # Physical fields
        self._topography = grid.at_node['topographic__elevation']
        self._stratigraphy = grid.stacked_layers
        if self._stratigraphy.number_of_layers == 0:
            _fields_to_track = {field: grid.at_node[field][grid.core_nodes] for field in self.fields_to_track}
            self._stratigraphy.add(0., time=0., **_fields_to_track)
        self._bathymetry = reshape_to_match(grid.at_node['bathymetric__depth'], (n_nodes, n_sediments))
        self._time = 0.

        # Field for the steepest slope
        self._flow_receivers = grid.at_node["flow__receiver_node"]
        n_receivers = 1 if self._flow_receivers.ndim == 1 else self._flow_receivers.shape[1]
        self._flow_receivers = reshape_to_match(self._flow_receivers, (n_nodes, n_receivers))
        self._slope = reshape_to_match(grid.at_node["topographic__steepest_slope"], (n_nodes, n_receivers))
        self._steepest_receivers = np.zeros((n_nodes, 1), dtype=int)

        # Fields for sediment fluxes
        self._sediment_influx = np.zeros((n_nodes, n_sediments))
        self._sediment_outflux = np.zeros((n_nodes, n_sediments))
        self._max_sediment_outflux = np.zeros((n_nodes, n_sediments))
        self._sediment_rate = np.zeros((n_nodes, n_sediments))
        self._weathering_depth = np.zeros((n_nodes, 1))
        self._superficial_layer = np.zeros((n_nodes, n_sediments))
        self._superficial_composition = np.zeros((n_nodes, n_sediments))
        self._superficial_thickness = np.zeros((n_nodes, 1))

        # Grid elements
        self._cell_area = reshape_to_match(self._grid.cell_area_at_node, (n_nodes, n_sediments))

    def _calculate_sediment_diffusivity(self):
        """
        Calculates the diffusivity coefficient of the sediments over the continental
        and marine domains.
        """
        self._K_sed[self._bathymetry[:, 0] == 0.] = self.K_cont
        self._K_sed[self._bathymetry[:, 0] > 0.] = self.K_mar*np.exp(-self._bathymetry[self._bathymetry[:, 0] > 0.]/self.wave_base)

    def _calculate_weathering_depth(self, dt):
        """
        Calculates the weathering depth over the continental and marine domains.
        This defines the superficial layer where sediments can be mobilized.
        """
        self._weathering_depth[:] = self.max_erosion_rate*dt
        self._weathering_depth[self._bathymetry[:, 0] > 0.] *= np.exp(-self._bathymetry[self._bathymetry[:, 0] > 0.]/self.wave_base)

    def _calculate_superfical_layer(self):
        """
        Calculates the superfical layer.
        """
        core_nodes = self._grid.core_nodes

        self._superficial_layer[core_nodes] = self._stratigraphy.get_superficial_layer(self._weathering_depth[core_nodes])
        self._superficial_thickness[:] = np.sum(self._superficial_layer, axis=1, keepdims=True)

    def _calculate_sediment_outflux(self):
        """
        Calculates the sediment outflux for multiple lithologies.
        """
        self._superficial_composition[self._superficial_thickness[:, 0] > 0.] = self._superficial_layer[self._superficial_thickness[:, 0] > 0.]/self._superficial_thickness[self._superficial_thickness[:, 0] > 0.]
        self._superficial_composition[self._superficial_thickness[:, 0] == 0.] = 0.

        self._steepest_receivers[:] = np.argmax(self._slope, axis=1, keepdims=True)
        self._sediment_outflux[:] = self._K_sed * self._cell_area * self._superficial_composition * np.take_along_axis(self._slope, self._steepest_receivers, axis=1)**self.n

    def _threshold_sediment_outflux(self, dt):
        """
        Thresholds the sediment outflux to not go beyond the amount of sediments
        available in the superficial layer for each lithology.
        """
        self._max_sediment_outflux[:] = self._superficial_layer*self._cell_area/dt
        self._sediment_outflux[self._sediment_outflux > self._max_sediment_outflux] = self._max_sediment_outflux[self._sediment_outflux > self._max_sediment_outflux]

    def _calculate_sediment_influx(self):
        """
        Calculate the influx of sediments based on the outflux and the steepest
        slope.
        """
        self._sediment_influx[np.take_along_axis(self._flow_receivers, self._steepest_receivers, axis=1)[:, 0]] = self._sediment_outflux

    def run_one_step(self, dt):
        """Run the diffuser for one timestep, dt.

        Parameters
        ----------
        dt : float (time)
            The imposed timestep.
        """
        core_nodes = self._grid.core_nodes

        self._time += dt

        self._calculate_sediment_diffusivity()
        self._calculate_weathering_depth(dt)
        self._calculate_superfical_layer()
        self._calculate_sediment_outflux()
        self._threshold_sediment_outflux(dt)

        self._calculate_sediment_influx()

        self._sediment_rate[core_nodes] = (self._sediment_influx[core_nodes] - self._sediment_outflux[core_nodes])/self._cell_area[core_nodes]
        fields_to_track = {field: self._grid.at_node[field][core_nodes] for field in self.fields_to_track}
        self._stratigraphy.add(self._sediment_rate[core_nodes]*dt, time=self._time, **fields_to_track)
        self._topography[core_nodes] += np.sum(self._sediment_rate[core_nodes], axis=1)*dt
