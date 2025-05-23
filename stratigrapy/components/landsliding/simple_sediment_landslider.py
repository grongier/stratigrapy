"""Simple sediment landslider"""

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
from landlab import RasterModelGrid

from ...utils import convert_to_array, format_fields_to_track
from .cfuncs import calculate_sediment_influx


################################################################################
# Component

class SimpleSedimentLandslider(Component):
    """Simple slope failure and mass flows of sediments in a StackedLayers.

    References
    ----------
    Granjeon, D. (2014)
        3D forward modelling of the impact of sediment transport and base level cycles on continental margins and incised valleys
        https://doi.org/10.1002/9781118920435.ch16
    """

    _name = "SimpleSedimentLandslider"

    _unit_agnostic = True

    _info = {
        "flow__link_to_receiver_node": {
            "dtype": int,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "ID of link downstream of each node, which carries the discharge",
        },
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
        repose_angle_cont=2.,
        repose_angle_mar=2.,
        active_layer_rate=1e-2,
        update_compatible=False,
        fields_to_track=None,
    ):
        """
        Parameters
        ----------
        grid : ModelGrid
            A grid.
        repose_angle_cont : float or array-like (degree)
            The static angle of repose in the continental domain for one or
            multiple lithologies above which sediments move downslope.
        repose_angle_mar : float or array-like (degree)
            The static angle of repose in the continental domain for one or
            multiple lithologies above which sediments move downslope.
        active_layer_rate : float or array-like (m/time), optional
            The rate of formation of the active layer, which is used to determine
            the composition of the transported sediments. Erosion is not limited
            to that layer. If None, the entire layer of sediments is used as
            active layer.
        update_compatible : bool, optional
            If False, create a new layer and deposit in that layer; otherwise,
            deposition occurs in the existing layer at the top of the stack only
            if the new layer is compatible with the existing layer.
        fields_to_track : str or array-like, optional
            The name of the fields at grid nodes to add to the StackedLayers at
            each iteration.
        """
        super().__init__(grid)

        # Parameters
        n_nodes = grid.number_of_nodes
        self.repose_angle_cont = convert_to_array(repose_angle_cont)
        n_sediments = len(self.repose_angle_cont)
        self.repose_angle_mar = convert_to_array(repose_angle_mar)
        self.active_layer_rate = np.inf if active_layer_rate is None else active_layer_rate
        self.update_compatible = update_compatible
        self.fields_to_track = format_fields_to_track(fields_to_track)

        # Physical fields
        self._topography = grid.at_node['topographic__elevation']
        self._stratigraphy = grid.stacked_layers
        if self._stratigraphy.number_of_layers == 0:
            _fields_to_track = {field: grid.at_node[field][grid.core_nodes] for field in self.fields_to_track}
            self._stratigraphy.add(0., time=0., **_fields_to_track)
        self._bathymetry = grid.at_node['bathymetric__depth']
        self._time = 0.

        # Field for the steepest slope
        if isinstance(grid, RasterModelGrid):
            self._link_lengths = grid.length_of_d8
        else:
            self._link_lengths = grid.length_of_link
        self._node_order = np.zeros(n_nodes, dtype=int)
        self._flow_receivers = grid.at_node["flow__receiver_node"]
        self._link_to_receiver = grid.at_node["flow__link_to_receiver_node"]
        self._slope = grid.at_node["topographic__steepest_slope"]
        if self._flow_receivers.ndim == 1:
            self._flow_receivers = self._flow_receivers[:, np.newaxis]
            self._link_to_receiver = self._link_to_receiver[:, np.newaxis]
            self._slope = self._slope[:, np.newaxis]
        self._steepest_receivers = np.zeros((n_nodes, 1), dtype=int)
        self._steepest_slope = np.zeros((n_nodes, 1))
        self._steepest_link = np.zeros((n_nodes, 1), dtype=int)

        # Fields for sediment fluxes
        self._repose_angle = np.zeros((n_nodes, n_sediments))
        self._repose_slope = np.zeros((n_nodes, 1))
        self._slope_difference = np.zeros((n_nodes, 1))
        self._sediment_influx = np.zeros((n_nodes, n_sediments))
        self._sediment_outflux = np.zeros((n_nodes, n_sediments))
        self._max_sediment_outflux = np.zeros((n_nodes, n_sediments))
        self._sediment_rate = np.zeros((n_nodes, n_sediments))
        self._active_layer_composition = np.zeros((n_nodes, n_sediments))

    def _calculate_sediment_repose_angle(self):
        """
        Calculates the repose angle of the sediments over the continental and
        marine domains.
        """
        self._repose_angle[self._bathymetry == 0.] = self.repose_angle_cont
        self._repose_angle[self._bathymetry > 0.] = self.repose_angle_mar

    def _calculate_sediment_outflux(self, dt):
        """
        Calculates the sediment outflux for multiple lithologies.
        """
        cell_area = self._grid.cell_area_at_node[:, np.newaxis]

        self._active_layer_composition[self._grid.core_nodes] = self._stratigraphy.get_superficial_composition(self.active_layer_rate*dt)

        self._steepest_receivers[:] = np.argmax(self._slope, axis=1, keepdims=True)
        self._steepest_slope[:] = np.take_along_axis(self._slope, self._steepest_receivers, axis=1)
        self._steepest_link[:] = np.take_along_axis(self._link_to_receiver, self._steepest_receivers, axis=1)
        self._repose_slope[:] = np.arctan(np.deg2rad(np.sum(self._repose_angle*self._active_layer_composition, axis=1, keepdims=True)))
        self._slope_difference[:] = self._steepest_slope - self._repose_slope
        self._slope_difference[self._slope_difference < 0.] = 0.
        self._sediment_outflux[:] = self._active_layer_composition*self._slope_difference*self._link_lengths[self._steepest_link]*cell_area/dt

    def _threshold_sediment_outflux(self, dt):
        """
        Thresholds the sediment outflux to not go beyond the amount of sediments
        available for each lithology.
        """
        cell_area = self._grid.cell_area_at_node[:, np.newaxis]

        self._max_sediment_outflux[self._grid.core_nodes] = self._stratigraphy.class_thickness*cell_area[self._grid.core_nodes]/dt
        self._sediment_outflux[self._sediment_outflux > self._max_sediment_outflux] = self._max_sediment_outflux[self._sediment_outflux > self._max_sediment_outflux]

    def run_one_step(self, dt):
        """Run the landslider for one timestep, dt.

        Parameters
        ----------
        dt : float (time)
            The imposed timestep.
        """
        core_nodes = self._grid.core_nodes
        cell_area = self._grid.cell_area_at_node[:, np.newaxis]

        self._time += dt

        self._calculate_sediment_repose_angle()
        self._calculate_sediment_outflux(dt)
        self._threshold_sediment_outflux(dt)

        self._sediment_influx[:] = 0.
        self._node_order[:] = np.argsort(self._topography)
        calculate_sediment_influx(self._node_order,
                                  np.take_along_axis(self._flow_receivers, self._steepest_receivers, axis=1)[:, 0],
                                  self._link_lengths[self._steepest_link][:, 0],
                                  cell_area[:, 0],
                                  self._steepest_slope[:, 0],
                                  self._repose_angle,
                                  self._sediment_influx,
                                  self._sediment_outflux,
                                  dt)

        self._sediment_rate[core_nodes] = (self._sediment_influx[core_nodes] - self._sediment_outflux[core_nodes])/cell_area[core_nodes]
        fields_to_track = {field: self._grid.at_node[field][core_nodes] for field in self.fields_to_track}
        self._stratigraphy.add(self._sediment_rate[core_nodes]*dt, update_compatible=self.update_compatible, time=self._time, **fields_to_track)
        self._topography[core_nodes] += np.sum(self._sediment_rate[core_nodes], axis=1)*dt
