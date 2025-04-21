"""Linear-decline transporter"""

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
from .cfuncs import calculate_linear_decline_sediment_influx


################################################################################
# Component

class LinearDeclineTransporter(Component):
    """Linear decline erosion and transport of a Landlab field in continental
    and marine domains.

    References
    ----------
    Whipple, K. X., & Tucker, G. E. (2002)
        Implications of sediment‐flux‐dependent river incision models for landscape evolution
        https://doi.org/10.1029/2000JB000044
    """

    _name = "LinearDeclineTransporter"

    _unit_agnostic = True

    _info = {
        "flow__upstream_node_order": {
            "dtype": int,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Node array containing downstream-to-upstream ordered list of node IDs",
        },
        "flow__receiver_node": {
            "dtype": int,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Node array of receivers (node that receives flow from current node)",
        },
        "flow__receiver_proportions": {
            "dtype": float,
            "intent": "in",
            "optional": True,
            "units": "-",
            "mapping": "node",
            "doc": "Node array of proportion of flow sent to each receiver.",
        },
        "sediment__unit_flux_in": {
            "dtype": float,
            "intent": "in",
            "optional": True,
            "units": "m3/s",
            "mapping": "node",
            "doc": "Sediment flux as boundary condition",
        },
        "surface_water__discharge": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m3/s",
            "mapping": "node",
            "doc": "Volumetric discharge of surface water",
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
    }

    def __init__(
        self,
        grid,
        transportability_cont=1e-8,
        transportability_mar=1e-9,
        wave_base=20.,
        max_erosion_rate=1e-3,
        erodibility_cont=1e-10,
        erodibility_mar=1e-10,
        bedrock_composition=1.,
        exponent_discharge_sed=1.5,
        exponent_slope_sed=1.,
        exponent_discharge_br=0.5,
        exponent_slope_br=1.,
        ref_water_flux=None,
        fields_to_track=None,
    ):
        """
        Parameters
        ----------
        grid : ModelGrid
            A grid.
        transportability_cont : float or array-like (m/time)
            The transportability of the sediments over the continental domain
            for one or multiple lithologies.
        transportability_mar : float or array-like (m/time)
            The transportability of the sediments over the marine domain for one
            or multiple lithologies.
        wave_base : float (m)
            The wave base, below which weathering decreases exponentially.
        max_erosion_rate : float (m/time), optional
            The maximum erosion rate of the sediments, which defines the
            thickness of the superficial layer of sediments that can be eroded
            at each time step. If None, all the sediments may be eroded in a
            single time step.
        erodibility_cont : float (m/time)
            The erodibility of the berock over the continental domain.
        erodibility_mar : float (m/time)
            The erodibility of the berock over the marine domain.
        bedrock_composition : float or array-like (-)
            The composition of the material is added to the StackedLayers from
            the bedrock.
        exponent_discharge_sed : float (-)
            The exponent for the water discharge for sediment transport.
        exponent_slope_sed : float (-)
            The exponent for the slope for sediment transport.
        exponent_discharge_br : float (-)
            The exponent for the water discharge for bedrock erosion.
        exponent_slope_br : float (-)
            The exponent for the slope for bedrock erosion.
        ref_water_flux : float or string (m3/time), optional
            The reference water flux by which the water discharge is normalized.
            If a float, that value is used at each time step; if 'max', the
            maximum value of discharge at each time step is used.
        fields_to_track : str or array-like, optional
            The name of the fields at grid nodes to add to the StackedLayers at
            each iteration.
        """
        super().__init__(grid)

        self.initialize_output_fields()

        # Parameters
        n_nodes = grid.number_of_nodes
        self.K_cont = convert_to_array(transportability_cont)
        n_sediments = len(self.K_cont)
        self.K_mar = convert_to_array(transportability_mar)[np.newaxis]
        self.wave_base = wave_base
        self._K_sed = np.zeros((n_nodes, 1, n_sediments))
        self.max_erosion_rate = np.inf if max_erosion_rate is None else max_erosion_rate
        self.K_br_cont = convert_to_array(erodibility_cont)
        self.K_br_mar = convert_to_array(erodibility_mar)[np.newaxis]
        self._K_br = np.zeros((n_nodes, 1, n_sediments))
        self.bedrock_composition = convert_to_array(bedrock_composition)
        self.m_sed = exponent_discharge_sed
        self.n_sed = exponent_slope_sed
        self.m_br = exponent_discharge_br
        self.n_br = exponent_slope_br
        self.fields_to_track = format_fields_to_track(fields_to_track)

        # Physical fields
        self._topography = grid.at_node['topographic__elevation']
        self._stratigraphy = grid.stacked_layers
        if self._stratigraphy.number_of_layers == 0:
            _fields_to_track = {field: grid.at_node[field][grid.core_nodes] for field in self.fields_to_track}
            self._stratigraphy.add(0., time=0., **_fields_to_track)
        self._bathymetry = reshape_to_match(grid.at_node['bathymetric__depth'], (n_nodes, n_sediments))
        self._time = 0.

        # Fields for stream power
        self._node_order = grid.at_node["flow__upstream_node_order"]
        self._flow_receivers = grid.at_node["flow__receiver_node"]
        n_receivers = 1 if self._flow_receivers.ndim == 1 else self._flow_receivers.shape[1]
        self._flow_receivers = reshape_to_match(self._flow_receivers, (n_nodes, n_receivers, n_sediments))
        self._slope = reshape_to_match(grid.at_node["topographic__steepest_slope"],
                                       (n_nodes, n_receivers, n_sediments))
        if "flow__receiver_proportions" in grid.at_node:
            self._flow_proportions = reshape_to_match(grid.at_node["flow__receiver_proportions"],
                                                      (n_nodes, n_receivers, n_sediments))
        else:
            self._flow_proportions = np.ones((n_nodes, n_receivers, 1))
        self._water_flux = reshape_to_match(grid.at_node["surface_water__discharge"],
                                            (n_nodes, n_receivers, n_sediments))
        self.ref_water_flux = ref_water_flux

        # Fields for sediment fluxes
        if "sediment__unit_flux_in" in grid.at_node:
            self._sediment_input = reshape_to_match(grid.at_node["sediment__unit_flux_in"],
                                                    (n_nodes, n_sediments))
        else:
            self._sediment_input = np.zeros((n_nodes, n_sediments))
        self._sediment_influx = np.zeros((n_nodes, n_sediments))
        self._sediment_outflux = np.zeros((n_nodes, n_receivers, n_sediments))
        self._bedrock_outflux = np.zeros((n_nodes, n_receivers, n_sediments))
        self._sediment_rate = np.zeros((n_nodes, n_sediments))
        self._bedrock_rate = np.zeros(n_nodes)
        self._weathering_depth = np.zeros((n_nodes, 1, 1))
        self._superficial_layer = np.zeros((n_nodes, 1, n_sediments))
        self._superficial_composition = np.zeros((n_nodes, 1, n_sediments))
        self._superficial_thickness = np.zeros((n_nodes, 1, 1))

        # Grid elements
        self._cell_area = reshape_to_match(self._grid.cell_area_at_node,
                                           (n_nodes, n_receivers, n_sediments))

    def _normalize_water_flux(self):
        """
        Normalizes the water flux if needed.
        """
        if self.ref_water_flux == 'max':
            self._water_flux[:] /= np.max(self._water_flux)
        elif isinstance(self.ref_water_flux, (int, float)):
            self._water_flux[:] /= self.ref_water_flux

    def _calculate_sediment_diffusivity(self):
        """
        Calculates the diffusivity coefficient of the sediments over the continental
        and marine domains.
        """
        self._K_sed[self._bathymetry[:, 0] == 0.] = self.K_cont
        self._K_sed[self._bathymetry[:, 0] > 0., 0] = self.K_mar*np.exp(-self._bathymetry[self._bathymetry[:, 0] > 0.]/self.wave_base)

    def _calculate_bedrock_diffusivity(self):
        """
        Calculates the erodibility coefficient of the bedrock over the continental
        and marine domains.
        """
        self._K_br[self._bathymetry[:, 0] == 0.] = self.K_br_cont
        self._K_br[self._bathymetry[:, 0] > 0., 0] = self.K_br_mar*np.exp(-self._bathymetry[self._bathymetry[:, 0] > 0.]/self.wave_base)

    def _calculate_weathering_depth(self, dt):
        """
        Calculates the weathering depth over the continental and marine domains.
        This defines the superficial layer where sediments can be mobilized.
        """
        self._weathering_depth[:] = self.max_erosion_rate*dt
        self._weathering_depth[self._bathymetry[:, 0] > 0., 0] *= np.exp(-self._bathymetry[self._bathymetry[:, 0] > 0.]/self.wave_base)

    def _calculate_superfical_layer(self):
        """
        Calculates the superfical layer.
        """
        core_nodes = self._grid.core_nodes

        self._superficial_layer[core_nodes, 0] = self._stratigraphy.get_superficial_layer(self._weathering_depth[core_nodes])
        self._superficial_thickness[:] = np.sum(self._superficial_layer, axis=2, keepdims=True)

    def _calculate_sediment_outflux(self):
        """
        Calculates the sediment outflux for multiple lithologies.
        """
        self._superficial_composition[self._superficial_thickness[:, 0] > 0.] = self._superficial_layer[self._superficial_thickness[:, 0] > 0.]/self._superficial_thickness[self._superficial_thickness[:, 0] > 0.]
        self._superficial_composition[self._superficial_thickness[:, 0] == 0.] = 0.

        self._sediment_outflux[:] = self._K_sed * self._cell_area * self._superficial_composition * (self._water_flux*self._flow_proportions)**self.m_sed * self._slope**self.n_sed

    def _calculate_bedrock_outflux(self):
        """
        Calculates the bedrock outflux for multiple lithologies.
        """
        self._bedrock_outflux[:] = self._K_br * self._cell_area * self.bedrock_composition * (self._water_flux*self._flow_proportions)**self.m_br * self._slope**self.n_br

    def run_one_step(self, dt):
        """Run the transporter for one timestep, dt.

        Parameters
        ----------
        dt : float (time)
            The imposed timestep.
        """
        core_nodes = self._grid.core_nodes

        self._time += dt

        self._normalize_water_flux()

        self._calculate_sediment_diffusivity()
        self._calculate_bedrock_diffusivity()
        self._calculate_weathering_depth(dt)
        self._calculate_superfical_layer()
        self._calculate_sediment_outflux()
        self._calculate_bedrock_outflux()

        self._sediment_influx[:] = self._sediment_input
        calculate_linear_decline_sediment_influx(self._node_order,
                                                 self._flow_receivers[..., 0],
                                                 self._cell_area[:, 0, 0],
                                                 self._superficial_layer[:, 0],
                                                 self._sediment_influx,
                                                 self._sediment_outflux,
                                                 self._bedrock_outflux,
                                                 dt)

        self._sediment_rate[core_nodes] = (self._sediment_influx[core_nodes] - np.sum(self._sediment_outflux[core_nodes], axis=1))/self._cell_area[core_nodes, 0]
        fields_to_track = {field: self._grid.at_node[field][core_nodes] for field in self.fields_to_track}
        self._stratigraphy.add(self._sediment_rate[core_nodes]*dt, time=self._time, **fields_to_track)
        self._bedrock_rate[core_nodes] = np.sum(self._bedrock_outflux[core_nodes], axis=(1, 2))/self._cell_area[core_nodes, 0, 0]
        self._topography[core_nodes] += np.sum(self._sediment_rate[core_nodes], axis=1)*dt - self._bedrock_rate[core_nodes]*dt
