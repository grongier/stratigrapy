"""Water-driven displacer"""

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

from ...utils import convert_to_array, format_fields_to_track
from .cfuncs import calculate_sediment_fluxes


################################################################################
# Component

class WaterDrivenDisplacer(Component):
    """xi-q model for erosion and transport of a Landlab field in continental
    and marine domains.

    References
    ----------
    Davy, P., & Lague, D. (2009)
        Fluvial erosion/transport equation of landscape evolution models revisited
        https://doi.org/10.1029/2008JF001146
    Shobe, C. M., Tucker, G. E., & Barnhart, K. R. (2017)
        The SPACE 1.0 model: A Landlab component for 2-D calculation of sediment transport, bedrock erosion, and landscape evolution
        https://doi.org/10.5194/gmd-10-4577-2017
    """

    _name = "WaterDrivenDisplacer"

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
        erodibility_sed_cont=1e-10,
        erodibility_sed_mar=1e-10,
        settling_velocity=1.,
        wave_base=20.,
        max_erosion_rate_sed=1e-2,
        active_layer_rate=1e-2,
        erodibility_br_cont=1e-10,
        erodibility_br_mar=1e-10,
        bedrock_composition=1.,
        exponent_discharge=0.5,
        exponent_slope=1.,
        ref_water_flux=None,
        fields_to_track=None,
    ):
        """
        Parameters
        ----------
        grid : ModelGrid
            A grid.
        erodibility_sed_cont : float or array-like (m/time)
            The erodibility of the sediments over the continental domain for one
            or multiple lithologies.
        erodibility_sed_mar : float or array-like (m/time)
            The erodibility of the sediments over the marine domain for one or
            multiple lithologies.
        settling_velocity : float or array-like (m/time)
            The effective settling velocity for one or multiple lithologies.
        wave_base : float (m)
            The wave base, below which weathering decreases exponentially.
        max_erosion_rate_sed : float (m/time), optional
            The maximum erosion rate of the sediments. If None, all the sediments
            may be eroded in a single time step. The erosion rate defines the
            thickness of the active layer if `active_layer_rate` is None.
        active_layer_rate : float or array-like (m/time), optional
            The rate of formation of the active layer, which is used to determine
            the composition of the transported sediments. By default, it is set
            by the maximum erosion rate.
        erodibility_br_cont : float (m/time)
            The erodibility of the berock over the continental domain.
        erodibility_br_mar : float (m/time)
            The erodibility of the berock over the marine domain.
        bedrock_composition : float or array-like (-)
            The composition of the material is added to the StackedLayers from
            the bedrock.
        exponent_discharge : float (-)
            The exponent for the water discharge.
        exponent_slope : float (-)
            The exponent for the slope.
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
        self.K_sed_cont = convert_to_array(erodibility_sed_cont)
        n_sediments = len(self.K_sed_cont)
        self.K_sed_mar = convert_to_array(erodibility_sed_mar)[np.newaxis]
        self._K_sed = np.zeros((n_nodes, 1, n_sediments))
        self.settling_velocity = convert_to_array(settling_velocity)
        self.wave_base = wave_base
        self.max_erosion_rate_sed = np.inf if max_erosion_rate_sed is None else max_erosion_rate_sed
        self.active_layer_rate = max_erosion_rate_sed if active_layer_rate is None else active_layer_rate
        self.K_br_cont = convert_to_array(erodibility_br_cont)
        self.K_br_mar = convert_to_array(erodibility_br_mar)[np.newaxis]
        self._K_br = np.zeros((n_nodes, 1, n_sediments))
        self.bedrock_composition = convert_to_array(bedrock_composition)
        self.m = exponent_discharge
        self.n = exponent_slope
        self.fields_to_track = format_fields_to_track(fields_to_track)

        # Physical fields
        self._topography = grid.at_node['topographic__elevation']
        self._stratigraphy = grid.stacked_layers
        if self._stratigraphy.number_of_layers == 0:
            _fields_to_track = {field: grid.at_node[field][grid.core_nodes] for field in self.fields_to_track}
            self._stratigraphy.add(0., time=0., **_fields_to_track)
        self._bathymetry = grid.at_node['bathymetric__depth'][:, np.newaxis]
        self._time = 0.

        # Fields for stream power
        self._node_order = grid.at_node["flow__upstream_node_order"]
        self._flow_receivers = grid.at_node["flow__receiver_node"][..., np.newaxis]
        self._link_to_receiver = grid.at_node["flow__link_to_receiver_node"][..., np.newaxis]
        self._slope = grid.at_node["topographic__steepest_slope"][..., np.newaxis]
        if self._flow_receivers.ndim == 2:
            self._flow_receivers = self._flow_receivers[..., np.newaxis]
            self._link_to_receiver = self._link_to_receiver[..., np.newaxis]
            self._slope = self._slope[..., np.newaxis]
        n_receivers = self._flow_receivers.shape[1]
        if "flow__receiver_proportions" in grid.at_node:
            self._flow_proportions = grid.at_node["flow__receiver_proportions"][..., np.newaxis]
        else:
            self._flow_proportions = np.ones((n_nodes, n_receivers, 1))
        self._water_flux = grid.at_node["surface_water__discharge"][:, np.newaxis, np.newaxis]
        self.ref_water_flux = ref_water_flux

        # Fields for sediment fluxes
        if "sediment__unit_flux_in" in grid.at_node:
            self._sediment_input = grid.at_node["sediment__unit_flux_in"]
            if self._sediment_input.ndim == 1:
                self._sediment_input = self._sediment_input[:, np.newaxis]
        else:
            self._sediment_input = np.zeros((n_nodes, n_sediments))
        self._sediment_influx = np.zeros((n_nodes, n_sediments))
        self._sediment_outflux = np.zeros((n_nodes, n_receivers, n_sediments))
        self._erosion_capacity_sed = np.zeros((n_nodes, n_receivers, n_sediments))
        self._erosion_capacity_br = np.zeros((n_nodes, n_receivers, n_sediments))
        self._max_sediment_outflux = np.zeros((n_nodes, n_sediments))
        self._max_bedrock_outflux = np.zeros((n_nodes, n_sediments))
        self._sediment_rate = np.zeros((n_nodes, n_sediments))
        self._active_layer_composition = np.zeros((n_nodes, 1, n_sediments))

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
        self._K_sed[self._bathymetry[:, 0] == 0.] = self.K_sed_cont
        self._K_sed[self._bathymetry[:, 0] > 0., 0] = self.K_sed_mar*np.exp(-self._bathymetry[self._bathymetry[:, 0] > 0.]/self.wave_base)

    def _calculate_bedrock_diffusivity(self):
        """
        Calculates the diffusivity coefficient of the bedrock over the continental
        and marine domains.
        """
        self._K_br[self._bathymetry[:, 0] == 0.] = self.K_br_cont
        self._K_br[self._bathymetry[:, 0] > 0., 0] = self.K_br_mar*np.exp(-self._bathymetry[self._bathymetry[:, 0] > 0.]/self.wave_base)
        self._K_br[self._grid.core_nodes][self._stratigraphy.thickness > 0.] = 0.

    def _calculate_sediment_capacity(self, dt):
        """
        Calculates the erosion and transport capacities of sediments for multiple
        lithologies.
        """
        cell_area = self._grid.cell_area_at_node[:, np.newaxis, np.newaxis]

        self._active_layer_composition[self._grid.core_nodes, 0] = self._stratigraphy.get_superficial_composition(self.active_layer_rate*dt)

        self._erosion_capacity_sed[:] = self._K_sed * cell_area * self._active_layer_composition * (self._water_flux*self._flow_proportions)**self.m * self._slope**self.n

    def _calculate_bedrock_capacity(self):
        """
        Calculates the erosion capacity of the bedrock for multiple lithologies.
        """
        cell_area = self._grid.cell_area_at_node[:, np.newaxis, np.newaxis]

        self._erosion_capacity_br[:] = self._K_br * cell_area * self.bedrock_composition * (self._water_flux*self._flow_proportions)**self.m * self._slope**self.n

    def run_one_step(self, dt):
        """Run the transporter for one timestep, dt.

        Parameters
        ----------
        dt : float (time)
            The imposed timestep.
        """
        core_nodes = self._grid.core_nodes
        cell_area = self._grid.cell_area_at_node[:, np.newaxis]

        self._time += dt

        self._normalize_water_flux()

        self._calculate_sediment_diffusivity()
        self._calculate_bedrock_diffusivity()
        self._calculate_sediment_capacity(dt)
        self._calculate_bedrock_capacity()
        self._max_sediment_outflux[self._grid.core_nodes] = cell_area[core_nodes]*np.minimum(self.max_erosion_rate_sed, self._stratigraphy.class_thickness/dt)
        # self._max_bedrock_outflux[self._grid.core_nodes] = cell_area[core_nodes]*self.max_erosion_rate_br

        self._sediment_influx[:] = self._sediment_input
        self._sediment_outflux[:] = 0.
        calculate_sediment_fluxes(self._node_order,
                                  cell_area[:, 0],
                                  self._flow_receivers[..., 0],
                                  self._water_flux[:, 0, 0],
                                  self._flow_proportions[..., 0],
                                  self._sediment_influx,
                                  self._sediment_outflux,
                                  self.settling_velocity,
                                  self._erosion_capacity_sed,
                                  self._erosion_capacity_br,
                                  self._max_sediment_outflux,
                                  dt)

        self._sediment_rate[core_nodes] = (self._sediment_influx[core_nodes] - np.sum(self._sediment_outflux[core_nodes], axis=1))/cell_area[core_nodes]
        fields_to_track = {field: self._grid.at_node[field][core_nodes] for field in self.fields_to_track}
        # _sediment_rate also includes bedrock erosion, which only affects the
        # topography (sediment thickness in _stratigraphy won't become negative
        # if more material is to be removed than what is available)
        self._stratigraphy.add(self._sediment_rate[core_nodes]*dt, time=self._time, **fields_to_track)
        self._topography[core_nodes] += np.sum(self._sediment_rate[core_nodes], axis=1)*dt
