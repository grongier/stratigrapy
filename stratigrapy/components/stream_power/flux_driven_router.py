"""Flux-driven router"""

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

from .._base import _BaseRouter, _BaseStreamPower
from ...utils import convert_to_array
from .cfuncs import calculate_sediment_fluxes


################################################################################
# Component


class FluxDrivenRouter(_BaseRouter, _BaseStreamPower):
    """SPACE model for erosion and transport of a Landlab field in continental
    and marine domains.

    This is a simple implementation that doesn't include all the elements of the
    original Landlab component. In particular, this component updates the
    stratigraphy based on the difference between sediment influx and outflux,
    and not based on erosion and deposition like SPACE.

    TODO: Properly adapt to the marine domain. For now the erosion coefficient
    follows Granjeon (1996) and decreases as the bathymetry increases; the
    settling velocity follows a similar approach to force deposition in the
    marine domain.

    References
    ----------
    Shobe, C. M., Tucker, G. E., & Barnhart, K. R. (2017)
        The SPACE 1.0 model: A Landlab component for 2-D calculation of sediment transport, bedrock erosion, and landscape evolution
        https://doi.org/10.5194/gmd-10-4577-2017
    Granjeon, D. (1996)
        Modélisation stratigraphique déterministe: Conception et applications d'un modèle diffusif 3D multilithologique
        https://theses.hal.science/tel-00648827
    """

    _name = "FluxDrivenRouter"

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
        "water__unit_flux_in": {
            "dtype": float,
            "intent": "in",
            "optional": True,
            "units": "m/s",
            "mapping": "node",
            "doc": "External volume water per area per time input to each node (e.g., rainfall rate)",
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
        "bathymetric__depth": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "The water depth under the sea",
        },
    }

    def __init__(
        self,
        grid,
        erodibility_sed_cont=1e-10,
        erodibility_sed_mar=1e-10,
        wave_base=20.0,
        settling_velocity=1.0,
        critical_flux_sed=0.0,
        critical_thickness=0.1,
        porosity=0.0,
        max_erosion_rate_sed=1e-2,
        active_layer_rate_sed=None,
        erodibility_br_cont=1e-10,
        erodibility_br_mar=1e-10,
        bedrock_composition=1.0,
        critical_flux_br=0.0,
        exponent_discharge=0.5,
        exponent_slope=1.0,
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
        wave_base : float (m)
            The wave base, below which weathering decreases exponentially.
        settling_velocity : float or array-like (m/time)
            The effective settling velocity for one or multiple lithologies.
        critical_flux_sed : float or array-like (m3/time)
            Critical sediment flux to start displace sediments in the stream
            power law.
        critical_thickness : float (m)
            Sediment thickness required for full entrainment.
        porosity : float or array-like (-)
            The porosity of the sediments at the time of deposition for one or
            multiple lithologies. When computing the active layer, this porosity
            is used unless the field 'sediment__porosity' is being tracked in
            the stratigraphy.
        max_erosion_rate_sed : float (m/time), optional
            The maximum erosion rate of the sediments. If None, all the sediments
            may be eroded in a single time step. The erosion rate defines the
            thickness of the active layer of the sediments if `active_layer_rate_sed`
            is None.
        active_layer_rate_sed : float (m/time), optional
            The rate of formation of the active layer for sediments, which is used
            to determine the composition of the transported sediments. By default,
            it is set by the maximum erosion rate of the sediments.
        erodibility_br_cont : float (m/time)
            The erodibility of the berock over the continental domain.
        erodibility_br_mar : float (m/time)
            The erodibility of the berock over the marine domain.
        bedrock_composition : float or array-like (-)
            The composition of the material is added to the StackedLayers from
            the bedrock.
        critical_flux_br : float or array-like (m3/time)
            Critical sediment flux to start erode the bedrock in the stream
            power law.
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
        self._flow_receivers = grid.at_node["flow__receiver_node"][..., np.newaxis]
        n_receivers = self._flow_receivers.shape[1]

        super().__init__(
            grid=grid,
            number_of_neighbors=n_receivers,
            diffusivity_cont=erodibility_sed_cont,
            diffusivity_mar=erodibility_sed_mar,
            wave_base=wave_base,
            critical_flux=critical_flux_sed,
            porosity=porosity,
            max_erosion_rate_sed=max_erosion_rate_sed,
            active_layer_rate_sed=active_layer_rate_sed,
            bedrock_composition=bedrock_composition,
            max_erosion_rate_br=0.0,
            active_layer_rate_br=0.0,
            exponent_discharge=exponent_discharge,
            exponent_slope=exponent_slope,
            ref_water_flux=ref_water_flux,
            fields_to_track=fields_to_track,
        )

        # Parameters
        self._settling_velocity_cont = convert_to_array(settling_velocity)
        self.critical_thickness = critical_thickness
        self._K_br_cont = convert_to_array(erodibility_br_cont)
        self._K_br_mar = convert_to_array(erodibility_br_mar)[np.newaxis]
        self._critical_flux_br = convert_to_array(critical_flux_br)

        # Fields for sediment fluxes
        n_nodes = grid.number_of_nodes
        n_receivers = self._flow_receivers.shape[1]
        n_sediments = self._stratigraphy.number_of_classes
        self._settling_velocity = np.zeros((n_nodes, n_sediments))
        self._K_br = np.zeros((n_nodes, 1, n_sediments))
        self._erosion_flux_sed = np.zeros((n_nodes, n_receivers, n_sediments))
        self._erosion_flux_br = np.zeros((n_nodes, n_receivers, n_sediments))
        self._ratio_excess_thickness = np.zeros((n_nodes, 1, 1))
        self._critical_rate_br = np.zeros((n_nodes, 1, n_sediments))
        self._ratio_critical_outflux_br = np.zeros((n_nodes, n_receivers, n_sediments))

    def _calculate_bedrock_diffusivity(self):
        """
        Calculates the diffusivity coefficient of the bedrock over the continental
        and marine domains.
        """
        self._K_br[self._bathymetry[:, 0] == 0.0] = self._K_br_cont
        self._K_br[self._bathymetry[:, 0] > 0.0, 0] = self._K_br_mar * np.exp(
            -self._bathymetry[self._bathymetry[:, 0] > 0.0] / self.wave_base
        )

    def _calculate_sediment_flux(self, dt):
        """
        Calculates the erosion flux of sediments for multiple lithologies.
        """
        core_nodes = self._grid.core_nodes
        cell_area = self._grid.cell_area_at_node[:, np.newaxis, np.newaxis]

        self._calculate_sediment_diffusivity()
        self._calculate_active_layer_composition(dt)

        self._erosion_flux_sed[:] = (
            self._K_sed
            * self._active_layer_composition
            * (self._water_flux * self._flow_proportions) ** self._m
            * self._slope**self._n
        )
        self._critical_rate[core_nodes] = self._critical_flux / cell_area[core_nodes]
        self._ratio_critical_outflux[:] = 0.0
        np.divide(
            self._erosion_flux_sed,
            self._critical_rate,
            out=self._ratio_critical_outflux,
            where=self._critical_rate != 0,
        )
        self._erosion_flux_sed[:] -= self._critical_rate * (
            1.0 - np.exp(-self._ratio_critical_outflux)
        )

        self._ratio_excess_thickness[self._grid.core_nodes, 0, 0] = np.exp(
            -self._stratigraphy.thickness / self.critical_thickness
        )
        self._erosion_flux_sed[:] *= 1.0 - self._ratio_excess_thickness
        self._erosion_flux_sed[:] *= cell_area

    def _calculate_bedrock_flux(self):
        """
        Calculates the erosion flux of the bedrock for multiple lithologies.
        """
        core_nodes = self._grid.core_nodes
        cell_area = self._grid.cell_area_at_node[:, np.newaxis, np.newaxis]

        self._calculate_bedrock_diffusivity()

        self._erosion_flux_br[:] = (
            self._K_br
            * self._bedrock_composition
            * (self._water_flux * self._flow_proportions) ** self._m
            * self._slope**self._n
        )
        self._critical_rate_br[self._grid.core_nodes] = (
            self._critical_flux_br / cell_area[core_nodes]
        )
        self._ratio_critical_outflux_br[:] = 0.0
        np.divide(
            self._erosion_flux_br,
            self._critical_rate_br,
            out=self._ratio_critical_outflux_br,
            where=self._critical_rate_br != 0,
        )
        self._erosion_flux_br[:] -= self._critical_rate_br * (
            1.0 - np.exp(-self._ratio_critical_outflux_br)
        )

        self._erosion_flux_br[:] *= self._ratio_excess_thickness
        self._erosion_flux_br[:] *= cell_area

    def _calculate_settling_velocity(self):
        """
        Calculates the settling velocity over the continental and marine domains.
        """
        self._settling_velocity[:] = self._settling_velocity_cont
        self._settling_velocity[self._bathymetry[:, 0] > 0.0, 0] *= np.exp(
            self._bathymetry[self._bathymetry[:, 0] > 0.0, 0] / self.wave_base
        )

    def run_one_step(self, dt, update_compatible=False, update=False):
        """Run the router for one timestep, dt.

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

        self._normalize_water_flux()

        self._calculate_sediment_flux(dt)
        if self._max_erosion_rate_sed != self._active_layer_rate_sed:
            self._calculate_active_layer(self._max_erosion_rate_sed * dt, 0.0)
        self._max_sediment_outflux[:] = cell_area * self._active_layer[:, 0] / dt
        self._calculate_bedrock_flux()
        self._calculate_settling_velocity()

        self._sediment_influx[:] = self._sediment_input
        self._sediment_outflux[:] = 0.0
        calculate_sediment_fluxes(
            self._node_order,
            cell_area[:, 0],
            self._flow_receivers[..., 0],
            self._water_flux[:, 0, 0],
            self._flow_proportions[..., 0],
            self._sediment_influx,
            self._sediment_outflux,
            self._settling_velocity,
            self._erosion_flux_sed,
            self._erosion_flux_br,
            self._max_sediment_outflux,
            self._porosity,
        )

        self._apply_fluxes(dt, update_compatible, update)
