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

from .._base import _BaseStreamPower
from ...utils import convert_to_array
from .cfuncs import calculate_sediment_fluxes


################################################################################
# Component


class WaterDrivenDisplacer(_BaseStreamPower):
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

    def __init__(
        self,
        grid,
        erodibility_sed_cont=1e-10,
        erodibility_sed_mar=1e-10,
        wave_base=20.0,
        settling_velocity=1.0,
        critical_flux_sed=0.0,
        porosity=0.0,
        max_erosion_rate_sed=1e-2,
        active_layer_rate=None,
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
        porosity : float or array-like (-)
            The porosity of the sediments at the time of deposition for one or
            multiple lithologies. When computing the active layer, this porosity
            is used unless the field 'sediment__porosity' is being tracked in
            the stratigraphy.
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
        super().__init__(
            grid,
            erodibility_sed_cont,
            erodibility_sed_mar,
            wave_base,
            critical_flux_sed,
            porosity,
            max_erosion_rate_sed,
            active_layer_rate,
            bedrock_composition,
            exponent_discharge,
            exponent_slope,
            ref_water_flux,
            fields_to_track,
        )

        # Parameters
        self.settling_velocity = convert_to_array(settling_velocity)
        self.K_br_cont = convert_to_array(erodibility_br_cont)
        self.K_br_mar = convert_to_array(erodibility_br_mar)[np.newaxis]
        self.critical_flux_br = convert_to_array(critical_flux_br)
        n_nodes = grid.number_of_nodes
        n_sediments = self._K_sed.shape[2]
        self._K_br = np.zeros((n_nodes, 1, n_sediments))

        # Fields for sediment fluxes
        n_receivers = self._flow_receivers.shape[1]
        self._erosion_flux_sed = np.zeros((n_nodes, n_receivers, n_sediments))
        self._erosion_flux_br = np.zeros((n_nodes, n_receivers, n_sediments))
        self._ratio_critical_outflux_br = np.zeros((n_nodes, n_receivers, n_sediments))

    def _calculate_bedrock_diffusivity(self):
        """
        Calculates the diffusivity coefficient of the bedrock over the continental
        and marine domains.
        """
        self._K_br[self._bathymetry[:, 0] == 0.0] = self.K_br_cont
        self._K_br[self._bathymetry[:, 0] > 0.0, 0] = self.K_br_mar * np.exp(
            -self._bathymetry[self._bathymetry[:, 0] > 0.0] / self.wave_base
        )
        self._K_br[self._grid.core_nodes][self._stratigraphy.thickness > 0.0] = 0.0

    def _calculate_sediment_flux(self, dt):
        """
        Calculates the erosion flux of sediments for multiple lithologies.
        """
        cell_area = self._grid.cell_area_at_node[:, np.newaxis, np.newaxis]

        porosity = (
            "sediment__porosity"
            if "sediment__porosity" in self._stratigraphy._attrs
            else self.porosity
        )
        self._active_layer_composition[self._grid.core_nodes, 0] = (
            self._stratigraphy.get_active_composition(
                self.active_layer_rate * dt, porosity
            )
        )

        self._erosion_flux_sed[:] = (
            self._K_sed
            * cell_area
            * self._active_layer_composition
            * (self._water_flux * self._flow_proportions) ** self.m
            * self._slope**self.n
        )
        np.divide(
            self._erosion_flux_sed,
            self.critical_flux,
            out=self._ratio_critical_outflux,
            where=self.critical_flux != 0,
        )
        self._erosion_flux_sed[:] -= self.critical_flux * (
            1.0 - np.exp(-self._ratio_critical_outflux)
        )

    def _calculate_bedrock_flux(self):
        """
        Calculates the erosion flux of the bedrock for multiple lithologies.
        """
        cell_area = self._grid.cell_area_at_node[:, np.newaxis, np.newaxis]

        self._erosion_flux_br[:] = (
            self._K_br
            * cell_area
            * self.bedrock_composition
            * (self._water_flux * self._flow_proportions) ** self.m
            * self._slope**self.n
        )
        np.divide(
            self._erosion_flux_br,
            self.critical_flux_br,
            out=self._ratio_critical_outflux_br,
            where=self.critical_flux_br != 0,
        )
        self._erosion_flux_br[:] -= self.critical_flux_br * (
            1.0 - np.exp(-self._ratio_critical_outflux_br)
        )

    def run_one_step(self, dt, update_compatible=False, update=False):
        """Run the displacer for one timestep, dt.

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
        core_nodes = self._grid.core_nodes
        cell_area = self._grid.cell_area_at_node[:, np.newaxis]

        self._normalize_water_flux()

        self._calculate_sediment_diffusivity()
        self._calculate_bedrock_diffusivity()
        self._calculate_sediment_flux(dt)
        self._calculate_bedrock_flux()
        porosity = (
            "sediment__porosity"
            if "sediment__porosity" in self._stratigraphy._attrs
            else self.porosity
        )
        self._max_sediment_outflux[core_nodes] = cell_area[core_nodes] * np.minimum(
            (1.0 - self.porosity) * self.max_erosion_rate,
            self._stratigraphy.get_class_thickness(porosity) / dt,
        )
        # self._max_bedrock_outflux[core_nodes] = (1. - self.porosity)*cell_area[core_nodes]*self.max_erosion_rate_br

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
            self.settling_velocity,
            self._erosion_flux_sed,
            self._erosion_flux_br,
            self._max_sediment_outflux,
            dt,
        )

        self._apply_fluxes(dt, update_compatible, update)
