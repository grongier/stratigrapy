"""Water-driven diffuser"""

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
from .cfuncs import calculate_sediment_influx


################################################################################
# Component


class WaterDrivenDiffuser(_BaseStreamPower):
    """Water-driven diffusion of a Landlab field in continental and marine domains.

    References
    ----------
    Granjeon, D. (1996)
        Modélisation stratigraphique déterministe: Conception et applications d'un modèle diffusif 3D multilithologique
        https://theses.hal.science/tel-00648827
    Granjeon, D., & Joseph, P. (1999)
        Concepts and applications of a 3-D multiple lithology, diffusive model in stratigraphic modeling
        https://doi.org/10.2110/pec.99.62.0197
    Shobe, C. M., Tucker, G. E., & Barnhart, K. R. (2017)
        The SPACE 1.0 model: A Landlab component for 2-D calculation of sediment transport, bedrock erosion, and landscape evolution
        https://doi.org/10.5194/gmd-10-4577-2017
    """

    _name = "WaterDrivenDiffuser"

    def __init__(
        self,
        grid,
        transportability_cont=1e-5,
        transportability_mar=1e-6,
        wave_base=20.0,
        critical_flux=0.0,
        porosity=0.0,
        max_erosion_rate_sed=1e-2,
        max_erosion_rate_br=1e-2,
        active_layer_rate=None,
        bedrock_composition=1.0,
        exponent_discharge=1.0,
        exponent_slope=1.0,
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
        critical_flux : float or array-like (m3/time)
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
        max_erosion_rate_br : float (m/time)
            The maximum erosion rate of the bedrock. The erosion rate defines the
            thickness of the active layer if `active_layer_rate` is None.
        active_layer_rate : float or array-like (m/time), optional
            The rate of formation of the active layer, which is used to determine
            the composition of the transported sediments. By default, it is set
            by the maximum of `max_erosion_rate_sed` and `max_erosion_rate_br`.
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
        super().__init__(
            grid,
            transportability_cont,
            transportability_mar,
            wave_base,
            critical_flux,
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
        self.max_erosion_rate_br = max_erosion_rate_br
        self.active_layer_rate = (
            max(max_erosion_rate_sed, max_erosion_rate_br)
            if active_layer_rate is None
            else active_layer_rate
        )

        # Fields for sediment fluxes
        n_nodes = grid.number_of_nodes
        n_sediments = self._K_sed.shape[2]
        self._max_bedrock_outflux = np.zeros((n_nodes, n_sediments))
        self._active_layer = np.zeros((n_nodes, 1, n_sediments))
        self._active_layer_thickness = np.zeros((n_nodes, 1, 1))

    def _calculate_sediment_outflux(self, dt):
        """
        Calculates the sediment outflux for multiple lithologies.
        """
        cell_area = self._grid.cell_area_at_node[:, np.newaxis, np.newaxis]

        porosity = (
            "sediment__porosity"
            if "sediment__porosity" in self._stratigraphy._attrs
            else self.porosity
        )
        self._active_layer[self._grid.core_nodes, 0] = (
            self._stratigraphy.get_active_layer(self.active_layer_rate * dt, porosity)
        )
        self._active_layer_thickness[:] = np.sum(
            self._active_layer, axis=2, keepdims=True
        )
        self._active_layer += self.bedrock_composition * (
            (1.0 - self.porosity) * self.active_layer_rate * dt
            - self._active_layer_thickness
        )
        self._active_layer[self._active_layer < 0.0] = 0.0
        self._active_layer_thickness[:] = np.sum(
            self._active_layer, axis=2, keepdims=True
        )
        self._active_layer_composition[:] = 0.0
        np.divide(
            self._active_layer,
            self._active_layer_thickness,
            out=self._active_layer_composition,
            where=self._active_layer_thickness > 0.0,
        )

        self._sediment_outflux[:] = (
            self._K_sed
            * cell_area
            * self._active_layer_composition
            * (self._water_flux * self._flow_proportions) ** self.m
            * self._slope**self.n
        )
        np.divide(
            self._sediment_outflux,
            self.critical_flux,
            out=self._ratio_critical_outflux,
            where=self.critical_flux != 0,
        )
        self._sediment_outflux[:] -= self.critical_flux * (
            1.0 - np.exp(-self._ratio_critical_outflux)
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
        core_nodes = self._grid.core_nodes
        cell_area = self._grid.cell_area_at_node[:, np.newaxis]

        self._normalize_water_flux()

        self._calculate_sediment_diffusivity()
        self._calculate_sediment_outflux(dt)
        porosity = (
            "sediment__porosity"
            if "sediment__porosity" in self._stratigraphy._attrs
            else self.porosity
        )
        self._max_sediment_outflux[self._grid.core_nodes] = cell_area[
            self._grid.core_nodes
        ] * np.minimum(
            (1.0 - self.porosity) * self.max_erosion_rate,
            self._stratigraphy.get_class_thickness(porosity) / dt,
        )
        self._max_bedrock_outflux[self._grid.core_nodes] = (
            (1.0 - self.porosity) * cell_area[core_nodes] * self.max_erosion_rate_br
        )

        self._sediment_influx[:] = self._sediment_input
        calculate_sediment_influx(
            self._node_order,
            self._flow_receivers[..., 0],
            self._sediment_influx,
            self._sediment_outflux,
            self._max_sediment_outflux,
            self._max_bedrock_outflux,
            dt,
        )

        self._apply_fluxes(dt, update_compatible, update)
