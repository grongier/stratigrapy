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
from .cfuncs import calculate_flux_limiter


################################################################################
# Component


class WaterDrivenDiffuser(_BaseStreamPower):
    """Water-driven diffusion of a Landlab field in continental and marine domains
    based on a finite-volume scheme.

    This is a simple extension of the GravityDrivenDiffuser to account for water
    flow in a stream-power-like formula. Contrary to the WaterDrivenRouter,
    sediments can only be transported in a D4 scheme. This should be similar to
    Granjeon (1996), but details are lacking to actually ensure that it is the
    case. It needs a more thorough testing, as it looks quite unstable.

    References
    ----------
    Granjeon, D. (1996)
        Modélisation stratigraphique déterministe: Conception et applications d'un modèle diffusif 3D multilithologique
        https://theses.hal.science/tel-00648827
    Gervais, V. (2004)
        Étude et Simulation d'un Modèle Stratigraphique Multi-Lithologique sous Contrainte de Taux d'Érosion Maximal
        https://theses.hal.science/tel-01445562/
    """

    _name = "WaterDrivenDiffuser"

    def __init__(
        self,
        grid,
        transportability_cont=1e-5,
        transportability_mar=1e-6,
        wave_base=20.0,
        porosity=0.0,
        max_erosion_rate_sed=1e-2,
        active_layer_rate_sed=None,
        bedrock_composition=1.0,
        max_erosion_rate_br=1e-2,
        active_layer_rate_br=None,
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
        bedrock_composition : float or array-like (-)
            The composition of the material is added to the StackedLayers from
            the bedrock.
        max_erosion_rate_br : float (m/time)
            The maximum erosion rate of the bedrock. The erosion rate defines the
            thickness of the active layer of the bedrock if `active_layer_rate_br`
            is None.
        active_layer_rate_br : float (m/time), optional
            The rate of formation of the active layer for the bedrock, which is
            used to determine the composition of the transported sediments. By
            default, it is set by the maximum erosion rate of the bedrock.
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
            diffusivity_cont=transportability_cont,
            diffusivity_mar=transportability_mar,
            wave_base=wave_base,
            critical_flux=0.0,
            porosity=porosity,
            max_erosion_rate_sed=max_erosion_rate_sed,
            active_layer_rate_sed=active_layer_rate_sed,
            bedrock_composition=bedrock_composition,
            max_erosion_rate_br=max_erosion_rate_br,
            active_layer_rate_br=active_layer_rate_br,
            exponent_discharge=exponent_discharge,
            exponent_slope=exponent_slope,
            ref_water_flux=ref_water_flux,
            fields_to_track=fields_to_track,
        )

        # Fields for stream power
        n_links = grid.number_of_links
        self._receiver_link = grid.at_node["flow__link_to_receiver_node"]
        if self._receiver_link.ndim == 1:
            self._receiver_link = self._receiver_link[:, np.newaxis]
        self._water_flux_at_links = np.zeros((n_links, 1))

        # Fields for sediment fluxes
        n_nodes = grid.number_of_nodes
        n_sediments = self._stratigraphy.number_of_classes
        self._K_sed_at_links = np.zeros((n_links, n_sediments))
        self._slope_at_links = np.zeros((n_links, 1))
        self._active_layer_composition_at_links = np.zeros((n_links, n_sediments))
        self._sediment_flux_at_links = np.zeros((n_links, n_sediments))
        self._sediment_flux = np.zeros((n_links, n_sediments))
        self._sediment_rate = np.zeros((n_nodes, n_sediments))
        self._max_sediment_flux = np.zeros((n_nodes, n_sediments))
        self._flux_limiter = np.zeros((n_nodes, n_sediments))
        self._flux_limiter_at_links = np.zeros((n_links, n_sediments))

    def _map_water_flux_to_links(self):
        """
        Maps the water flux from the nodes to the links.
        """
        # With D8 MFD it looks like the first four receivers are the D4 ones
        self._water_flux_at_links[self._receiver_link[:, :4].ravel(), 0] = (
            self._flow_proportions[:, :4] * self._water_flux[:, :4]
        ).ravel()

    def _calculate_sediment_flux(self, dt):
        """
        Calculates the sediment fluxes for multiple lithologies.
        """
        self._slope_at_links[self._grid.active_links, 0] = self._grid.calc_grad_at_link(
            self._topography
        )[self._grid.active_links]
        self._calculate_active_layer_composition(dt)
        self._calculate_sediment_diffusivity()

        self._grid.map_mean_of_link_nodes_to_link(
            self._K_sed[:, 0], out=self._K_sed_at_links
        )
        self._grid.map_value_at_max_node_to_link(
            self._topography[:, np.newaxis],
            self._active_layer_composition[:, 0],
            out=self._active_layer_composition_at_links,
        )
        self._map_water_flux_to_links()

        self._sediment_flux_at_links[:] = (
            -self._K_sed_at_links
            * np.sign(self._slope_at_links)
            * self._active_layer_composition_at_links
            * self._water_flux_at_links**self._m
            * np.abs(self._slope_at_links) ** self._n
        )

    def _threshold_sediment_flux(self, dt):
        """
        Thresholds the sediment fluxes to not exceed a maximum erosion rate.
        """
        cell_area = self._grid.cell_area_at_node[:, np.newaxis]

        # Here we merge fluxes from the sediments and the bedrock together,
        # assuming that weathered bedrock is perfectly equivalent to sediments,
        # including in terms of porosity.
        if (
            self._max_erosion_rate_sed != self._active_layer_rate_sed
            or self.max_erosion_rate_br != self._active_layer_rate_br
        ):
            self._calculate_active_layer(
                self._max_erosion_rate_sed * dt, self.max_erosion_rate_br * dt
            )
        self._max_sediment_flux[:] = cell_area * self._active_layer[:, 0] / dt

        self._sediment_flux[self._grid.link_at_face] = (
            self._sediment_flux_at_links[self._grid.link_at_face]
            * self._grid.length_of_face[:, np.newaxis]
        )
        calculate_flux_limiter(
            self._node_order,
            self._grid.active_adjacent_nodes_at_node,
            self._grid.links_at_node,
            self._grid.link_dirs_at_node,
            self._sediment_flux,
            self._max_sediment_flux,
            self._sediment_input,
            self._flux_limiter,
        )
        self._grid.map_value_at_max_node_to_link(
            self._topography[:, np.newaxis],
            self._flux_limiter,
            out=self._flux_limiter_at_links,
        )

        self._sediment_flux_at_links *= self._flux_limiter_at_links

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

        self._calculate_sediment_flux(dt)
        self._threshold_sediment_flux(dt)
        self._grid.calc_mult_flux_div_at_node(
            self._sediment_flux_at_links, out=self._sediment_rate
        )

        self._sediment_thickness[core_nodes] = (
            (
                self._sediment_input[core_nodes] / cell_area[core_nodes]
                - self._sediment_rate[core_nodes]
            )
            * dt
            / (1.0 - self._porosity)
        )

        self._update_physical_fields(dt, update_compatible, update)
