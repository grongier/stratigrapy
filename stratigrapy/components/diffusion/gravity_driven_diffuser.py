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

from ...utils import convert_to_array
from .._base import _BaseDiffuser
from ..stream_power.cfuncs import calculate_flux_limiter


################################################################################
# Component


class GravityDrivenDiffuser(_BaseDiffuser):
    """Gravity-driven diffusion of a Landlab field in continental and marine domains
    based on a finite-volume scheme. The diffusion can be linear (the default) or
    non-linear (when the critical-anlge parameters are defined) following Roering
    et al. (1999).

    This component is not a re-implementation of Gervais (2004), but it is
    heavily based on it. Instead of a implicit or semi-implicit approach
    starting from updating the topography, it uses a fully explicit approach
    that ends with updating the topography.

    References
    ----------
    Gervais, V. (2004)
        Étude et Simulation d'un Modèle Stratigraphique Multi-Lithologique sous Contrainte de Taux d'Érosion Maximal
        https://theses.hal.science/tel-01445562/
    Roering, J. J., Kirchner, J. W., & Dietrich, W. E. (1999)
        Evidence for nonlinear, diffusive sediment transport on hillslopes and implications for landscape morphology
        https://doi.org/10.1029/1998WR900090
    """

    _name = "GravityDrivenDiffuser"

    def __init__(
        self,
        grid,
        diffusivity_cont=0.01,
        diffusivity_mar=0.001,
        critical_angle_cont=None,
        critical_angle_mar=None,
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
        diffusivity_cont : float or array-like (m2/time)
            The diffusivity of the sediments over the continental domain for one
            or multiple lithologies.
        diffusivity_mar : float or array-like (m2/time)
            The diffusivity of the sediments over the marine domain for one or
            multiple lithologies.
        critical_angle_cont : float or array-like (degree), optional
            The critical angle in the continental domain for one or multiple
            lithologies above which sediments move downslope by mass wasting.
        critical_angle_mar : float or array-like (degree), optional
            The critical angle in the continental domain for one or multiple
             ithologies above which sediments move downslope by mass wasting.
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
        super().__init__(
            grid,
            diffusivity_cont,
            diffusivity_mar,
            wave_base,
            porosity,
            max_erosion_rate_sed,
            active_layer_rate_sed,
            bedrock_composition,
            max_erosion_rate_br,
            active_layer_rate_br,
            exponent_slope,
            fields_to_track,
        )

        # Parameters
        if critical_angle_cont is None:
            self._critical_slope_cont = None
        else:
            self._critical_slope_cont = convert_to_array(critical_angle_cont)
            if np.all(self._critical_slope_cont == 0.0):
                self._critical_slope_cont = None
            else:
                self._critical_slope_cont = np.tan(
                    np.deg2rad(self._critical_slope_cont)
                )
        if critical_angle_mar is None:
            self._critical_slope_mar = None
        else:
            self._critical_slope_mar = convert_to_array(critical_angle_mar)
            if np.all(self._critical_slope_mar == 0.0):
                self._critical_slope_mar = None
            else:
                self._critical_slope_mar = np.tan(np.deg2rad(self._critical_slope_mar))

        # Physical fields
        n_links = grid.number_of_links
        self._bathymetry_at_links = np.zeros(n_links)

        # Fields for sediment fluxes
        n_nodes = grid.number_of_nodes
        n_sediments = self._stratigraphy.number_of_classes
        self._node_order = np.zeros(n_nodes, dtype=int)
        self._K_sed_at_links = np.zeros((n_links, n_sediments))
        self._slope_at_links = np.zeros((n_links, 1))
        if (
            self._critical_slope_cont is not None
            and self._critical_slope_mar is not None
        ):
            self._thres_slope_at_links = np.zeros((n_links, n_sediments))
            self._critical_slope_at_links = np.zeros((n_links, n_sediments))
        self._active_layer_composition_at_links = np.zeros((n_links, n_sediments))
        self._sediment_flux_at_links = np.zeros((n_links, n_sediments))
        self._sediment_flux = np.zeros((n_links, n_sediments))
        self._sediment_rate = np.zeros((n_nodes, n_sediments))
        self._max_sediment_flux = np.zeros((n_nodes, n_sediments))
        self._sediment_input = np.zeros((n_nodes, n_sediments))
        self._flux_limiter = np.zeros((n_nodes, n_sediments))
        self._flux_limiter_at_links = np.zeros((n_links, n_sediments))

    def _adjust_sediment_diffusivity(self):
        """
        Ajusts the sediment diffusivity based on the slope over the continental
        and marine domains.
        """
        self._grid.map_mean_of_link_nodes_to_link(
            self._bathymetry[:, 0], out=self._bathymetry_at_links
        )

        self._critical_slope_at_links[self._bathymetry_at_links == 0.0] = (
            self._critical_slope_cont
        )
        self._critical_slope_at_links[self._bathymetry_at_links > 0.0] = (
            self._critical_slope_mar
        )

        self._thres_slope_at_links[:] = np.where(
            np.abs(self._slope_at_links) >= self._critical_slope_at_links,
            self._critical_slope_at_links - 1e-12,
            self._slope_at_links,
        )

        self._K_sed_at_links /= (
            1.0 - (self._thres_slope_at_links / self._critical_slope_at_links) ** 2
        )

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

        if (
            self._critical_slope_cont is not None
            and self._critical_slope_mar is not None
        ):
            self._adjust_sediment_diffusivity()

        self._sediment_flux_at_links[:] = (
            -self._K_sed_at_links
            * np.sign(self._slope_at_links)
            * self._active_layer_composition_at_links
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

        self._node_order[:] = np.argsort(self._topography)
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

        self._calculate_sediment_flux(dt)
        self._threshold_sediment_flux(dt)
        self._grid.calc_mult_flux_div_at_node(
            self._sediment_flux_at_links, out=self._sediment_rate
        )

        self._sediment_thickness[core_nodes] = (
            -self._sediment_rate[core_nodes] * dt / (1.0 - self._porosity)
        )

        self._update_physical_fields(dt, update_compatible, update)
