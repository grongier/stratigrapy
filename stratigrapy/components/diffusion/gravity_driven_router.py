"""Gravity-driven router"""

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
from .._base import _BaseRouter, _BaseDiffuser
from .cfuncs import calculate_sediment_fluxes


################################################################################
# Component


class GravityDrivenRouter(_BaseRouter, _BaseDiffuser):
    """Gravity-driven diffusion of a Landlab field in continental and marine domains
    based on a routing scheme. The diffusion can be linear (the default) or
    non-linear (when the critical-anlge parameters are defined) following Roering
    et al. (1999).

    This component updates the stratigraphy based on the difference between sediment
    influx and outflux, and not based on erosion and deposition like Carretier et
    al. (2016).

    References
    ----------
    Carretier, S., Martinod, P., Reich, M., & Godderis, Y. (2016)
        Modelling sediment clasts transport during landscape evolution
        https://doi.org/10.5194/esurf-4-237-2016
    Gervais, V. (2004)
        Étude et Simulation d'un Modèle Stratigraphique Multi-Lithologique sous Contrainte de Taux d'Érosion Maximal
        https://theses.hal.science/tel-01445562/
    """

    _name = "GravityDrivenRouter"

    _info = {
        "topographic__elevation": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Land surface topographic elevation",
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
        diffusivity_cont : float or array-like (m/time)
            The diffusivity of the sediments over the continental domain for one
            or multiple lithologies.
        diffusivity_mar : float or array-like (m/time)
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
        exponent_slope : float (-)
            The exponent for the slope.
        fields_to_track : str or array-like, optional
            The name of the fields at grid nodes to add to the StackedLayers at
            each iteration.
        """
        self._neighbors = grid.active_adjacent_nodes_at_node
        n_neighbors = self._neighbors.shape[1]

        super().__init__(
            grid=grid,
            number_of_neighbors=n_neighbors,
            diffusivity_cont=diffusivity_cont,
            diffusivity_mar=diffusivity_mar,
            wave_base=wave_base,
            porosity=porosity,
            max_erosion_rate_sed=max_erosion_rate_sed,
            active_layer_rate_sed=active_layer_rate_sed,
            bedrock_composition=bedrock_composition,
            max_erosion_rate_br=max_erosion_rate_br,
            active_layer_rate_br=active_layer_rate_br,
            exponent_slope=exponent_slope,
            fields_to_track=fields_to_track,
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

        # Field for the slopes
        n_nodes = grid.number_of_nodes
        self._link_lengths = grid.length_of_link
        self._links_to_neighbors = grid.links_at_node
        self._slope = np.zeros((n_nodes, n_neighbors, 1))

        # Fields for sediment fluxes
        n_sediments = self._stratigraphy.number_of_classes
        self._node_order = np.zeros(n_nodes, dtype=int)
        self._erosion_flux_sed = np.zeros((n_nodes, n_neighbors, n_sediments))
        if (
            self._critical_slope_cont is not None
            and self._critical_slope_mar is not None
        ):
            self._thres_slope = np.zeros((n_nodes, n_neighbors, n_sediments))
            self._critical_slope = np.zeros((n_nodes, 1, n_sediments))
        self._transferred_fraction = np.zeros((n_nodes, n_neighbors, n_sediments))
        self._flux_proportions = np.zeros((n_nodes, n_neighbors))
        self._sum_slopes = np.zeros((n_nodes, n_neighbors))

    def _calculate_slopes(self):
        """
        Calculates the slope between each node and its neighbors.
        """
        self._slope[..., 0] = (
            self._topography[:, np.newaxis] - self._topography[self._neighbors]
        ) / self._link_lengths[self._links_to_neighbors]
        self._slope[self._neighbors == -1] = 0.0
        self._slope[self._slope < 0.0] = 0.0

    def _calculate_sediment_flux(self, dt):
        """
        Calculates the erosion flux of sediments for multiple lithologies.
        """
        self._calculate_sediment_diffusivity()
        self._calculate_active_layer_composition(dt)
        self._calculate_slopes()

        self._erosion_flux_sed[:] = (
            self._K_sed
            * self._grid.cell_area_at_node[:, np.newaxis, np.newaxis]
            * self._active_layer_composition
            * self._slope**self._n
        )

    def _calculate_transferred_fraction(self):
        """
        Calculates the fraction of sediments transferred downstream for multiple
        lithologies.
        """
        self._critical_slope[self._bathymetry == 0.0] = self._critical_slope_cont
        self._critical_slope[self._bathymetry > 0.0] = self._critical_slope_mar

        self._thres_slope[:] = np.where(
            self._slope >= self._critical_slope,
            self._critical_slope - 1e-12,
            self._slope,
        )

        self._transferred_fraction[:] = (self._thres_slope / self._critical_slope) ** 2

    def _calculate_flux_proportions(self):
        """
        Calculates the flux proportions based on the slopes around a node.
        """
        self._sum_slopes[:] = np.sum(self._slope[..., 0], axis=1, keepdims=True)
        self._flux_proportions[:] = 0.0
        np.divide(
            self._slope[..., 0],
            self._sum_slopes,
            out=self._flux_proportions,
            where=self._sum_slopes > 0.0,
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

        # Here we merge fluxes from the sediments and the bedrock together,
        # assuming that weathered bedrock is perfectly equivalent to sediments,
        # including in terms of porosity.
        self._calculate_sediment_flux(dt)
        if (
            self._max_erosion_rate_sed != self._active_layer_rate_sed
            or self.max_erosion_rate_br != self._active_layer_rate_br
        ):
            self._calculate_active_layer(
                self._max_erosion_rate_sed * dt, self.max_erosion_rate_br * dt
            )
        self._max_sediment_outflux[:] = cell_area * self._active_layer[:, 0] / dt

        if (
            self._critical_slope_cont is not None
            and self._critical_slope_mar is not None
        ):
            self._calculate_transferred_fraction()
            self._calculate_flux_proportions()

        self._node_order[:] = np.argsort(self._topography)
        self._sediment_influx[:] = 0.0
        self._sediment_outflux[:] = 0.0
        calculate_sediment_fluxes(
            self._node_order,
            self._neighbors,
            self._flux_proportions,
            self._sediment_influx,
            self._sediment_outflux,
            self._transferred_fraction,
            self._erosion_flux_sed,
            self._max_sediment_outflux,
        )

        self._apply_fluxes(dt, update_compatible, update)
