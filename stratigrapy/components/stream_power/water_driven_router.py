"""Water-driven router"""

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
from .cfuncs import calculate_sediment_influx


################################################################################
# Component


class WaterDrivenRouter(_BaseRouter, _BaseStreamPower):
    """Water-driven diffusion of a Landlab field in continental and marine domains
    based on a routing scheme.

    References
    ----------
    Granjeon, D. (1996)
        Modélisation stratigraphique déterministe: Conception et applications d'un modèle diffusif 3D multilithologique
        https://theses.hal.science/tel-00648827
    Shobe, C. M., Tucker, G. E., & Barnhart, K. R. (2017)
        The SPACE 1.0 model: A Landlab component for 2-D calculation of sediment transport, bedrock erosion, and landscape evolution
        https://doi.org/10.5194/gmd-10-4577-2017
    """

    _name = "WaterDrivenRouter"

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
        transportability_cont=1e-5,
        transportability_mar=1e-6,
        wave_base=20.0,
        critical_flux=0.0,
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
            critical_flux=critical_flux,
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

    def _calculate_sediment_outflux(self, dt):
        """
        Calculates the sediment outflux for multiple lithologies.
        """
        core_nodes = self._grid.core_nodes
        cell_area = self._grid.cell_area_at_node[:, np.newaxis, np.newaxis]

        self._calculate_sediment_diffusivity()
        self._calculate_active_layer_composition(dt)

        self._sediment_outflux[:] = (
            self._K_sed
            * self._active_layer_composition
            * (self._water_flux * self._flow_proportions) ** self._m
            * self._slope**self._n
        )
        self._critical_rate[core_nodes] = self._critical_flux / cell_area[core_nodes]
        self._ratio_critical_outflux[:] = 0.0
        np.divide(
            self._sediment_outflux,
            self._critical_rate,
            out=self._ratio_critical_outflux,
            where=self._critical_rate != 0,
        )
        self._sediment_outflux[:] -= self._critical_rate * (
            1.0 - np.exp(-self._ratio_critical_outflux)
        )
        self._sediment_outflux[:] *= cell_area

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

        # Here we merge fluxes from the sediments and the bedrock together,
        # assuming that weathered bedrock is perfectly equivalent to sediments,
        # including in terms of porosity.
        self._calculate_sediment_outflux(dt)
        if (
            self._max_erosion_rate_sed != self._active_layer_rate_sed
            or self.max_erosion_rate_br != self._active_layer_rate_br
        ):
            self._calculate_active_layer(
                self._max_erosion_rate_sed * dt, self.max_erosion_rate_br * dt
            )
        self._max_sediment_outflux[:] = cell_area * self._active_layer[:, 0] / dt

        self._sediment_influx[:] = self._sediment_input
        calculate_sediment_influx(
            self._node_order,
            self._flow_receivers[..., 0],
            self._sediment_influx,
            self._sediment_outflux,
            self._max_sediment_outflux,
        )

        self._apply_fluxes(dt, update_compatible, update)
