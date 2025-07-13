"""Base components"""

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

from ..utils import convert_to_array, format_fields_to_track


################################################################################
# Base components


class _BaseHandler(Component):
    """Base class to handle sediments in stacked layers."""

    _name = "_BaseHandler"

    _unit_agnostic = True

    def __init__(
        self,
        grid,
        fields_to_track=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        grid : ModelGrid
            A grid.
        fields_to_track : str or array-like, optional
            The name of the fields at grid nodes to add to the StackedLayers at
            each iteration.
        """
        super().__init__(grid)

        # Parameters
        self._fields_to_track = format_fields_to_track(fields_to_track)

        # Physical fields
        self._stratigraphy = grid.stacked_layers
        self._time = 0.0

        # Fields for sediment fluxes
        self._sediment_thickness = np.zeros(
            (grid.number_of_nodes, self._stratigraphy.number_of_classes)
        )

    def _update_stratigraphy(
        self, dt, update_compatible=False, update=False, at_bottom=False
    ):
        """
        Updates the stratigraphy based on the sediment changes.
        """
        core_nodes = self._grid.core_nodes

        self._time += dt

        fields_to_track = {
            field: self._grid.at_node[field][core_nodes]
            for field in self._fields_to_track
        }
        # _sediment_thickness also includes bedrock erosion, which only affects
        # the topography (sediment thickness in _stratigraphy won't become negative
        # if more material is to be removed than what is available, and the
        # bedrock below that is considered infinite)
        self._stratigraphy.add(
            self._sediment_thickness[core_nodes],
            at_bottom=at_bottom,
            update=update,
            update_compatible=update_compatible,
            time=self._time,
            **fields_to_track,
        )


class _BaseMover(_BaseHandler):
    """Base class to move sediments in continental and marine domains."""

    _name = "_BaseMover"

    _unit_agnostic = True

    _info = {
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
        fields_to_track=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        grid : ModelGrid
            A grid.
        fields_to_track : str or array-like, optional
            The name of the fields at grid nodes to add to the StackedLayers at
            each iteration.
        """
        super().__init__(grid=grid, fields_to_track=fields_to_track, **kwargs)

        # Physical fields
        self._topography = grid.at_node["topographic__elevation"]

    def _update_physical_fields(self, dt, update_compatible=False, update=False):
        """
        Applies the sediment fluxes to the topography and stratigraphy.
        """
        core_nodes = self._grid.core_nodes

        self._update_stratigraphy(dt, update_compatible, update)
        self._topography[core_nodes] += np.sum(
            self._sediment_thickness[core_nodes], axis=1
        )


################################################################################
# Base components for diffusion models


class _BaseDiffuser(_BaseMover):
    """Base class to diffuse sediments in continental and marine domains."""

    _name = "_BaseDiffuser"

    _unit_agnostic = True

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
        wave_base=20.0,
        porosity=0.0,
        max_erosion_rate_sed=0.01,
        active_layer_rate_sed=None,
        bedrock_composition=1.0,
        max_erosion_rate_br=0.01,
        active_layer_rate_br=None,
        exponent_slope=1.0,
        fields_to_track=None,
        **kwargs,
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
        super().__init__(grid=grid, fields_to_track=fields_to_track, **kwargs)

        # Parameters
        self._K_cont = convert_to_array(diffusivity_cont)
        self._K_mar = convert_to_array(diffusivity_mar)
        self.wave_base = wave_base
        self._porosity = convert_to_array(porosity)
        self._max_erosion_rate_sed = (
            np.inf if max_erosion_rate_sed is None else max_erosion_rate_sed
        )
        self._active_layer_rate_sed = (
            self._max_erosion_rate_sed
            if active_layer_rate_sed is None
            else active_layer_rate_sed
        )
        self._bedrock_composition = convert_to_array(bedrock_composition)
        self.max_erosion_rate_br = max_erosion_rate_br
        self._active_layer_rate_br = (
            self.max_erosion_rate_br
            if active_layer_rate_br is None
            else active_layer_rate_br
        )
        self._n = exponent_slope

        # Physical fields
        self._bathymetry = grid.at_node["bathymetric__depth"][:, np.newaxis]

        # Fields for sediment fluxes
        n_nodes = grid.number_of_nodes
        n_sediments = self._stratigraphy.number_of_classes
        self._K_sed = np.zeros((n_nodes, 1, n_sediments))
        self._active_layer = np.zeros((n_nodes, 1, n_sediments))
        self._active_layer_thickness = np.zeros((n_nodes, 1, 1))
        self._active_layer_composition = np.zeros((n_nodes, 1, n_sediments))
        self._total_layer_thickness = np.zeros((n_nodes, 1, 1))

    def _calculate_sediment_diffusivity(self):
        """
        Calculates the diffusivity coefficient of the sediments over the continental
        and marine domains.
        """
        self._K_sed[self._bathymetry[:, 0] == 0.0] = self._K_cont
        self._K_sed[self._bathymetry[:, 0] > 0.0, 0] = self._K_mar * np.exp(
            -self._bathymetry[self._bathymetry[:, 0] > 0.0] / self.wave_base
        )

    def _calculate_active_layer(self, max_thickness_sed, max_thickness_br):
        """
        Calculates the active layer based on the sediments and the bedrock.
        """
        porosity = (
            "sediment__porosity"
            if "sediment__porosity" in self._stratigraphy._attrs
            else self._porosity
        )
        self._active_layer[self._grid.core_nodes, 0] = (
            self._stratigraphy.get_active_layer(max_thickness_sed, porosity)
        )
        if max_thickness_br > 0.0:
            self._active_layer_thickness[:] = np.sum(
                self._active_layer, axis=2, keepdims=True
            )
            # TODO: This seems to really slow things down
            self._total_layer_thickness[self._grid.core_nodes, 0, 0] = (
                self._stratigraphy.get_thickness(porosity)
            )
            np.add(
                self._active_layer,
                self._bedrock_composition
                * (
                    (1.0 - self._porosity) * max_thickness_br
                    - self._active_layer_thickness
                ),
                out=self._active_layer,
                where=(
                    np.abs(self._active_layer_thickness - self._total_layer_thickness)
                    < 1e-10
                )
                & (
                    self._active_layer_thickness
                    < (1.0 - self._porosity) * max_thickness_br
                ),
            )

    def _calculate_active_layer_composition(self, dt):
        """
        Calculates the composition of the active layer based on the sediments
        and the bedrock.
        """
        self._calculate_active_layer(
            self._active_layer_rate_sed * dt, self._active_layer_rate_br * dt
        )

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


class _BaseStreamPower(_BaseDiffuser):
    """Base class to move sediments in continental and marine domains using the
    stream power law.
    """

    _name = "_BaseStreamPower"

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
        diffusivity_cont=0.01,
        diffusivity_mar=0.001,
        wave_base=20.0,
        critical_flux=0.0,
        porosity=0.0,
        max_erosion_rate_sed=0.01,
        active_layer_rate_sed=None,
        bedrock_composition=1.0,
        max_erosion_rate_br=0.01,
        active_layer_rate_br=None,
        exponent_discharge=1.0,
        exponent_slope=1.0,
        ref_water_flux=None,
        fields_to_track=None,
        **kwargs,
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
            thickness of the active layer of the sediments if `active_layer_rate`
            is None.
        max_erosion_rate_br : float (m/time)
            The maximum erosion rate of the bedrock. The erosion rate defines the
            thickness of the active layer of the bedrock if `active_layer_rate`
            is None.
        active_layer_rate_sed : float (m/time), optional
            The rate of formation of the active layer for sediments, which is used
            to determine the composition of the transported sediments. By default,
            it is set by the maximum erosion rate of the sediments.
        active_layer_rate_br : float (m/time), optional
            The rate of formation of the active layer for the bedrock, which is
            used to determine the composition of the transported sediments. By
            default, it is set by the maximum erosion rate of the bedrock.
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
        self._critical_flux = convert_to_array(critical_flux)
        self._m = exponent_discharge
        self.ref_water_flux = ref_water_flux

        # Fields for stream power
        self._node_order = grid.at_node["flow__upstream_node_order"]
        self._flow_receivers = grid.at_node["flow__receiver_node"][..., np.newaxis]
        self._link_to_receiver = grid.at_node["flow__link_to_receiver_node"][
            ..., np.newaxis
        ]
        self._slope = grid.at_node["topographic__steepest_slope"][..., np.newaxis]
        if self._flow_receivers.ndim == 2:
            self._flow_receivers = self._flow_receivers[..., np.newaxis]
            self._link_to_receiver = self._link_to_receiver[..., np.newaxis]
            self._slope = self._slope[..., np.newaxis]
        n_nodes = grid.number_of_nodes
        n_receivers = self._flow_receivers.shape[1]
        if "flow__receiver_proportions" in grid.at_node:
            self._flow_proportions = grid.at_node["flow__receiver_proportions"][
                ..., np.newaxis
            ]
        else:
            self._flow_proportions = np.ones((n_nodes, n_receivers, 1))
        self._water_flux = grid.at_node["surface_water__discharge"][
            :, np.newaxis, np.newaxis
        ]

        # Fields for sediment fluxes
        n_sediments = self._stratigraphy.number_of_classes
        if "sediment__unit_flux_in" in grid.at_node:
            self._sediment_input = grid.at_node["sediment__unit_flux_in"]
            if self._sediment_input.ndim == 1:
                self._sediment_input = self._sediment_input[:, np.newaxis]
        else:
            self._sediment_input = np.zeros((n_nodes, n_sediments))
        self._critical_rate = np.zeros((n_nodes, 1, n_sediments))
        self._ratio_critical_outflux = np.zeros((n_nodes, n_receivers, n_sediments))

    def _normalize_water_flux(self):
        """
        Normalizes the water flux if needed.
        """
        if self.ref_water_flux == "max":
            self._water_flux[:] /= np.max(self._water_flux)
        elif isinstance(self.ref_water_flux, (int, float)):
            self._water_flux[:] /= self.ref_water_flux


################################################################################
# Base components for routing models


class _BaseRouter(_BaseMover):
    """Base class to route sediments in continental and marine domains."""

    _name = "_BaseRouter"

    _unit_agnostic = True

    _info = {
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
        number_of_neighbors=1,
        porosity=0.0,
        fields_to_track=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        grid : ModelGrid
            A grid.
        number_of_neighbors : int, optional
            The number of neighbors around a cell to consider when computing
            sediment displacement.
        porosity : float or array-like (-)
            The porosity of the sediments at the time of deposition for one or
            multiple lithologies.
        fields_to_track : str or array-like, optional
            The name of the fields at grid nodes to add to the StackedLayers at
            each iteration.
        """
        super().__init__(grid=grid, fields_to_track=fields_to_track, **kwargs)

        # Parameters
        self._porosity = convert_to_array(porosity)

        # Fields for sediment fluxes
        n_nodes = grid.number_of_nodes
        n_sediments = self._stratigraphy.number_of_classes
        self._sediment_influx = np.zeros((n_nodes, n_sediments))
        self._sediment_outflux = np.zeros((n_nodes, number_of_neighbors, n_sediments))
        self._max_sediment_outflux = np.zeros((n_nodes, n_sediments))

    def _apply_fluxes(self, dt, update_compatible=False, update=False):
        """
        Applies the sediment fluxes to the topography and stratigraphy.
        """
        core_nodes = self._grid.core_nodes
        cell_area = self._grid.cell_area_at_node[:, np.newaxis]

        self._sediment_thickness[core_nodes] = (
            (
                self._sediment_influx[core_nodes]
                - np.sum(self._sediment_outflux[core_nodes], axis=1)
            )
            * dt
            / (1.0 - self._porosity)
            / cell_area[core_nodes]
        )
        self._update_physical_fields(dt, update_compatible, update)
