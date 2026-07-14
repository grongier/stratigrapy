"""Base components"""

# MIT License

# Copyright (c) 2025-2026 Guillaume Rongier

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
        substeps=None,
        substep_fraction=0.3,
        max_substeps=1000,
        min_slope=1e-7,
        fields_to_track=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        grid : ModelGrid
            A grid.
        substeps : None, int, or 'adaptive', optional
            Controls how the imposed time step is subdivided when running the component:
                - If None, the time step is used as is (no subdivision).
                - If an integer, the time step is split into that fixed number of
                  equal substeps, with the topography (but not the flow) updated
                  between substeps.
                - If 'adaptive', the time step is adaptively subdivided to keep
                  the explicit scheme stable, following the approach of CHILD:
                  a Courant criterion for hillslope diffusion and a time-to-flattening
                  criterion for fluvial transport.
        substep_fraction : float, optional
            Fraction of the stability limit used as the substep, to stay comfortably
            below it. Only used when `substeps` is 'adaptive'.
        max_substeps : int, optional
            Maximum number of substeps allowed within a single time step, which
            guarantees termination and sets the smallest possible substep
            (dt / max_substeps). Only used when `substeps` is 'adaptive'.
        min_slope : float, optional
            Slope below which a pair of nodes is ignored when estimating the
            stable substep, to avoid vanishingly small substeps on near-flat
            terrain. Only used when `substeps` is 'adaptive'.
        fields_to_track : str or array-like, optional
            The name of the fields at grid nodes to add to the StackedLayers at
            each iteration.
        """
        super().__init__(grid)

        # Parameters
        self._fields_to_track = format_fields_to_track(fields_to_track)
        self._substeps = 1 if substeps is None else substeps
        self._substep_fraction = substep_fraction
        self._max_substeps = max_substeps
        self._min_slope = min_slope

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
        **kwargs,
    ):
        """
        Parameters
        ----------
        grid : ModelGrid
            A grid.
        kwargs : dict, optional
            Other input parameters of `_BaseHandler`.
        """
        super().__init__(grid=grid, **kwargs)

        # Physical fields
        self._topography = grid.at_node["topographic__elevation"]

    def _calculate_sediment_thickness(self, dt):
        """
        Calculates the sediment thickness change over the time step `dt` and
        stores it in `_sediment_thickness`. Must be implemented by subclasses.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _calculate_sediment_thickness"
        )

    def _prepare_step(self):
        """
        Precomputes the quantities that stay constant across the substeps of a
        single `run_one_step` and caches them on the component. The base
        implementation does nothing; subclasses extend it (always calling
        ``super()._prepare_step()`` first).
        """
        pass

    def _max_stable_timestep_flattening(self, dt):
        """
        Estimates the largest stable time step based on the time it would take for
        a pair of adjacent core nodes to reverse their slope, given the current
        sediment thickness change over `dt`. Returns infinity when no pair of
        nodes is converging, in which case the remaining time can be used at once.
        """
        grid = self._grid
        links = grid.active_links
        head = grid.node_at_link_head[links]
        tail = grid.node_at_link_tail[links]

        z = self._topography
        rate = np.sum(self._sediment_thickness, axis=1) / dt

        head_is_higher = z[head] >= z[tail]
        node_hi = np.where(head_is_higher, head, tail)
        node_lo = np.where(head_is_higher, tail, head)

        rate_diff = rate[node_lo] - rate[node_hi]
        elevation_diff = z[node_hi] - z[node_lo]
        slope = elevation_diff / grid.length_of_link[links]
        converging = (rate_diff > 0.0) & (slope > self._min_slope)
        if not np.any(converging):
            return np.inf

        return self._substep_fraction * np.min(
            elevation_diff[converging] / rate_diff[converging]
        )

    def _max_stable_timestep_courant_at_links(self, dt):
        """
        Estimates the largest stable time step from the Courant condition
        ``dt <= length**2 / (2 * D)``, evaluated element-wise and minimized over
        the grid. The effective diffusivity is recovered from the sediment flux
        and slope at each active link, so it accounts for the slope exponent and
        active-layer composition.
        """
        links = self._grid.active_links
        link_lengths = self._grid.length_of_link[links]

        flux = np.abs(np.sum(self._sediment_flux_at_links[links], axis=1))
        slope = np.abs(self._slope_at_links[links, 0])
        effective_diffusivity = np.zeros_like(slope)
        np.divide(flux, slope, out=effective_diffusivity, where=slope > self._min_slope)

        stable = np.full(effective_diffusivity.shape, np.inf)
        np.divide(
            link_lengths * link_lengths,
            2.0 * effective_diffusivity,
            out=stable,
            where=effective_diffusivity > 0.0,
        )

        return self._substep_fraction * np.min(stable)

    def _max_stable_timestep_courant_at_nodes(self, dt):
        """
        Estimates the largest stable time step from the Courant condition
        ``dt <= length**2 / (2 * D)`` like `_max_stable_timestep_courant_at_links`,
        but with the effective diffusivity recovered from the node-to-neighbor
        outfluxes of a routing scheme. For each pair of nodes,
        ``D = outflux * length / (cell_area * slope)``, so the criterion becomes
        ``dt <= slope * length * cell_area / (2 * outflux)``. This accounts for
        the slope exponent, the active-layer composition, and the non-linear
        amplification near the critical slope.
        """
        slope = self._slope[..., 0]
        outflux = np.sum(self._sediment_outflux, axis=2)
        lengths = self._link_lengths[self._links_to_neighbors]
        cell_area = self._grid.cell_area_at_node[:, np.newaxis]

        stable = np.full(slope.shape, np.inf)
        np.divide(
            slope * lengths * cell_area,
            2.0 * outflux,
            out=stable,
            where=(outflux > 0.0) & (slope > self._min_slope),
        )

        return self._substep_fraction * np.min(stable)

    def _max_stable_timestep(self, dt):
        """
        Estimates the largest stable time step for the adaptive substepping.
        Defaults to the time-to-flattening criterion, which only relies on the
        topography and the sediment thickness change; subclasses override this
        to select the criterion suited to their scheme.
        """
        return self._max_stable_timestep_flattening(dt)

    def _update_slope(self):
        """
        Refreshes the slope from the current topography between substeps. By
        default this is a no-op (components that derive the slope from the
        topography on the fly need nothing here); subclasses that read a slope set
        externally (e.g. by a flow accumulator) override this to keep the slope
        consistent with the evolving topography, while leaving the flow directions
        and discharge unchanged.
        """
        pass

    def _apply_substep(
        self, dt, accumulated_thickness, update_compatible=False, update=False
    ):
        """
        Applies the current sediment thickness to the topography and the
        stratigraphy, accumulates it for bookkeeping, and refreshes the slope.
        """
        core_nodes = self._grid.core_nodes

        self._topography[core_nodes] += np.sum(
            self._sediment_thickness[core_nodes], axis=1
        )
        accumulated_thickness[core_nodes] += self._sediment_thickness[core_nodes]
        self._update_stratigraphy(dt, update_compatible, update)
        self._update_slope()

    def _run(self, dt, update_compatible=False, update=False):
        """
        Runs the component over the time step `dt`, optionally subdividing it
        according to `substeps`, and applies the resulting sediment
        fluxes to the topography and stratigraphy. The topography, slope, and
        stratigraphy are updated at every substep (but not the flow directions or
        discharge). Only the first substep honors the caller's layer flags; the
        remaining substeps deposit into that same layer (`update=True`), so that a
        single consolidated layer is still produced per `run_one_step`.
        """
        core_nodes = self._grid.core_nodes

        self._prepare_step()

        accumulated_thickness = np.zeros_like(self._sediment_thickness)

        if isinstance(self._substeps, int):
            sub_dt = dt / self._substeps
            for substep in range(self._substeps):
                self._calculate_sediment_thickness(sub_dt)
                if substep == 0:
                    self._apply_substep(
                        sub_dt, accumulated_thickness, update_compatible, update
                    )
                else:
                    self._apply_substep(sub_dt, accumulated_thickness, update=True)
        else:  # 'adaptive'
            remaining = dt
            floor = dt / self._max_substeps
            for step in range(self._max_substeps):
                self._calculate_sediment_thickness(remaining)
                if step < self._max_substeps - 1:
                    sub_dt = min(
                        max(self._max_stable_timestep(remaining), floor), remaining
                    )
                else:
                    sub_dt = remaining
                self._sediment_thickness *= sub_dt / remaining
                if step == 0:
                    self._apply_substep(
                        sub_dt, accumulated_thickness, update_compatible, update
                    )
                else:
                    self._apply_substep(sub_dt, accumulated_thickness, update=True)
                remaining -= sub_dt
                if remaining <= 0.0:
                    break

        self._sediment_thickness[core_nodes] = accumulated_thickness[core_nodes]

    def run_one_step(self, dt, update_compatible=False, update=False):
        """Run the component for one timestep, dt.

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
        self._run(dt, update_compatible, update)


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
        kwargs : dict, optional
            Other input parameters of `_BaseMover`.
        """
        super().__init__(grid=grid, **kwargs)

        # Parameters
        self._K_cont = convert_to_array(diffusivity_cont)
        self._K_mar = convert_to_array(diffusivity_mar)
        self.wave_base = wave_base
        self._porosity = np.ascontiguousarray(
            np.broadcast_to(
                convert_to_array(porosity), self._stratigraphy.number_of_classes
            ),
            dtype=float,
        )
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
        # Nodes whose stack is fully included in the active layer, i.e., where
        # the bedrock is exposed
        self._stack_exhausted = np.ones((n_nodes, 1, 1), dtype=bool)

    def _prepare_step(self):
        """
        Precomputes the substep-invariant quantities for the diffusion models.
        The sediment diffusivity depends only on the bathymetry, which is frozen
        between substeps, so it is computed once per step here.
        """
        super()._prepare_step()
        self._calculate_sediment_diffusivity()

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
        if max_thickness_br > 0.0:
            active_layer, exhausted = self._stratigraphy.get_active_layer(
                max_thickness_sed, porosity, return_exhausted=True
            )
            self._active_layer[self._grid.core_nodes, 0] = active_layer
            self._active_layer_thickness[:] = np.sum(
                self._active_layer, axis=2, keepdims=True
            )
            # The bedrock contributes to the active layer where it is exposed,
            # i.e., where the active layer already includes all the sediments
            # of the stack
            self._stack_exhausted[self._grid.core_nodes, 0, 0] = exhausted
            np.add(
                self._active_layer,
                self._bedrock_composition
                * (
                    (1.0 - self._porosity) * max_thickness_br
                    - self._active_layer_thickness
                ),
                out=self._active_layer,
                where=self._stack_exhausted
                & (
                    self._active_layer_thickness
                    < (1.0 - self._porosity) * max_thickness_br
                ),
            )
        else:
            self._active_layer[self._grid.core_nodes, 0] = (
                self._stratigraphy.get_active_layer(max_thickness_sed, porosity)
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

    def _max_stable_timestep(self, dt):
        """
        Selects the link-based Courant criterion, matching the link-based
        finite-volume scheme of the diffusers.
        """
        return self._max_stable_timestep_courant_at_links(dt)


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
        kwargs : dict, optional
            Other input parameters of `_BaseDiffuser`.
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
            **kwargs,
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
            if self._flow_proportions.ndim == 2:
                self._flow_proportions = self._flow_proportions[..., np.newaxis]
        else:
            self._flow_proportions = np.ones((n_nodes, n_receivers, 1))
        self._discharge = grid.at_node["surface_water__discharge"]
        self._water_flux = np.zeros((n_nodes, 1, 1))
        self._water_flux_term = np.zeros((n_nodes, n_receivers, 1))
        self._distance_to_receiver = np.zeros((n_nodes, n_receivers))
        self._has_receiver = np.zeros((n_nodes, n_receivers), dtype=bool)
        self._elevation_drop = np.zeros((n_nodes, n_receivers))

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

    def _prepare_step(self):
        """
        Precomputes the substep-invariant quantities for the stream power models:
        the normalized water flux, the discharge term of the stream power law,
        and the distance to the flow receivers, all of which depend only on the
        (frozen) flow network and discharge.
        """
        super()._prepare_step()
        self._normalize_water_flux()
        self._update_water_flux_term()
        self._update_receiver_distances()

    def _update_receiver_distances(self):
        """
        Updates the distance to the (fixed) flow receivers, used to derive the
        slope from the topography between substeps.
        """
        receivers = self._flow_receivers[..., 0]
        np.hypot(
            self._grid.x_of_node[receivers] - self._grid.x_of_node[:, np.newaxis],
            self._grid.y_of_node[receivers] - self._grid.y_of_node[:, np.newaxis],
            out=self._distance_to_receiver,
        )
        np.not_equal(self._distance_to_receiver, 0.0, out=self._has_receiver)

    def _update_water_flux_term(self):
        """
        Updates the discharge term ``(water_flux * flow_proportions) ** m`` of the
        stream power law from the current (normalized) water flux.
        """
        np.power(
            self._water_flux * self._flow_proportions,
            self._m,
            out=self._water_flux_term,
        )

    def _normalize_water_flux(self):
        """
        Refreshes the private water-flux buffer from the grid discharge field and
        normalizes it if needed. The grid field itself is never modified.
        """
        self._water_flux[:, 0, 0] = self._discharge
        if self.ref_water_flux == "max":
            max_flux = np.max(self._water_flux)
            if max_flux > 0.0:
                self._water_flux[:] /= max_flux
        elif isinstance(self.ref_water_flux, (int, float)):
            self._water_flux[:] /= self.ref_water_flux

    def _max_stable_timestep(self, dt):
        """
        Selects the time-to-flattening criterion for fluvial transport,
        following CHILD, instead of the link-based Courant criterion inherited
        from `_BaseDiffuser`.
        """
        return self._max_stable_timestep_flattening(dt)

    def _update_slope(self):
        """
        Recomputes the steepest slope along the (fixed) flow receivers from the
        current topography, reusing the distances precomputed in `_prepare_step`.
        """
        receivers = self._flow_receivers[..., 0]
        z = self._topography
        np.subtract(z[:, np.newaxis], z[receivers], out=self._elevation_drop)

        slope = self._slope[..., 0]
        slope[...] = 0.0
        np.divide(
            self._elevation_drop,
            self._distance_to_receiver,
            out=slope,
            where=self._has_receiver,
        )
        np.clip(slope, 0.0, None, out=slope)


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
        kwargs : dict, optional
            Other input parameters of `_BaseMover`.
        """
        super().__init__(grid=grid, **kwargs)

        # Parameters
        self._porosity = np.ascontiguousarray(
            np.broadcast_to(
                convert_to_array(porosity), self._stratigraphy.number_of_classes
            ),
            dtype=float,
        )

        # Fields for sediment fluxes
        n_nodes = grid.number_of_nodes
        n_sediments = self._stratigraphy.number_of_classes
        self._sediment_influx = np.zeros((n_nodes, n_sediments))
        self._sediment_outflux = np.zeros((n_nodes, number_of_neighbors, n_sediments))
        self._max_sediment_outflux = np.zeros((n_nodes, n_sediments))

    def _convert_fluxes_to_thickness(self, dt):
        """
        Converts the sediment in- and outfluxes into a sediment thickness change
        over the time step `dt`, stored in `_sediment_thickness`.
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
