"""Stress-driven router"""

# MIT License

# Copyright (c) 2026 Guillaume Rongier

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
from ...utils import convert_to_array

################################################################################
# Component


class StressDrivenRouter(_BaseRouter, _BaseStreamPower):
    """Water-driven diffusion of a Landlab field in continental and marine domains
    based on a routing scheme and a Meyer-Peter-Mueller-like transport formula.

    This component was developed to compare the results of StratigraPy to those
    of CHILD, but has not been thoroughly tested.

    References
    ----------
    Tucker, G. E. (2014)
        CHILD Users Guide for version R10.7
        https://github.com/childmodel/child/blob/master/Child/Docs/child_users_guide.pdf
    Granjeon, D. (1996)
        Modélisation stratigraphique déterministe: Conception et applications d'un modèle diffusif 3D multilithologique
        https://theses.hal.science/tel-00648827
    """

    _name = "StressDrivenRouter"

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
        shear_coefficient=1000.0,
        roughness=0.05,
        sediment_diameter=0.0005,
        wave_base=20.0,
        porosity=0.0,
        max_erosion_rate_sed=1e-2,
        active_layer_rate_sed=None,
        bedrock_composition=1.0,
        max_erosion_rate_br=1e-2,
        active_layer_rate_br=None,
        exponent_discharge=0.66667,
        exponent_slope=0.66667,
        exponent_shear=1.5,
        exponent_hiding=0.75,
        critical_shields_stress=0.045,
        sediment_density=2650.0,
        water_density=1000.0,
        gravitational_acceleration=9.81,
        ref_water_flux=None,
        substeps=None,
        substep_fraction=0.3,
        max_substeps=1000,
        min_slope=1e-7,
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
        shear_coefficient : float or array-like
            The coeﬃcient relating shear stress to discharge and slope. If None,
            it is calculated from `water_density`, `gravitational_acceleration`,
            and `roughness`.
        roughness : float or array-like (-), optional
            A dimensionless friction factor, only used when `shear_coefficient`
            is None (see Tucker & Slingerland, 1997, https://doi.org/10.1029/97WR00409).
        sediment_diameter : float or array-like (m)
            The diameter of the sediments for one or multiple lithologies.
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
        exponent_shear : float (-)
            The exponent for the excess shear.
        exponent_hiding : float (-)
            The exponent for correcting the critical shear stress for protusion
            and hiding.
        critical_shields_stress : float
            The critical Shields stress.
        sediment_density : float (kg/m3)
            The density of the sediments for one or multiple lithologies.
        water_density : float (kg/m3)
            The density of the water.
        gravitational_acceleration : float (m/time2)
            The gravitational acceleration.
        ref_water_flux : float or string (m3/time), optional
            The reference water flux by which the water discharge is normalized.
            If a float, that value is used at each time step; if 'max', the
            maximum value of discharge at each time step is used.
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
            substeps=substeps,
            substep_fraction=substep_fraction,
            max_substeps=max_substeps,
            min_slope=min_slope,
        )

        second_per_year = 60.0 * 60.0 * 24.0 * 365.25
        if shear_coefficient is None:
            self._Kt = (
                water_density
                * gravitational_acceleration ** (2 / 3)
                * convert_to_array(roughness) ** (1 / 3)
                / 2.0
            )
        else:
            self._Kt = convert_to_array(shear_coefficient)
        self._Kt *= second_per_year ** (-self._m)
        self._d = convert_to_array(sediment_diameter)
        self._p = exponent_shear
        self._hiding_exp = exponent_hiding
        self._thetac = critical_shields_stress
        self._rho_sed = convert_to_array(sediment_density)
        self._rho_water = water_density
        self._g = gravitational_acceleration

        self._d50 = np.zeros((grid.number_of_nodes, 1, 1))
        self._base_shear_stress = (
            self._thetac * (self._rho_sed - self._rho_water) * self._g * self._d
        )

    def _calculate_sediment_outflux(self, dt):
        """
        Calculates the sediment outflux for multiple lithologies.
        """
        core_nodes = self._grid.core_nodes
        cell_area = self._grid.cell_area_at_node[:, np.newaxis, np.newaxis]

        self._calculate_active_layer_composition(dt)

        self._d50[:] = np.sum(
            self._active_layer_composition * self._d, axis=-1, keepdims=True
        )
        self._critical_rate[:] = 0.0
        np.divide(
            self._d50,
            self._d,
            out=self._critical_rate,
            where=self._d != 0,
        )
        self._critical_rate[:] = (
            self._base_shear_stress * self._critical_rate**self._hiding_exp
        )

        self._sediment_outflux[:] = (
            self._K_sed
            * cell_area
            * self._active_layer_composition
            * np.maximum(
                self._Kt * self._water_flux_term * self._slope**self._n
                - self._critical_rate,
                0.0,
            )
            ** self._p
        )

    def _calculate_sediment_thickness(self, dt):
        """Calculates the sediment thickness change over the time step, dt."""
        cell_area = self._grid.cell_area_at_node[:, np.newaxis]

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

        self._convert_fluxes_to_thickness(dt)
