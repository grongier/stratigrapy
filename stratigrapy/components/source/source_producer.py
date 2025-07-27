"""Source producer"""

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


from functools import partial
import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import CubicSpline, PchipInterpolator, make_interp_spline
from landlab import Component


################################################################################
# Component


class SourceProducer(Component):
    """Produce water and sediments from sources based on interpolating from
    control points.

    Negative fluxes produced by interpolation are truncated to zero.
    """

    _name = "SourceProducer"

    _unit_agnostic = True

    _info = {
        "water__unit_flux_in": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m/s",
            "mapping": "node",
            "doc": "External volume water per area per time input to each node (e.g., rainfall rate)",
        },
        "sediment__unit_flux_in": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m3/s",
            "mapping": "node",
            "doc": "Sediment flux as boundary condition",
        },
    }

    def __init__(
        self,
        grid,
        source_xy,
        water_source_time,
        water_source_influx,
        sediment_source_time,
        sediment_source_influx,
        interpolation="linear",
    ):
        """
        Parameters
        ----------
        grid : ModelGrid
            A grid.
        source_xy : array of shape (n_source, 2) or (2,)
            The x and y coordinates of the source(s). When two sources end up in
            the same cell, their contributions are added.
        water_source_time : array of shape (n_source, n_time) or (n_time,)
            The time at each control point for each source for the water influx.
        water_source_influx : array of shape (n_source, n_time) or (n_time,)
            The water influx at each control point for each source.
        sediment_source_time : array of shape (n_source, n_time) or (n_time,)
            The time at each control point for each source for the sediment influx.
        sediment_source_influx : array of shape (n_source, n_time, n_class), (n_time, n_class), or (n_time,)
            The sediment influx at each control point for each source and each
            sediment class.
        interpolation : str
            The interpolation method, which can be:
                - 'linear' for linear interpolation (the default).
                - 'quadratic' for spline interpolation of second order.
                - 'cubic' for spline interpolation of third order.
                - 'monotonic' for monotonic spline interpolation of third order.
        """
        super().__init__(grid)

        # Parameters
        source_xy = np.asarray(source_xy)
        if source_xy.ndim == 1:
            source_xy = source_xy[np.newaxis]
        water_source_influx = np.asarray(water_source_influx)
        if water_source_influx.ndim == 1:
            water_source_influx = np.tile(water_source_influx, (len(source_xy), 1))
        water_source_time = np.asarray(water_source_time)
        if water_source_time.ndim == 1:
            water_source_time = np.broadcast_to(
                water_source_time, water_source_influx.shape
            )
        sediment_source_influx = np.asarray(sediment_source_influx)
        if sediment_source_influx.ndim == 1:
            sediment_source_influx = np.tile(
                sediment_source_influx, (len(source_xy), 1)
            )
        if sediment_source_influx.ndim == 2:
            sediment_source_influx = sediment_source_influx[..., np.newaxis]
        sediment_source_time = np.asarray(sediment_source_time)
        if sediment_source_time.ndim == 1:
            sediment_source_time = np.tile(sediment_source_time, (len(source_xy), 1))
        self._time = 0.0

        # Fields
        if "water__unit_flux_in" not in grid.at_node.keys():
            self._water_influx = grid.add_zeros(
                "water__unit_flux_in", at="node", clobber=True
            )
        else:
            self._water_influx = grid.at_node["water__unit_flux_in"]
        if "sediment__unit_flux_in" not in grid.at_node.keys():
            self._sediment_influx = grid.add_field(
                "sediment__unit_flux_in",
                np.zeros((grid.number_of_nodes, sediment_source_influx.shape[2])),
                clobber=True,
            )
        else:
            self._sediment_influx = grid.at_node["sediment__unit_flux_in"]
        if self._sediment_influx.ndim == 1:
            self._sediment_influx = self._sediment_influx[:, np.newaxis]

        # Source location in the grid
        kdtree = KDTree(grid.xy_of_cell)
        _, self._source_idx = kdtree.query(source_xy)
        self._source_idx = grid.core_nodes[self._source_idx]

        # Interpolators
        self._water_interpolators = []
        for i in range(len(water_source_time)):
            self._water_interpolators.append(
                self._build_interpolator(
                    water_source_time[i], water_source_influx[i], interpolation
                )
            )
        self._sediment_interpolators = []
        for i in range(len(sediment_source_time)):
            _interpolators = []
            for j in range(sediment_source_influx.shape[2]):
                _interpolators.append(
                    self._build_interpolator(
                        sediment_source_time[i],
                        sediment_source_influx[i, :, j],
                        interpolation,
                    )
                )
            self._sediment_interpolators.append(_interpolators)

    def _build_interpolator(self, x, y, interpolation):
        """Builds an interpolator based on a given interpolation technique."""
        if interpolation == "linear":
            return partial(np.interp, xp=x, fp=y)
        elif interpolation == "quadratic":
            return make_interp_spline(x, y, k=2)
        elif interpolation == "cubic":
            return CubicSpline(x, y)
        elif interpolation == "monotonic":
            return PchipInterpolator(x, y)

    def run_one_step(self, dt):
        """Run the producer for one timestep, dt.

        Parameters
        ----------
        dt : float (time)
            The imposed timestep.
        """
        self._time += dt

        self._water_influx[self._source_idx] = 0.0
        for i, interpolator in enumerate(self._water_interpolators):
            value = interpolator(self._time)
            if value > 0.:
                self._water_influx[self._source_idx[i]] += value
        self._sediment_influx[self._source_idx] = 0.0
        for i, interpolators in enumerate(self._sediment_interpolators):
            for j, interpolator in enumerate(interpolators):
                value = interpolator(self._time)
                if value > 0.:
                    self._sediment_influx[self._source_idx[i], j] += value
