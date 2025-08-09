"""Sea level calculator"""

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
from scipy.interpolate import CubicSpline, PchipInterpolator, make_interp_spline
from landlab import Component


################################################################################
# Component


class SeaLevelCalculator(Component):
    """Calculate sea level based on interpolating from control points."""

    _name = "SeaLevelCalculator"

    _unit_agnostic = True

    _info = {
        "topographic__elevation": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Land surface topographic elevation",
        },
        "sea_level__elevation": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "grid",
            "doc": "Sea level elevation",
        },
        "bathymetric__depth": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "The water depth under the sea",
        },
    }

    def __init__(
        self,
        grid,
        control_time,
        control_sea_level,
        interpolation="linear",
    ):
        """
        Parameters
        ----------
        grid : ModelGrid
            A grid.
        control_time : array
            The time at each control point.
        control_sea_level : array
            The sea level at each control point.
        interpolation : str
            The interpolation method, which can be:
                - 'linear' for linear interpolation (the default).
                - 'quadratic' for spline interpolation of second order.
                - 'cubic' for spline interpolation of third order.
                - 'monotonic' for monotonic spline interpolation of third order.
        """
        super().__init__(grid)

        self._grid.at_grid["sea_level__elevation"] = 0.
        self.initialize_output_fields()

        # Parameters
        self._time = 0.0

        # Physical fields
        self._topography = grid.at_node["topographic__elevation"]
        self._bathymetry = grid.at_node["bathymetric__depth"]

        # Interpolator
        if interpolation == "linear":
            self._interpolator = partial(
                np.interp, xp=control_time, fp=control_sea_level
            )
        elif interpolation == "quadratic":
            self._interpolator = make_interp_spline(
                control_time, control_sea_level, k=2
            )
        elif interpolation == "cubic":
            self._interpolator = CubicSpline(control_time, control_sea_level)
        elif interpolation == "monotonic":
            self._interpolator = PchipInterpolator(control_time, control_sea_level)

        # Initialize the fields
        self.run_one_step(0.0)

    def run_one_step(self, dt):
        """Run the calculator for one timestep, dt.

        Parameters
        ----------
        dt : float (time)
            The imposed timestep.
        """
        self._time += dt

        sea_level = self._interpolator(self._time)
        self._grid.at_grid["sea_level__elevation"] = sea_level

        np.subtract(
            sea_level,
            self._topography,
            out=self._bathymetry,
            where=(self._topography < sea_level),
        )
