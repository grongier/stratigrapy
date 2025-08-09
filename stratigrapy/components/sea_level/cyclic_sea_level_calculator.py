"""Cyclic sea level calculator"""

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

from ...utils import convert_to_array, match_array_lengths


################################################################################
# Component


class CyclicSeaLevelCalculator(Component):
    """Calculate sea level based on a mixture of sinusoids."""

    _name = "CyclicSeaLevelCalculator"

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
        wavelength=100000.0,
        time_shift=0.0,
        amplitude=5.0,
        mean=0.0,
        rate=0.0,
    ):
        """
        Parameters
        ----------
        grid : ModelGrid
            A grid.
        wavelength : float or array
            The wavelength(s) of the sinusoid(s) driving sea level variation.
        time_shift : float or array
            The time shift(s) of the sinusoid(s) driving sea level variation,
            which control(s) sea level at time 0.
        amplitude : float or array
            The amplitude(s) of the sinusoid(s) driving sea level variation.
        mean : float
            The mean sea level.
        rate : float
            The linear rate of sea level increase.
        """
        super().__init__(grid)

        self._grid.at_grid["sea_level__elevation"] = 0.
        self.initialize_output_fields()

        # Parameters
        self.wavelength = convert_to_array(wavelength)
        self.time_shift = convert_to_array(time_shift)
        self.amplitude = convert_to_array(amplitude)
        self.wavelength, self.time_shift, self.amplitude = match_array_lengths(
            self.wavelength, self.time_shift, self.amplitude
        )
        self.mean = mean
        self.rate = rate
        self._time = 0.0

        # Physical fields
        self._topography = grid.at_node["topographic__elevation"]
        self._bathymetry = grid.at_node["bathymetric__depth"]

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

        sea_level = self.mean + self.rate * self._time
        for wavelength, time_shift, amplitude in zip(
            self.wavelength, self.time_shift, self.amplitude
        ):
            sea_level += amplitude * np.sin(
                2.0 * np.pi * (self._time - time_shift) / wavelength
            )
        self._grid.at_grid["sea_level__elevation"] = sea_level

        np.subtract(
            sea_level,
            self._topography,
            out=self._bathymetry,
            where=(
                self._topography < sea_level
            ),  # & (self._grid.status_at_node == self._grid.BC_NODE_IS_CORE),
        )
