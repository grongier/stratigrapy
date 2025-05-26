"""Bedrock weatherer"""

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

from .._base import _BaseHandler
from ...utils import convert_to_array


################################################################################
# Component

class BedrockWeatherer(_BaseHandler):
    """Bedrock weathering in a StackedLayers.

    Reference
    ---------
    Tucker, G. E., & Slingerland, R. L. (1994)
        Erosional dynamics, flexural isostasy, and long‐lived escarpments: A numerical modeling study
        https://doi.org/10.1029/94JB00320
    Granjeon, D. (1996)
        Modélisation stratigraphique déterministe: Conception et applications d'un modèle diffusif 3D multilithologique
        https://theses.hal.science/tel-00648827
    """

    _name = "BedrockWeatherer"

    _unit_agnostic = True

    _info = {
        "bathymetric__depth": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "The water depth under the sea",
        },
    }

    def __init__(self,
                 grid,
                 max_weathering_rate=1e-5,
                 weathering_decay_depth=1.,
                 wave_base=20.,
                 bedrock_composition=1.,
                 fields_to_track=None):
        """
        Parameters
        ----------
        grid : ModelGrid
            A grid.
        max_weathering_rate : float (m/time)
            The maximum weathering rate of the bedrock.
        weathering_decay_depth : float (m)
            The characteristic weathering depth.
        wave_base : float (m)
            The wave base, below which weathering decreases exponentially.
        bedrock_composition : float or array-like (-)
            The composition of the material is added to the StackedLayers from the
            bedrock.
        fields_to_track : str or array-like, optional
            The name of the fields at grid nodes to add to the StackedLayers at
            each iteration.
        """
        self.bedrock_composition = convert_to_array(bedrock_composition)
        n_sediments = len(self.bedrock_composition)

        super().__init__(grid, n_sediments, fields_to_track)

        # Parameters
        self.max_weathering_rate = max_weathering_rate
        self.weathering_decay_depth = weathering_decay_depth
        self.wave_base = wave_base

        # Physical fields
        self._bathymetry = grid.at_node['bathymetric__depth']

        # Fields for weathering
        self._weathering_depth = np.zeros((grid.number_of_nodes, 1))

    def _calculate_weathering_depth(self, dt):
        """
        Calculates the weathering depth over the continental and marine domains.
        """
        self._weathering_depth[:, 0] = self.max_weathering_rate*dt
        self._weathering_depth[self._bathymetry > 0., 0] *= np.exp(-self._bathymetry[self._bathymetry > 0.]/self.wave_base)
        self._weathering_depth[self._grid.core_nodes, 0] *= np.exp(-self._stratigraphy.thickness/self.weathering_decay_depth)

    def run_one_step(self, dt, update_compatible=False, update=False, update_time=False):
        """Run the weatherer for one timestep, dt.

        Parameters
        ----------
        dt : float (time)
            The imposed timestep.
        update_compatible : bool, optional
            If False, create a new layer and deposit in that layer; otherwise,
            deposition occurs in the existing layer at the bottom of the stack
            only if the new layer is compatible with the existing layer.
        update : bool
            If false, a new layer is addded at each iteration to the StackedLayers
            of the grid; otherwise, the first layer is simply updated to save
            memory.
        update_time : bool, optional
            If false, time is updated when `update` is true; otherwise the original
            time of the layer is preserved.
        """
        core_nodes = self._grid.core_nodes

        self._time += dt

        self._calculate_weathering_depth(dt)
        self._sediment_thickness[core_nodes] = self.bedrock_composition*self._weathering_depth[core_nodes]

        self._update_stratigraphy(update_compatible, update, True)
