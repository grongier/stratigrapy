"""Sediment compactor"""

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

from ...utils import convert_to_array, format_fields_to_track


################################################################################
# Component


class SedimentCompactor(Component):
    """Compaction of sediments in a StackedLayers.

    Warning: This component is slow, especially when the number of layers increases.

    References
    ----------
    Granjeon, D. (1996)
        Modélisation stratigraphique déterministe: Conception et applications d'un modèle diffusif 3D multilithologique
        https://theses.hal.science/tel-00648827
    Salles, T., Mallard, C., & Zahirovic, S. (2020)
        goSPL: global scalable paleo landscape evolution
        https://doi.org/10.21105/joss.02804
    """

    _name = "SedimentCompactor"

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
        "sediment__porosity": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Porosity of the sediments at the surface",
        },
    }

    def __init__(
        self,
        grid,
        initial_porosity=0.5,
        efolding_thickness=1000.0,
        fields_to_track=None,
    ):
        """
        Parameters
        ----------
        grid : ModelGrid
            A grid.
        initial_porosity : float or array-like (m/time)
            The initial porosity of the sediments for one or multiple lithologies.
        efolding_thickness : float or array-like (m/time)
            The e-folding sediment thickness for one or multiple lithologies.
        fields_to_track : str or array-like, optional
            The name of the fields at grid nodes to add to the StackedLayers at
            each iteration.
        """
        super().__init__(grid)

        # Parameters
        self.initial_porosity = convert_to_array(initial_porosity)
        if "sediment__porosity" not in grid.at_node:
            _ = grid.add_field(
                "sediment__porosity",
                np.full(
                    (grid.number_of_nodes, len(self.initial_porosity)),
                    self.initial_porosity,
                ),
            )
        self.efolding_thickness = convert_to_array(efolding_thickness)
        self.fields_to_track = format_fields_to_track(fields_to_track)
        if "sediment__porosity" not in self.fields_to_track:
            self.fields_to_track.append("sediment__porosity")

        # Physical fields
        self._topography = grid.at_node["topographic__elevation"]
        self._stratigraphy = grid.stacked_layers

        # Fields for compaction
        self._initial_thickness = np.zeros(self._stratigraphy.number_of_stacks)

    def run_one_step(self):
        """Run the compactor for one timestep."""
        core_nodes = self._grid.core_nodes

        self._initial_thickness[:] = self._stratigraphy.thickness

        _depth = self._stratigraphy.z[..., np.newaxis]  # This is awfully slow
        _depth[1:] = _depth[1:] - (_depth[1:] - _depth[:-1]) / 2.0
        _depth[0] /= 2.0
        porosity = self.initial_porosity * np.exp(
            -_depth / self.efolding_thickness
        )  # This is awfully slow
        porosity = np.minimum(self._stratigraphy["sediment__porosity"], porosity)

        self._stratigraphy["_dz"][:] *= (
            1.0 - self._stratigraphy["sediment__porosity"]
        ) / (1.0 - porosity)
        self._stratigraphy["sediment__porosity"][:] = porosity
        self._topography[core_nodes] -= (
            self._initial_thickness - self._stratigraphy.thickness
        )
