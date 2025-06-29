"""Cython functions for grid calculations"""

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


cimport cython
from cython.parallel cimport prange

# https://cython.readthedocs.io/en/stable/src/userguide/fusedtypes.html
ctypedef fused float_or_int:
    cython.integral
    long long
    cython.floating


################################################################################
# Functions


@cython.boundscheck(False)
@cython.wraparound(False)
def _calc_mult_flux_div_at_node(
    shape,
    xy_spacing,
    const float_or_int[:, :] value_at_link,
    cython.floating[:, :] out,
):
    cdef int n_rows = shape[0]
    cdef int n_cols = shape[1]
    cdef int n_classes = value_at_link.shape[1]
    cdef double dx = xy_spacing[0]
    cdef double dy = xy_spacing[1]
    cdef int links_per_row = 2 * n_cols - 1
    cdef double inv_area_of_cell = 1.0 / (dx * dy)
    cdef int row, col, clas
    cdef int node, link

    for row in prange(1, n_rows - 1, nogil=True, schedule="static"):
        node = row * n_cols
        link = row * links_per_row

        for col in range(1, n_cols - 1):
            for clas in range(n_classes):
                out[node + col, clas] = (
                    dy * (value_at_link[link + 1, clas] - value_at_link[link, clas])
                    + dx * (value_at_link[link + n_cols, clas] - value_at_link[link - n_cols + 1, clas])
                ) * inv_area_of_cell
            link = link + 1
