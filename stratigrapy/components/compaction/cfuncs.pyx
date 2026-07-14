"""Cython functions for compaction"""

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


cimport cython
from libc.math cimport exp


################################################################################
# Functions


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compact(
    cython.floating [:, :, :] dz,
    cython.floating [:, :, :] porosity,
    const cython.floating [:] initial_porosity,
    const cython.floating [:] efolding_thickness,
    cython.floating [:] thickness_change,
):
    """Compact the sediments of every stack in a single pass.

    For each stack, walks the layers from the surface down to the bottom while
    accumulating the depth to the bottom of each layer, computes the compacted
    porosity from the depth of the middle of each layer, rescales the thickness
    of each class to conserve solid volume, and accumulates the resulting change
    in total thickness.

    Parameters
    ----------
    dz : memoryview, shape (n_layers, n_stacks, n_classes)
        Thickness of each class in each layer, with index 0 the bottom (oldest)
        layer and the last index the surface (youngest) layer. Modified in place.
    porosity : memoryview, shape (n_layers, n_stacks, n_classes)
        Porosity of each class in each layer. Modified in place.
    initial_porosity : memoryview, shape (n_classes,)
        Initial porosity of the sediments for each class.
    efolding_thickness : memoryview, shape (n_classes,)
        E-folding sediment thickness for each class.
    thickness_change : memoryview, shape (n_stacks,)
        Output: change in total thickness of each stack (new minus old, so
        negative where compaction occurs).
    """
    cdef Py_ssize_t n_layers = dz.shape[0]
    cdef Py_ssize_t n_stacks = dz.shape[1]
    cdef Py_ssize_t n_classes = dz.shape[2]
    cdef Py_ssize_t col
    cdef Py_ssize_t cla
    cdef Py_ssize_t layer
    cdef double depth_to_bottom
    cdef double layer_thickness
    cdef double middle_depth
    cdef double new_porosity
    cdef double old_porosity
    cdef double old_thickness
    cdef double new_thickness
    cdef double delta

    with nogil:
        for col in range(n_stacks):
            depth_to_bottom = 0.
            delta = 0.
            for layer in range(n_layers - 1, -1, -1):
                layer_thickness = 0.
                for cla in range(n_classes):
                    layer_thickness += dz[layer, col, cla]
                depth_to_bottom += layer_thickness
                middle_depth = depth_to_bottom - 0.5 * layer_thickness
                for cla in range(n_classes):
                    new_porosity = initial_porosity[cla] * exp(
                        -middle_depth / efolding_thickness[cla]
                    )
                    old_porosity = porosity[layer, col, cla]
                    if old_porosity < new_porosity:
                        new_porosity = old_porosity
                    old_thickness = dz[layer, col, cla]
                    new_thickness = (
                        old_thickness * (1. - old_porosity) / (1. - new_porosity)
                    )
                    delta += new_thickness - old_thickness
                    dz[layer, col, cla] = new_thickness
                    porosity[layer, col, cla] = new_porosity
            thickness_change[col] = delta
