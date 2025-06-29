"""Cython functions for layers"""

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

# https://cython.readthedocs.io/en/stable/src/userguide/fusedtypes.html
ctypedef fused id_t:
    cython.integral
    long long


################################################################################
# Functions


@cython.boundscheck(False)
@cython.wraparound(False)
def deposit_or_erode(
    cython.floating [:, :, :] layers,
    long bottom_index,
    long top_index,
    const cython.floating [:, :] dz,
):
    cdef int n_stacks = layers.shape[1]
    cdef int n_classes = layers.shape[2]
    cdef int col
    cdef int cla
    cdef int layer
    cdef double removed
    cdef double amount_to_remove

    with nogil:
        for col in range(n_stacks):
            for cla in range(n_classes):
                if dz[col, cla] >= 0.:
                    layers[top_index, col, cla] += dz[col, cla]
                else:
                    amount_to_remove = - dz[col, cla]
                    removed = 0.
                    for layer in range(top_index, bottom_index - 1, -1):
                        removed += layers[layer, col, cla]
                        layers[layer, col, cla] = 0.
                        if removed > amount_to_remove:
                            layers[layer, col, cla] = removed - amount_to_remove
                            break


@cython.boundscheck(False)
@cython.wraparound(False)
def get_surface_index(
    const cython.floating [:, :, :] layers,
    long bottom_index,
    long top_index,
    id_t [:] surface_index,
):
    cdef int n_stacks = layers.shape[1]
    cdef int n_classes = layers.shape[2]
    cdef int col
    cdef int cla
    cdef int layer

    with nogil:
        for col in range(n_stacks):
            surface_index[col] = bottom_index
            for layer in range(top_index, bottom_index - 1, -1):
                for cla in range(n_classes):
                    if layers[layer, col, cla] > 0:
                        surface_index[col] = layer
                        break
                else:
                    continue
                break


@cython.boundscheck(False)
@cython.wraparound(False)
def get_superficial_layer(
    const cython.floating [:, :, :] layers,
    long bottom_index,
    const id_t [:] surface_index,
    const cython.floating [:] dz,
    cython.floating [:, :] superficial_layer,
):
    cdef int n_stacks = layers.shape[1]
    cdef int n_classes = layers.shape[2]
    cdef int col
    cdef int cla
    cdef int layer
    cdef double thickness
    cdef double superficial_layer_thickness
    cdef double layer_thickness
    cdef double ratio

    with nogil:
        for col in range(n_stacks):
            superficial_layer_thickness = dz[col]
            thickness = 0.
            for layer in range(surface_index[col], bottom_index - 1, -1):
                for cla in range(n_classes):
                    superficial_layer[col, cla] += layers[layer, col, cla]
                    thickness += layers[layer, col, cla]
                if thickness > superficial_layer_thickness:
                    layer_thickness = 0.
                    for cla in range(n_classes):
                        layer_thickness += layers[layer, col, cla]
                    ratio = (thickness - superficial_layer_thickness)/layer_thickness
                    for cla in range(n_classes):
                        superficial_layer[col, cla] -= ratio*layers[layer, col, cla]
                    break


@cython.boundscheck(False)
@cython.wraparound(False)
def get_active_layer(
    const cython.floating [:, :, :] layers,
    const cython.floating [:, :, :] porosity,
    long bottom_index,
    const id_t [:] surface_index,
    const cython.floating [:] dz,
    cython.floating [:, :] active_layer,
):
    cdef int n_stacks = layers.shape[1]
    cdef int n_classes = layers.shape[2]
    cdef int col
    cdef int cla
    cdef int layer
    cdef double thickness
    cdef double active_layer_thickness
    cdef double layer_thickness
    cdef double ratio

    with nogil:
        for col in range(n_stacks):
            active_layer_thickness = dz[col]
            thickness = 0.
            for layer in range(surface_index[col], bottom_index - 1, -1):
                for cla in range(n_classes):
                    active_layer[col, cla] += (1. - porosity[layer, col, cla])*layers[layer, col, cla]
                    thickness += layers[layer, col, cla]
                if thickness > active_layer_thickness:
                    layer_thickness = 0.
                    for cla in range(n_classes):
                        layer_thickness += layers[layer, col, cla]
                    ratio = (thickness - active_layer_thickness)/layer_thickness
                    for cla in range(n_classes):
                        active_layer[col, cla] -= ratio*(1. - porosity[layer, col, cla])*layers[layer, col, cla]
                    break
