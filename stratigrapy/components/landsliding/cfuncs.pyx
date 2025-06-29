"""Cython functions for landsliding"""

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
from libc.math cimport atan, M_PI

# https://cython.readthedocs.io/en/stable/src/userguide/fusedtypes.html
ctypedef fused id_t:
    cython.integral
    long long


################################################################################
# Functions


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void calculate_sediment_influx(
    const id_t [:] node_order,
    const id_t [:] flow_receivers,
    const cython.floating [:] link_lengths,
    const cython.floating [:] cell_area,
    const cython.floating [:] slope,
    const cython.floating [:, :] repose_angle,
    cython.floating [:, :] sediment_influx,
    cython.floating [:, :] sediment_outflux,
    const double dt,
) noexcept nogil:
    """Calculates sediment influx."""
    cdef unsigned int n_nodes = node_order.shape[0]
    cdef unsigned int n_sediments = sediment_outflux.shape[1]
    cdef unsigned int node, i, k
    cdef double total_sediment_influx, repose_slope, new_slope

    # Iterate top to bottom through the nodes, update sediment out- and influx.
    # Because calculation of the outflux requires the influx, this operation
    # must be done in an upstream to downstream loop, and cannot be vectorized.
    for i in range(n_nodes - 1, -1, -1):

        # Choose the node id
        node = node_order[i]

        # Here we assume that the influx gets transported (if needed) before the
        # outflux, i.e., we use the current slope instead of calculating the
        # new slope after the ouflux is gone. This might be conceptually less
        # correct, but it's numerically slightly easier and in the grand scheme
        # of things I don't think it matters much
        # Compute the slope of repose of the influx
        repose_slope = 0.
        total_sediment_influx = 0.
        for k in range(n_sediments):
            total_sediment_influx += sediment_influx[node, k]
            repose_slope += repose_angle[node, k]*sediment_influx[node, k]
        if total_sediment_influx > 0.:
            repose_slope /= n_sediments*total_sediment_influx
            repose_slope = atan(repose_slope*M_PI/180.)
            # Compute the new depositional slope
            new_slope = slope[node]
            if cell_area[node] > 0. and link_lengths[node] > 0.:
                new_slope += total_sediment_influx*dt/cell_area[node]/link_lengths[node]
            # If the new slope is larger than the slope of repose...
            if new_slope > repose_slope:
                # Add the excess sediments to the outflux
                for k in range(n_sediments):
                    sediment_outflux[node, k] += min(sediment_influx[node, k],
                                                     (sediment_influx[node, k]/total_sediment_influx)*(new_slope - repose_slope)*link_lengths[node]*cell_area[node]/dt)

        # TODO: Check this, it's not in the Landlab components, but it seems that it can fail otherwise
        if flow_receivers[node] > -1:
            # For each sediment class...
            for k in range(n_sediments):
                # Add the outflux to the influx of the downstream node(s)
                sediment_influx[flow_receivers[node], k] += sediment_outflux[node, k]
