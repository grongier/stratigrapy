"""Cython functions for stream power"""

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
cpdef void calculate_sediment_influx(
    const id_t [:] node_order,
    const id_t [:, :] flow_receivers,
    const cython.floating [:] cell_area,
    const cython.floating [:, :, :] superficial_layer,
    cython.floating [:, :] sediment_influx,
    cython.floating [:, :, :] sediment_outflux,
    cython.floating [:, :, :] bedrock_outflux,
    const double dt,
) noexcept nogil:
    """Calculates sediment influx."""
    cdef unsigned int n_nodes = node_order.shape[0]
    cdef unsigned int n_receivers = flow_receivers.shape[1]
    cdef unsigned int n_sediments = superficial_layer.shape[1]
    cdef unsigned int node, i, j, k
    cdef double total_sediment_outflux, max_sediment_outflux
    cdef double ratio

    # Iterate top to bottom through the nodes, update sediment out- and influx.
    # Because calculation of the outflux requires the influx, this operation
    # must be done in an upstream to downstream loop, and cannot be vectorized.
    for i in range(n_nodes - 1, -1, -1):

        # Choose the node id
        node = node_order[i]

        # For each sediment class...
        for k in range(n_sediments):

            # Compute the available sediments, i.e., the maximum sediment ouflux
            max_sediment_outflux = sediment_influx[node, k] + (superficial_layer[node, k, 0] + superficial_layer[node, k, 1])*cell_area[node]/dt
            if max_sediment_outflux > 0.:
                # Compute the total sediment outflux
                total_sediment_outflux = 0.
                for j in range(n_receivers):
                    total_sediment_outflux += sediment_outflux[node, j, k]
                if total_sediment_outflux > 0.:
                    # Determine by how much the sediment outflux needs to be decreased
                    ratio = max_sediment_outflux/total_sediment_outflux
                    if ratio < 1.:
                        # Update the sediment outflux
                        for j in range(n_receivers):
                            sediment_outflux[node, j, k] *= ratio
                # Determine the contribution from the sediments and the bedrock
                ratio = superficial_layer[node, k, 1]*cell_area[node]/dt/max_sediment_outflux
                # Update the bedrock outflux (only used to update the topography)
                for j in range(n_receivers):
                    bedrock_outflux[node, j, k] = ratio*sediment_outflux[node, j, k]
            else:
                for j in range(n_receivers):
                    sediment_outflux[node, j, k] = 0.

            # Add the outflux to the influx of the downstream node(s)
            for j in range(n_receivers):
                # TODO: Check this, it's not in the Landlab components, but it seems that it can fail otherwise
                if flow_receivers[node, j] > -1:
                    sediment_influx[flow_receivers[node, j], k] += sediment_outflux[node, j, k]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void calculate_linear_decline_sediment_influx(
    const id_t [:] node_order,
    const id_t [:, :] flow_receivers,
    const cython.floating [:] cell_area,
    const cython.floating [:, :] superficial_layer,
    cython.floating [:, :] sediment_influx,
    cython.floating [:, :, :] sediment_outflux,
    cython.floating [:, :, :] bedrock_outflux,
    const double dt,
) noexcept nogil:
    """Calculates sediment influx."""
    cdef unsigned int n_nodes = node_order.shape[0]
    cdef unsigned int n_receivers = flow_receivers.shape[1]
    cdef unsigned int n_sediments = superficial_layer.shape[1]
    cdef unsigned int node, i, j, k
    cdef double total_sediment_outflux, max_sediment_outflux, transport_capacity
    cdef double ratio, decline

    # Iterate top to bottom through the nodes, update sediment out- and influx.
    # Because calculation of the outflux requires the influx, this operation
    # must be done in an upstream to downstream loop, and cannot be vectorized.
    for i in range(n_nodes - 1, -1, -1):

        # Choose the node id
        node = node_order[i]

        # For each sediment class...
        for k in range(n_sediments):

            # Compute the available sediments, i.e., the maximum sediment ouflux
            max_sediment_outflux = sediment_influx[node, k] + superficial_layer[node, k]*cell_area[node]/dt
            if max_sediment_outflux > 0.:
                # Compute the total sediment outflux, i.e., the transport capacity
                transport_capacity = 0.
                for j in range(n_receivers):
                    transport_capacity += sediment_outflux[node, j, k]
                if transport_capacity > 0.:
                    # Determine by how much the sediment outflux needs to be decreased
                    ratio = max_sediment_outflux/transport_capacity
                    if ratio < 1.:
                        # Update the sediment outflux
                        for j in range(n_receivers):
                            sediment_outflux[node, j, k] *= ratio
                    # Compute the linear decline for bedrock erodibility
                    total_sediment_outflux = 0.
                    for j in range(n_receivers):
                        total_sediment_outflux += sediment_outflux[node, j, k]
                    decline = 1. - total_sediment_outflux/transport_capacity
                else:
                    decline = 1.
            else:
                for j in range(n_receivers):
                    sediment_outflux[node, j, k] = 0.
                decline = 1.

            # Update the bedrock outflux (only used to update the topography)
            for j in range(n_receivers):
                bedrock_outflux[node, j, k] = decline*bedrock_outflux[node, j, k]
                sediment_outflux[node, j, k] += bedrock_outflux[node, j, k]

            # Add the outflux to the influx of the downstream node(s)
            for j in range(n_receivers):
                # TODO: Check this, it's not in the Landlab components, but it seems that it can fail otherwise
                if flow_receivers[node, j] > -1:
                    sediment_influx[flow_receivers[node, j], k] += sediment_outflux[node, j, k]
