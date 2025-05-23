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
    cython.floating [:, :] sediment_influx,
    cython.floating [:, :, :] sediment_outflux,
    const cython.floating [:, :] max_sediment_outflux,
    const cython.floating [:, :] max_bedrock_outflux,
    const double dt,
) noexcept nogil:
    """Calculates sediment influx."""
    cdef unsigned int n_nodes = node_order.shape[0]
    cdef unsigned int n_receivers = flow_receivers.shape[1]
    cdef unsigned int n_sediments = sediment_outflux.shape[2]
    cdef unsigned int node, i, j, k
    cdef double total_outflux, max_outflux
    cdef double ratio

    # Iterate top to bottom through the nodes, update sediment out- and influx.
    # Because calculation of the outflux requires the influx, this operation
    # must be done in an upstream to downstream loop, and cannot be vectorized.
    for i in range(n_nodes - 1, -1, -1):

        # Choose the node id
        node = node_order[i]

        # For each sediment class...
        for k in range(n_sediments):

            # Compute the available sediments, i.e., the maximum ouflux
            max_outflux = sediment_influx[node, k] + max_sediment_outflux[node, k] + max_bedrock_outflux[node, k]
            if max_outflux > 0.:
                # Compute the total outflux
                total_outflux = 0.
                for j in range(n_receivers):
                    total_outflux += sediment_outflux[node, j, k]
                if total_outflux > 0.:
                    # Determine by how much the sediment outflux needs to be decreased
                    ratio = max_outflux/total_outflux
                    if ratio < 1.:
                        # Update the sediment outflux
                        for j in range(n_receivers):
                            sediment_outflux[node, j, k] *= ratio
                # Add the outflux to the influx of the downstream node(s)
                for j in range(n_receivers):
                    # TODO: Check this, it's not in the Landlab components, but it seems that it can fail otherwise
                    if flow_receivers[node, j] > -1:
                        sediment_influx[flow_receivers[node, j], k] += sediment_outflux[node, j, k]
            else:
                for j in range(n_receivers):
                    sediment_outflux[node, j, k] = 0.


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void calculate_sediment_fluxes(
    const id_t [:] node_order,
    const cython.floating [:] cell_area,
    const id_t [:, :] flow_receivers,
    const cython.floating [:] water_flux,
    const cython.floating [:, :] flow_proportions,
    cython.floating [:, :] sediment_influx,
    cython.floating [:, :, :] sediment_outflux,
    const cython.floating [:] settling_velocity,
    const cython.floating [:, :, :] erosion_capacity_sed,
    const cython.floating [:, :, :] erosion_capacity_br,
    const cython.floating [:, :] max_sediment_outflux,
    const double dt,
) noexcept nogil:
    """Calculates sediment influx."""
    cdef unsigned int n_nodes = node_order.shape[0]
    cdef unsigned int n_receivers = flow_receivers.shape[1]
    cdef unsigned int n_sediments = sediment_outflux.shape[2]
    cdef unsigned int node, i, j, k
    cdef double total_outflux, max_outflux
    cdef double ratio

    # Iterate top to bottom through the nodes, update sediment out- and influx.
    # Because calculation of the outflux requires the influx, this operation
    # must be done in an upstream to downstream loop, and cannot be vectorized.
    for i in range(n_nodes - 1, -1, -1):

        # Choose the node id
        node = node_order[i]

        # For each sediment class...
        for k in range(n_sediments):

            # Compute the available sediments, i.e., the maximum ouflux
            max_outflux = sediment_influx[node, k] + max_sediment_outflux[node, k]
            for j in range(n_receivers):
                max_outflux += erosion_capacity_br[node, j, k]
            if max_outflux > 0.:
                # Compute the sediment outflux
                for j in range(n_receivers):
                    if flow_proportions[node, j]*water_flux[node] > 0.:
                        # This applies the formula for sediment outlfux from
                        # Shobe et al. (2017) to each receiver independantly by
                        # assuming that the sediment influx is divided between
                        # each receiver in the same proportions than the water
                        # flux, which might not be the correct appraoch
                        sediment_outflux[node, j, k] = flow_proportions[node, j]*sediment_influx[node, k] + erosion_capacity_sed[node, j, k] + erosion_capacity_br[node, j, k]
                        sediment_outflux[node, j, k] /= 1. + settling_velocity[k]*cell_area[node]/(flow_proportions[node, j]*water_flux[node])
                # Compute the total outflux
                total_outflux = 0.
                for j in range(n_receivers):
                    total_outflux += sediment_outflux[node, j, k]
                if total_outflux > 0.:
                    # Determine by how much the sediment outflux needs to be decreased
                    ratio = max_outflux/total_outflux
                    if ratio < 1.:
                        # Update the sediment outflux
                        for j in range(n_receivers):
                            sediment_outflux[node, j, k] *= ratio

                # Add the outflux to the influx of the downstream node(s)
                for j in range(n_receivers):
                    # TODO: Check this, it's not in the Landlab components, but it seems that it can fail otherwise
                    if flow_receivers[node, j] > -1:
                        sediment_influx[flow_receivers[node, j], k] += sediment_outflux[node, j, k]
