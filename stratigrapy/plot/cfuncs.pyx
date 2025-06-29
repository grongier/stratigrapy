"""Cython functions for plotting"""

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
from libc.math cimport NAN


################################################################################
# Functions


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void mask_layer(
    cython.floating [:] layer,
    const cython.floating [:] thickness,
    const bint mask_wedges,
    const bint mask_null_layers,
) noexcept nogil:
    """Masks the parts of a layer with zero thickness."""
    cdef unsigned int i

    for i in range(len(thickness)):
        if thickness[i] == 0.:
            if mask_wedges == True and i > 0 and thickness[i - 1] > 0.:
                layer[i] = layer[i - 1]
            elif mask_wedges == True and i < len(thickness) - 1 and thickness[i + 1] > 0.:
                layer[i] = layer[i + 1]
            elif (mask_null_layers == True
                  and ((i > 0 and thickness[i - 1] == 0. and i < len(thickness) - 1 and thickness[i + 1] == 0.)
                       or (i == 0 and thickness[i + 1] == 0.)
                       or (i == len(thickness) - 1 and thickness[i - 1] == 0.))):
                layer[i] = NAN
