"""Utils"""

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


################################################################################
# Array manipulation


def convert_to_array(x):
    """
    Converts a scalar or an array-like to a NumPy array.
    """
    return np.array([x]) if np.isscalar(x) else np.asarray(x)


def match_array_lengths(*arrays):
    """
    Duplicates the item in one or several arrays to match the length of the
    longest array.
    """
    lengths_over_one = [len(a) for a in arrays if len(a) > 1]
    if not lengths_over_one:
        return arrays

    _arrays = []
    for i, array in enumerate(arrays):
        if len(array) == 1:
            _arrays.append(np.full(lengths_over_one[0], array[0], dtype=array.dtype))
        else:
            _arrays.append(array)

    return _arrays


def reshape_to_match(x, shape):
    """
    Reshapes an array to match the dimensions in a shape.
    """
    new_shape = np.empty(len(shape), dtype=int)
    for i, size in enumerate(shape):
        try:
            j = x.shape.index(size)
            new_shape[i] = x.shape[j]
        except:
            new_shape[i] = 1

    return x.reshape(new_shape)


################################################################################
# Field manipulation


def format_fields_to_track(fields_to_track):
    """
    Formats some grid fields to track.
    """
    if fields_to_track is None:
        return []
    elif isinstance(fields_to_track, str):
        return [fields_to_track]
    else:
        return list(fields_to_track)
