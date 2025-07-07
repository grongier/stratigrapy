"""Stacked layers"""

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


import os
import numpy as np
from landlab.layers.eventlayers import (
    _reduce_matrix,
    _valid_keywords_or_raise,
    _allocate_layers_for,
)

from .cfuncs import (
    deposit_or_erode,
    get_surface_index,
    get_superficial_layer,
    get_active_layer,
)


################################################################################
# Base functions


def _broadcast(x, shape):
    """
    Broadcasts a float or an array into an array of shape `shape`.
    """
    try:
        x = x.reshape(shape)
    except (AttributeError, ValueError):
        x = np.broadcast_to(x, shape)
    finally:
        x = np.asarray(x, dtype=float)

    return x


def _deposit_or_erode(layers, bottom_index, top_index, dz):
    """Update the array that contains layers with deposition or erosion."""
    layers = layers.reshape((layers.shape[0], layers.shape[1], -1))
    dz = _broadcast(dz, layers.shape[1:3])

    deposit_or_erode(layers, bottom_index, top_index, dz)


def _get_surface_index(layers, bottom_index, top_index, surface_index):
    """Get index within each stack of the layer at the topographic surface."""
    layers = layers.reshape((layers.shape[0], layers.shape[1], -1))

    get_surface_index(layers, bottom_index, top_index, surface_index)


def _get_superficial_layer(layers, bottom_index, top_index, dz, superficial_layer):
    """Get the thicknesses of all classes in a superficial layer defined by its
    thickness from the surface `dz`.
    """
    layers = layers.reshape((layers.shape[0], layers.shape[1], -1))
    dz = _broadcast(dz, layers.shape[1:2])

    get_superficial_layer(layers, bottom_index, top_index, dz, superficial_layer)


def _get_active_layer(layers, porosity, bottom_index, top_index, dz, active_layer):
    """Get the thicknesses of all classes in an active layer defined by its
    thickness from the surface `dz`.
    """
    layers = layers.reshape((layers.shape[0], layers.shape[1], -1))
    dz = _broadcast(dz, layers.shape[1:2])

    if porosity is not None and porosity.shape == layers.shape:
        get_active_layer(layers, porosity, bottom_index, top_index, dz, active_layer)
    else:
        get_superficial_layer(layers, bottom_index, top_index, dz, active_layer)
        if porosity is not None:
            active_layer *= 1.0 - porosity


class _BlockSlice:
    """Slices that divide a matrix into equally sized blocks."""

    def __init__(self, *args):
        """_BlockSlice([start], stop, [step])"""
        if len(args) > 3:
            raise TypeError(
                f"_BlockSlice expected at most 3 arguments, got {len(args)}"
            )

        self._args = tuple(args)

        self._start = 0
        self._stop = None
        self._step = None

        if len(args) == 1:
            self._stop = args[0]
        elif len(args) == 2:
            self._start, self._stop = args
        elif len(args) == 3:
            self._start, self._stop, self._step = args

        if self._stop is not None and self._stop < self._start:
            raise ValueError(
                "stop ({}) must be greater than start ({})".format(
                    self._stop, self._start
                )
            )

    def __repr__(self):
        return "_BlockSlice({})".format(", ".join([repr(arg) for arg in self._args]))

    @property
    def start(self):
        return self._start

    @property
    def stop(self):
        return self._stop

    @property
    def step(self):
        return self._step

    def indices(self, n_rows):
        """Row indices to blocks within a matrix.

        Parameters
        ----------
        n_rows : int
            The number of rows in the matrix.

        Returns
        -------
        (start, stop, step)
            Tuple of (int* that gives the row of the first block, row of the
            last block, and the number of rows in each block.
        """
        start, stop, step = self.start, self.stop, self.step
        if stop is None:
            stop = n_rows

        start, stop, _ = slice(start, stop).indices(n_rows)

        if step is None:
            step = stop - start

        if step != 0 and (stop - start) % step != 0:
            stop = (stop - start) // step * step + start

        return start, stop, step


def resize_array(array, left_new_cap, right_new_cap):
    """Increase the size of an array, leaving room to grow.

    Parameters
    ----------
    array : ndarray
        The array to resize.
    left_new_cap : int
        New capacity at the start of the zero-th dimension of the resized array.
    right_new_cap : int
        New capacity at the end of the zero-th dimension of the resized array.

    Returns
    -------
    ndarray
        Copy of the input array resized.
    """
    left_new_cap = int(left_new_cap)
    right_new_cap = int(right_new_cap)
    if left_new_cap == 0 and right_new_cap == 0:
        return array

    new_allocated = left_new_cap + array.shape[0] + right_new_cap
    larger_array = np.zeros((new_allocated,) + array.shape[1:], dtype=array.dtype)
    _right_new_cap = -right_new_cap if right_new_cap > 0 else None
    larger_array[left_new_cap:_right_new_cap] = array

    return larger_array


################################################################################
# StackedLayersMixIn


class StackedLayersMixIn:
    """MixIn that adds a StackedLayers attribute to a ModelGrid."""

    def __init__(
        self,
        number_of_classes,
        initial_allocation,
        new_allocation,
        number_of_layers_to_fuse,
        number_of_top_layers,
        fuse_continuously,
        remove_empty_layers,
    ):
        """Instantiates the member variables of the StackedLayersMixIn.

        Parameters
        ----------
        number_of_classes : int
            Number of classes of material (e.g., different lithologies or grain
            sizes) making the layers of the StackedLayers.
        initial_allocation : int or array-like, optional
            Number of layers initially pre-allocated to speed up adding new layers.
            If an int, the layers are pre-allocated at the end; it an array-like,
            the first element corresponds to layers pre-allocated at the start, the
            second to layers pre-allocated at the end.
        new_allocation : int or array-like, optional
            Number of layers pre-allocated when adding new layers to speed up the
            process. If an int, the layers are pre-allocated at the end; it an
            array-like, the first element corresponds to layers pre-allocated at the
            start, the second to layers pre-allocated at the end.
        number_of_layers_to_fuse : int, optional
            Number of layers to merge together when calling `fuse`.
        number_of_top_layers : int, optional
            Number of layers at the top of the stack to ignore when calling `fuse`.
        fuse_continuously : bool, optional
            If True, fuse the layers as they are added; otherwise, wait until the
            number of unfused layers reaches `number_of_layers_to_fuse` (excluding
            the top layers) to fuse them.
        remove_empty_layers : bool, optional
            If True, remove the layers without any deposits at the top of the stack
            when adding a layer; otherwise, keep all the layers.
        """
        self.number_of_classes = number_of_classes
        self.initial_allocation = initial_allocation
        self.new_allocation = new_allocation
        self.number_of_layers_to_fuse = number_of_layers_to_fuse
        self.number_of_top_layers = number_of_top_layers
        self.fuse_continuously = fuse_continuously
        self.remove_empty_layers = remove_empty_layers

    @property
    def stacked_layers(self):
        """StackedLayers for each cell."""
        try:
            self._stacked_layers
        except AttributeError:
            self._stacked_layers = StackedLayers(
                self.number_of_cells,
                self.number_of_classes,
                self.initial_allocation,
                self.new_allocation,
                self.number_of_layers_to_fuse,
                self.number_of_top_layers,
                self.fuse_continuously,
                self.remove_empty_layers,
            )
        return self._stacked_layers

    @property
    def at_layer(self):
        """StackedLayers for each cell."""
        return self.stacked_layers


################################################################################
# StackedLayers


class StackedLayers:
    """Track EventLayers where each event is its own layer and is made of
    multiple classes of material (e.g., different lithologies or grain sizes).

    Parameters
    ----------
    number_of_stacks : int
        Number of layer stacks to track.
    number_of_classes : int
        Number of material classes composing each layer.
    initial_allocation : int or array-like, optional
        Number of layers initially pre-allocated to speed up adding new layers.
        If an int, the layers are pre-allocated at the end; it an array-like,
        the first element corresponds to layers pre-allocated at the start, the
        second to layers pre-allocated at the end.
    new_allocation : int or array-like, optional
        Number of layers pre-allocated when adding new layers to speed up the
        process. If an array-like, different number of layers are pre-allocated
        at the start and at the end of the stacks.
    number_of_layers_to_fuse : int, optional
        Number of layers to merge together when calling `fuse`.
    number_of_top_layers : int, optional
        Number of layers at the top of the stack to ignore when calling `fuse`.
    fuse_continuously : bool, optional
        If True, fuse the layers as they are added; otherwise, wait until the
        number of unfused layers reaches `number_of_layers_to_fuse` (excluding
        the top layers) to fuse them.
    remove_empty_layers : bool, optional
        If True, remove the layers without any deposits at the top of the stack
        when adding a layer; otherwise, keep all the layers.
    """

    def __init__(
        self,
        number_of_stacks,
        number_of_classes=1,
        initial_allocation=0,
        new_allocation=1,
        number_of_layers_to_fuse=1,
        number_of_top_layers=1,
        fuse_continuously=True,
        remove_empty_layers=False,
    ):
        super().__init__()

        if isinstance(initial_allocation, int):
            initial_allocation = (0, initial_allocation)

        self._first_layer = initial_allocation[0]
        self._number_of_layers = 0
        self._number_of_stacks = number_of_stacks
        self._number_of_classes = number_of_classes
        self._surface_index = np.zeros(number_of_stacks, dtype=int)
        self._attrs = {}

        self.number_of_layers_to_fuse = number_of_layers_to_fuse
        self.number_of_top_layers = number_of_top_layers
        self.fuse_continuously = fuse_continuously
        self._number_of_fused_layers = 0
        self._number_of_sublayers = 0

        dims = (self.number_of_layers, self.number_of_stacks, self.number_of_classes)
        self._attrs["_dz"] = np.zeros(dims, dtype=float)
        self._resize(initial_allocation[0], initial_allocation[1])
        self._left_allocated = initial_allocation[0]
        self._right_allocated = initial_allocation[1]
        if isinstance(new_allocation, int):
            self.new_allocation = (new_allocation, new_allocation)
        else:
            self.new_allocation = new_allocation
        self.remove_empty_layers = remove_empty_layers

    def __getitem__(self, name):
        return self._attrs[name][
            self._first_layer : self._first_layer + self.number_of_layers
        ]

    def __setitem__(self, name, values):
        dims = (self.allocated, self.number_of_stacks)
        values = np.asarray(values)
        # Landlab's EventLayers expands and broadcasts the layers; it seems
        # unnecessary, but maybe it's more robust?
        # if values.ndim == 1:
        #     values = np.expand_dims(values, 1)
        # values = np.broadcast_to(values, shape)
        self._attrs[name] = _allocate_layers_for(values, *dims)
        self._attrs[name][
            self._first_layer : self._first_layer + self.number_of_layers
        ] = values

    def __iter__(self):
        return (name for name in self._attrs if not name.startswith("_"))

    def __str__(self):
        lines = [
            "number_of_layers: {number_of_layers}",
            "number_of_stacks: {number_of_stacks}",
            "number_of_classes: {number_of_classes}",
            "tracking: {attrs}",
        ]
        return os.linesep.join(lines).format(
            number_of_layers=self.number_of_layers,
            number_of_stacks=self.number_of_stacks,
            number_of_classes=self.number_of_classes,
            attrs=", ".join(self.tracking) or "null",
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "({number_of_stacks}, {number_of_classes})".format(
                number_of_stacks=self.number_of_stacks,
                number_of_classes=self.number_of_classes,
            )
        )

    @property
    def tracking(self):
        """Layer properties being tracked."""
        return [name for name in self._attrs if not name.startswith("_")]

    def _setup_layers(self, **kwds):
        dims = (self.allocated, self.number_of_stacks)
        for name, array in kwds.items():
            self._attrs[name] = _allocate_layers_for(array, *dims)

    @property
    def number_of_stacks(self):
        """Number of stacks."""
        return self._number_of_stacks

    @property
    def number_of_classes(self):
        """Number of classes."""
        return self._number_of_classes

    @property
    def thickness(self):
        """Total thickness of the columns.

        The sum of all layer thicknesses for each stack as an array
        of shape `(number_of_stacks, )`.
        """
        return np.sum(self.dz, axis=(0, 2))

    def _get_thickness(self, axis, porosity=None):
        """Total sediment thickness of the columns, removing the porosity if given."""
        if porosity is None:
            return np.sum(self.dz, axis=axis)
        elif isinstance(porosity, str) and porosity in self._attrs:
            return np.sum(self.dz * (1.0 - self[porosity]), axis=axis)
        else:
            porosity = np.asarray(porosity)
            if porosity.shape == self.dz.shape[-len(porosity.shape) :]:
                return np.sum(self.dz * (1.0 - porosity), axis=axis)
            else:
                return np.sum(self.dz, axis=axis) * (1.0 - porosity)

    def get_thickness(self, porosity=None):
        """Total sediment thickness of the columns, removing the porosity if given.

        Parameters
        ----------
        porosity : array_like or str
            Porosity of all the layers for each class, which can be a layer
            property given by its name, or the porosity for each class, which
            is then identical for all layers.

        Returns
        -------
        thickness
            The sum of all layer thicknesses for each stack as an array of shape
            `(number_of_stacks,)`.
        """
        return self._get_thickness((0, 2), porosity=porosity)

    @property
    def class_thickness(self):
        """Total thickness of the columns for each class.

        The sum of all layer thicknesses for each stack as an array
        of shape `(number_of_stacks, number_of_classes)`.
        """
        return np.sum(self.dz, axis=0)

    def get_class_thickness(self, porosity=None):
        """Total sediment thickness of the columns for each class, removing the
        porosity if given.

        Parameters
        ----------
        porosity : array_like or str
            Porosity of all the layers for each class, which can be a layer
            property given by its name, or the porosity for each class, which
            is then identical for all layers.

        Returns
        -------
        thickness
            The sum of all layer thicknesses for each stack as an array of shape
            `(number_of_stacks, number_of_classes)`.
        """
        return self._get_thickness(0, porosity=porosity)

    @property
    def layer_thickness(self):
        """Thickness of each layer.

        The sum of all class thicknesses for each layer and stack as an array
        of shape `(number_of_layers, number_of_stacks)`.
        """
        return np.sum(self.dz, axis=2)

    @property
    def z(self):
        """Thickness to bottom of each layer.

        Thickness from the top of each stack to the bottom of each layer
        as an array of shape `(number_of_layers, number_of_stacks)`.
        """
        return np.cumsum(np.sum(self.dz[::-1], axis=2), axis=0)[::-1]

    @property
    def dz(self):
        """Thickness of each class in each layer.

        The thickness of each layer at each stack as an array of shape
        `(number_of_layers, number_of_stacks, number of classes)`.
        """
        return self._attrs["_dz"][
            self._first_layer : self._first_layer + self.number_of_layers
        ]

    def _get_composition(self, layer):
        """Get the composition of all classes in one or several layers."""
        thickness = np.sum(layer, axis=-1, keepdims=True)

        composition = np.zeros_like(layer)
        np.divide(layer, thickness, out=composition, where=thickness > 0.0)

        return composition

    @property
    def composition(self):
        """Composition of each layer.

        The composition of each layer at each stack as an array of shape
        `(number_of_layers, number_of_stacks, number of classes)`.
        """
        return self._get_composition(self.dz)

    @property
    def most_frequent_class(self):
        """The most frequent class of each layer.

        The most frequent class of each layer at each stack as an array of shape
        `(number_of_layers, number_of_stacks)`.
        """
        return np.argmax(self.composition, axis=2)

    @property
    def number_of_layers(self):
        """Total number of layers."""
        return self._number_of_layers

    @property
    def allocated(self):
        """Total number of allocated layers."""
        return self._attrs["_dz"].shape[0]

    def add(self, dz, at_bottom=False, update=False, update_compatible=False, **kwds):
        """Add a layer to the stack.

        Parameters
        ----------
        dz : float or array_like
            Thickness to add to each stack.
        at_bottom : bool, optional
            If False, add the layer to the top of the stack; otherwise, insert
            the layer at the bottom of the stack.
        update : bool, optional
            If False, create a new layer and deposit in that layer; otherwise,
            deposition occurs in the existing layer at the top or bottom of the
            stack (depending on `at_bottom`).
        update_compatible : bool, optional
            If False, create a new layer and deposit in that layer; otherwise,
            deposition occurs in the existing layer at the top or bottom of the
            stack (depending on `at_bottom`) only if the new layer is compatible
            with the existing layer.
        """
        if self.number_of_layers == 0:
            self._setup_layers(**kwds)

        if update_compatible == True:
            is_compatible = self.number_of_layers > 0 and self.is_compatible(dz, **kwds)
        else:
            is_compatible = False

        if (update == False and is_compatible == False) or (
            update == True and self.number_of_layers == 0
        ):
            self._add_empty_layer(at_bottom=at_bottom)

        layer = self._first_layer
        if at_bottom == False:
            layer += self.number_of_layers - 1
        _deposit_or_erode(self._attrs["_dz"], self._first_layer, layer, dz)
        last_layer = self._first_layer + self.number_of_layers - 1
        _get_surface_index(
            self._attrs["_dz"], self._first_layer, last_layer, self._surface_index
        )

        if is_compatible == False:
            for name in kwds:
                try:
                    self._attrs[name][layer] = kwds[name]
                except KeyError as exc:
                    raise ValueError(
                        f"{name!r} is not being tracked. Error in adding."
                    ) from exc

        if self.remove_empty_layers == True:
            self._remove_empty_layers()

    def update(self, layer, dz, **kwds):
        """Update a given layer in the stacks.

        Parameters
        ----------
        layer : int
            Index of the layer to update.
        dz : float or array_like
            Thickness to add to each stack.
        """
        if self.number_of_layers == 0:
            self._setup_layers(**kwds)

        _deposit_or_erode(
            self._attrs["_dz"], self._first_layer, self._first_layer + layer, dz
        )
        last_layer = self._first_layer + self.number_of_layers - 1
        _get_surface_index(
            self._attrs["_dz"], self._first_layer, last_layer, self._surface_index
        )

        for name in kwds:
            try:
                if kwds[name] is not None:
                    self._attrs[name][self._first_layer + layer] = kwds[name]
            except KeyError as exc:
                raise ValueError(
                    f"{name!r} is not being tracked. Error in adding."
                ) from exc

    def _reduce_attribute(self, array, start, stop, step, n_blocks, reducer):
        """Combines layers of a specific attribute."""
        if self.fuse_continuously == True and reducer == np.mean and stop - start == 2:
            factor = np.array([[self._number_of_sublayers], [1]])
            factor = factor.reshape(factor.shape + (1,) * (array.ndim - factor.ndim))
            middle = _reduce_matrix(factor * array[start:stop], step, np.sum) / (
                self._number_of_sublayers + 1.0
            )
        elif reducer == "weighted_mean":
            thickness = self._attrs["_dz"][start:stop]
            total_thickness = np.sum(thickness, axis=0, keepdims=True)
            factor = np.zeros_like(thickness)
            np.divide(
                thickness, total_thickness, out=factor, where=total_thickness > 0.0
            )
            middle = _reduce_matrix(factor * array[start:stop], step, np.sum)
        else:
            middle = _reduce_matrix(array[start:stop], step, reducer)
        top = array[stop : self._first_layer + self._number_of_layers]

        array[start : start + n_blocks] = middle
        array[start + n_blocks : start + n_blocks + len(top)] = top

    def reduce(self, *args, **kwds):
        """reduce([start], stop, [step])
        Combine layers.

        Reduce adjacent layers into a single layer. Layer reduction can be done
        using a function (e.g., np.sum or np.mean) or with the string 'weighted_mean',
        which uses a mean weighted by layer thickness.
        """
        _valid_keywords_or_raise(kwds, required=self.tracking, optional=self._attrs)

        start, stop, step = _BlockSlice(*args).indices(self._number_of_layers)
        start += self._first_layer
        stop += self._first_layer

        if step <= 1:
            return

        n_blocks = (stop - start) // step
        n_removed = n_blocks * (step - 1)
        for name, array in self._attrs.items():
            if name != "_dz":
                reducer = kwds.get(name, np.sum)
                self._reduce_attribute(array, start, stop, step, n_blocks, reducer)
        reducer = kwds.get("_dz", np.sum)
        self._reduce_attribute(self._attrs["_dz"], start, stop, step, n_blocks, reducer)

        self._number_of_layers -= n_removed
        self._right_allocated = (
            len(self._attrs["_dz"]) - self._first_layer - self._number_of_layers
        )
        self._left_allocated = self._first_layer
        last_layer = self._first_layer + self.number_of_layers - 1
        _get_surface_index(
            self._attrs["_dz"], self._first_layer, last_layer, self._surface_index
        )

    def fuse(self, finalize=False, **kwds):
        """Fuse layers to save computational resources during runtime. Layers
        are fused together from the top based on `number_of_layers_to_fuse` and,
        once fused, are not fused again. The top layers defined by
        `number_of_top_layers` are not fused unless `finalize` is True.

        Parameters
        ----------
        finalize : bool, optional
            If True, all the layers are fused, including the top layers defined
            by `number_of_top_layers`; otherwise, the top layers are ignored.
        """
        start_fuse = self._number_of_fused_layers
        stop_fuse = self._number_of_layers

        if finalize == False:
            stop_fuse -= self.number_of_top_layers
            if (
                self.fuse_continuously == False
                and stop_fuse - start_fuse >= self.number_of_layers_to_fuse
            ):
                self.reduce(
                    start_fuse, stop_fuse, self.number_of_layers_to_fuse, **kwds
                )
                self._number_of_fused_layers += (
                    stop_fuse - start_fuse
                ) // self.number_of_layers_to_fuse
            elif self.fuse_continuously == True and stop_fuse > start_fuse:
                self.reduce(start_fuse, stop_fuse, **kwds)
                # Increasing the number of sublayers here means that fuse needs
                # to be called multiple times when multiple components are used
                # to get the proper behavior
                self._number_of_sublayers += 1
                if self._number_of_sublayers == self.number_of_layers_to_fuse:
                    self._number_of_sublayers = 0
                    self._number_of_fused_layers += 1
        else:
            if stop_fuse - start_fuse > self.number_of_layers_to_fuse:
                self.reduce(
                    start_fuse, stop_fuse, self.number_of_layers_to_fuse, **kwds
                )
                self._number_of_fused_layers += (
                    stop_fuse - start_fuse
                ) // self.number_of_layers_to_fuse
                start_fuse = self._number_of_fused_layers
                stop_fuse = self._number_of_layers
            self.reduce(start_fuse, stop_fuse, **kwds)
            self._number_of_fused_layers += 1

    @property
    def surface_index(self):
        """Index to the top non-empty layer."""
        return self._surface_index

    def get_surface_values(self, name):
        """Values of a field on the surface layer."""
        return self._attrs[name][self.surface_index, np.arange(self._number_of_stacks)]

    def get_surface_composition(self):
        """Composition of the surface layer (i.e., proportion of each class)."""
        return self._get_composition(self.get_surface_values("_dz"))

    def get_superficial_layer(self, dz):
        """Get the thicknesses of all classes in a superficial layer defined by
        its thickness from the surface `dz`.

        Parameters
        ----------
        dz : float or array_like
            Thickness from the surface of the superficial layer.

        Returns
        -------
        superficial_layer
            The thickness of material from each class within the superficial layer.
        """
        superficial_layer = np.zeros((self.number_of_stacks, self.number_of_classes))
        _get_superficial_layer(
            self._attrs["_dz"],
            self._first_layer,
            self._surface_index,
            dz,
            superficial_layer,
        )

        return superficial_layer

    def get_superficial_composition(self, dz):
        """Get the composition of all classes in a superficial layer defined by
        its thickness from the surface `dz`.

        Parameters
        ----------
        dz : float or array_like
            Thickness from the surface of the superficial layer.

        Returns
        -------
        superficial_composition
            The composition of material from each class within the superficial layer.
        """
        return self._get_composition(self.get_superficial_layer(dz))

    def get_active_layer(self, dz, porosity=None):
        """Get the thicknesses of all classes in an active layer defined by its
        thickness from the surface `dz`.

        Parameters
        ----------
        dz : float or array_like
            Thickness from the surface of the active layer.
        porosity : array_like or str
            Porosity of all the layers for each class, which can be a layer
            property given by its name, or the porosity for each class, which
            is then identical for all layers.

        Returns
        -------
        active_layer
            The thickness of material from each class within the active layer.
        """
        if isinstance(porosity, str):
            porosity = self._attrs[porosity]
        elif porosity is not None:
            porosity = np.asarray(porosity)

        active_layer = np.zeros((self.number_of_stacks, self.number_of_classes))
        _get_active_layer(
            self._attrs["_dz"],
            porosity,
            self._first_layer,
            self._surface_index,
            dz,
            active_layer,
        )

        return active_layer

    def get_active_composition(self, dz, porosity=None):
        """Get the composition of all classes in an active layer defined by its
        thickness from the surface `dz`.

        Parameters
        ----------
        dz : float or array_like
            Thickness from the surface of the active layer.
        porosity : array_like or str
            Porosity of all the layers for each class, which can be a layer
            property given by its name, or the porosity for each class, which
            is then identical for all layers.

        Returns
        -------
        active_composition
            The composition of material from each class within the active layer.
        """
        return self._get_composition(self.get_active_layer(dz, porosity))

    def _add_empty_layer(self, at_bottom=False):
        """Add a new empty layer to the stacks."""
        if at_bottom == False and self._right_allocated == 0:
            self._resize(0, self.new_allocation[1])
            self._right_allocated = self.new_allocation[1]
        elif at_bottom == True and self._left_allocated == 0:
            self._resize(self.new_allocation[0], 0)
            self._left_allocated = self.new_allocation[0]
            self._first_layer = self.new_allocation[0]

        self._number_of_layers += 1
        if at_bottom == False:
            self._right_allocated -= 1
            layer = self._first_layer + self.number_of_layers - 1
        else:
            self._left_allocated -= 1
            self._first_layer -= 1
            layer = self._first_layer
        self._attrs["_dz"][layer, :] = 0.0
        for name in self._attrs:
            self._attrs[name][layer] = 0.0

    def _remove_empty_layers(self):
        """Remove empty layers at the top of the stack"""
        number_of_filled_layers = self.surface_index.max() + 1 - self._first_layer
        if number_of_filled_layers < self.number_of_layers:
            self._number_of_layers = number_of_filled_layers

    def remove_last_layers(self, number_of_layers=1):
        """Remove the last layers at the top of the stack

        Parameters
        ----------
        number_of_layers : int
            The number of layers to remove
        """
        self._number_of_layers -= number_of_layers
        self._right_allocated += number_of_layers
        self._surface_index -= number_of_layers

    def is_compatible(self, dz, **kwds):
        """Check if a new layer is compatible with the existing top layer.

        Parameters
        ----------
        dz : float or array_like
            Thickness to add to each stack.

        Returns
        -------
        bool
            ``True`` if the new layer is compatible, otherwise ``False``.
        """
        if isinstance(dz, np.ndarray) == True and dz.ndim == 2:
            dz = np.sum(dz, axis=1)

        where_deposition = dz > 0.0

        if np.any(where_deposition):
            if not_tracked := set(kwds) - set(self):
                raise ValueError(
                    "Error adding layer."
                    f" {', '.join(sorted(repr(t) for t in not_tracked))}"
                    " is not being tracked. Currently tracking:"
                    f" {', '.join(sorted(repr(t) for t in set(self)))}"
                )
            for name in kwds:
                is_compatible = self[name][-1] == kwds[name]

                if not np.all(is_compatible[where_deposition]):
                    return False
        return True

    def _resize(self, left_new_cap, right_new_cap):
        """Allocate more memory for the layers."""
        for name in self._attrs:
            self._attrs[name] = resize_array(
                self._attrs[name], left_new_cap, right_new_cap
            )
