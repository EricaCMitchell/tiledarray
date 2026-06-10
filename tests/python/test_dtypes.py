#  This file is a part of TiledArray.
#  Copyright (C) 2020  Virginia Tech
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import pytest
import tiledarray as ta

# All exposed array types paired with their NumPy dtype.
DTYPE_CASES = [
    (ta.TArray,    ta.TSpArray,    np.float64),
    (ta.TArrayF,   ta.TSpArrayF,   np.float32),
    (ta.TArrayZ,   ta.TSpArrayZ,   np.complex128),
    (ta.TArrayC,   ta.TSpArrayC,   np.complex64),
]
DTYPE_IDS = ["double", "float", "complex128", "complex64"]


@pytest.mark.parametrize("Dense,Sparse,dtype", DTYPE_CASES, ids=DTYPE_IDS)
class TestDtypes:

    def test_dense_construction(self, Dense, Sparse, dtype, world):
        a = Dense([4, 4], 2, world)
        a.fill(1)
        world.fence()

    def test_sparse_construction(self, Dense, Sparse, dtype, world):
        a = Sparse([4, 4], 2, world)
        a.fill(1)
        world.fence()

    def test_dense_buffer_dtype(self, Dense, Sparse, dtype, world):
        a = Dense([4, 4], 2, world)
        a.fill(1)
        world.fence()
        buf = np.array(a)
        assert buf.dtype == dtype
        world.fence()

    def test_from_array_round_trip(self, Dense, Sparse, dtype, world):
        if np.issubdtype(dtype, np.complexfloating):
            data = np.ones((4, 4), dtype=dtype) * (1 + 0j)
        else:
            data = np.ones((4, 4), dtype=dtype)
        a = Dense.from_array(data, [[0, 2, 4], [0, 2, 4]], world)
        world.fence()
        buf = np.array(a)
        assert buf.dtype == dtype
        assert np.allclose(buf, data)
        world.fence()

    def test_dense_sparse_same_shape(self, Dense, Sparse, dtype, world):
        d = Dense([4, 4], 2, world)
        s = Sparse([4, 4], 2, world)
        assert d.shape == s.shape
        world.fence()
