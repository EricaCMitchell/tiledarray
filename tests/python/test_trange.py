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

import pytest
import tiledarray as ta

ARRAY_TYPES = [ta.TArray, ta.TSpArray]
ARRAY_TYPE_IDS = ["TArray", "TSpArray"]


class TestTiledRange1:

    def test_ctor(self):
        tr1 = ta.TiledRange1([0, 2, 4, 8])
        assert tr1.tile_extent == 3
        assert tr1.extent == 8

    def test_make_uniform(self):
        tr1 = ta.TiledRange1.make_uniform(10, 4)
        assert tr1.extent == 10
        assert tr1.tile_extent == 3  # tiles: [0,4), [4,8), [8,10)

    def test_len(self):
        tr1 = ta.TiledRange1([0, 3, 6, 9])
        assert len(tr1) == 3

    def test_tiles(self):
        tr1 = ta.TiledRange1([0, 2, 5])
        tiles = tr1.tiles
        assert len(tiles) == 2
        assert tiles[0] == [0, 2]
        assert tiles[1] == [2, 5]

    def test_eq(self):
        tr1a = ta.TiledRange1([0, 2, 4])
        tr1b = ta.TiledRange1([0, 2, 4])
        tr1c = ta.TiledRange1([0, 3, 4])
        assert tr1a == tr1b
        assert tr1a != tr1c

    def test_str(self):
        tr1 = ta.TiledRange1([0, 2, 4])
        s = str(tr1)
        assert "2" in s

    def test_repr(self):
        tr1 = ta.TiledRange1([0, 2, 4])
        r = repr(tr1)
        assert len(r) > 0


class TestTiledRange:

    def test_from_trange1_list(self):
        tr1a = ta.TiledRange1([0, 2, 4])
        tr1b = ta.TiledRange1([0, 3, 6])
        tr = ta.TiledRange([tr1a, tr1b])
        assert tr.rank == 2

    def test_from_nested_list(self):
        tr = ta.TiledRange([[0, 2, 4], [0, 3, 6]])
        assert tr.rank == 2

    def test_elements_range(self):
        tr = ta.TiledRange([[0, 2, 4], [0, 3, 6]])
        er = tr.elements_range
        assert er.ndim == 2

    def test_tiles_range(self):
        tr = ta.TiledRange([[0, 2, 4], [0, 3, 6]])
        tr_r = tr.tiles_range
        assert tr_r.ndim == 2

    def test_make_tile_range(self):
        tr = ta.TiledRange([[0, 2, 4], [0, 3, 6]])
        tile_r = tr.make_tile_range([0, 0])
        assert tile_r.ndim == 2

    def test_data(self):
        tr = ta.TiledRange([[0, 2, 4], [0, 3, 6]])
        data = tr.data
        assert len(data) == 2

    def test_eq(self):
        tr1 = ta.TiledRange([[0, 2, 4], [0, 3, 6]])
        tr2 = ta.TiledRange([[0, 2, 4], [0, 3, 6]])
        assert tr1 == tr2

    def test_str(self):
        tr = ta.TiledRange([[0, 2, 4], [0, 3, 6]])
        s = str(tr)
        assert len(s) > 0

    def test_repr(self):
        tr = ta.TiledRange([[0, 2, 4], [0, 3, 6]])
        r = repr(tr)
        assert len(r) > 0


@pytest.mark.parametrize("Array", ARRAY_TYPES, ids=ARRAY_TYPE_IDS)
class TestTiledRangeWithArrays:

    def test_array_from_trange(self, Array, world):
        tr = ta.TiledRange([[0, 2, 4], [0, 3, 6]])
        a = Array(tr, world=world)
        assert a.shape == (4, 6)
        world.fence()

    def test_array_trange_round_trip(self, Array, world):
        tr = ta.TiledRange([[0, 2, 4], [0, 3, 6]])
        a = Array(tr, world=world)
        tr_back = a.trange
        assert len(tr_back) == 2
        world.fence()
