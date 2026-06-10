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

# Dense and sparse double-precision array types.
ARRAY_TYPES = [ta.TArray, ta.TSpArray]
ARRAY_TYPE_IDS = ["TArray", "TSpArray"]


@pytest.fixture(scope="session")
def world():
    return ta.get_default_world()


@pytest.fixture(scope="session", params=ARRAY_TYPES, ids=ARRAY_TYPE_IDS)
def Array(request):
    return request.param
