/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2020  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef TA_PYTHON_TRANGE_H
#define TA_PYTHON_TRANGE_H

#include "python.h"

#include <TiledArray/tiled_range.h>
#include <sstream>
#include <string>
#include <vector>

namespace TiledArray {
namespace python {
namespace trange {

auto list(const TiledRange &trange) {
  std::vector<std::vector<int64_t> > v;
  for (auto &&tr1 : trange.data()) {
    auto it = tr1.begin();
    v.push_back({it->first});
    for (; it != tr1.end(); ++it) {
      v.back().push_back(it->second);
    }
  }
  return v;
}

inline TiledRange make_trange(std::vector<std::vector<int64_t> > trange) {
  std::vector<TiledRange1> trange1;
  for (auto tr : trange) {
    trange1.emplace_back(tr.begin(), tr.end());
  }
  return TiledRange(trange1.begin(), trange1.end());
}

// template<>
inline TiledRange make_trange(std::vector<int64_t> shape, size_t block) {
  std::vector<TiledRange1> trange1;
  for (size_t i = 0; i < shape.size(); ++i) {
    trange1.emplace_back(TiledRange1::make_uniform(shape[i], block));
  }
  return TiledRange(trange1.begin(), trange1.end());
}

inline std::string tr1_str(const TiledRange1 &tr1) {
  std::ostringstream ss;
  ss << tr1;
  return ss.str();
}

inline std::string tr_str(const TiledRange &tr) {
  std::ostringstream ss;
  ss << tr;
  return ss.str();
}

// Return list of [lo, hi] pairs for each tile in TiledRange1
inline py::list tr1_tiles(const TiledRange1 &tr1) {
  py::list result;
  for (auto it = tr1.begin(); it != tr1.end(); ++it) {
    py::list pair;
    pair.append(it->first);
    pair.append(it->second);
    result.append(pair);
  }
  return result;
}

void __init__(py::module m) {

  py::class_<TiledRange1>(m, "TiledRange1")
    .def(py::init([](std::vector<int64_t> boundaries) {
      return TiledRange1(boundaries.begin(), boundaries.end());
    }), py::arg("boundaries"))
    .def_static("make_uniform",
      [](size_t extent, size_t block) {
        return TiledRange1::make_uniform(extent, block);
      },
      py::arg("extent"), py::arg("block"))
    .def_property_readonly("tile_extent",
      [](const TiledRange1 &tr1) { return tr1.tile_extent(); })
    .def_property_readonly("extent",
      [](const TiledRange1 &tr1) { return tr1.extent(); })
    .def_property_readonly("tiles", &tr1_tiles)
    .def("__str__", &tr1_str)
    .def("__repr__", &tr1_str)
    .def("__len__",
      [](const TiledRange1 &tr1) { return tr1.tile_extent(); })
    .def("__eq__",
      [](const TiledRange1 &a, const TiledRange1 &b) { return a == b; })
  ;

  py::class_<TiledRange>(m, "TiledRange")
    .def(py::init([](std::vector<TiledRange1> tr1s) {
      return TiledRange(tr1s.begin(), tr1s.end());
    }), py::arg("trange1_list"))
    .def(py::init([](std::vector<std::vector<int64_t>> trange_list) {
      return make_trange(trange_list);
    }), py::arg("trange_list"))
    .def_property_readonly("rank",
      [](const TiledRange &tr) { return tr.rank(); })
    .def_property_readonly("elements_range",
      [](const TiledRange &tr) { return tr.elements_range(); })
    .def_property_readonly("tiles_range",
      [](const TiledRange &tr) { return tr.tiles_range(); })
    .def("make_tile_range",
      [](const TiledRange &tr, std::vector<int64_t> idx) {
        return tr.make_tile_range(idx);
      }, py::arg("idx"))
    .def("__str__", &tr_str)
    .def("__repr__", &tr_str)
    .def("__eq__",
      [](const TiledRange &a, const TiledRange &b) { return a == b; })
    .def_property_readonly("data",
      [](const TiledRange &tr) {
        std::vector<TiledRange1> result(tr.data().begin(), tr.data().end());
        return result;
      })
  ;

  py::implicitly_convertible<std::vector<std::vector<int64_t>>, TiledRange>();
}

}  // namespace trange
}  // namespace python
}  // namespace TiledArray

#endif  // TA_PYTHON_TRANGE_H
