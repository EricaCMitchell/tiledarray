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

#ifndef TA_PYTHON_ARRAY_H
#define TA_PYTHON_ARRAY_H

#include "expression.h"
#include "python.h"
#include "range.h"
#include "trange.h"

#include <TiledArray/conversions/eigen.h>
#include <TiledArray/dist_array.h>

#include <complex>
#include <cstring>
#include <string>
#include <vector>

namespace TiledArray {
namespace python {
namespace array {

template <typename T>
auto make_tile(py::buffer data) {
  auto shape = data.request().shape;
  py::array_t<T> tmp(shape);
  int result =
      py::detail::npy_api::get().PyArray_CopyInto_(tmp.ptr(), data.ptr());
  if (result < 0) throw py::error_already_set();
  return Tensor<T>(Range(shape), tmp.data());
}

// std::function<py::buffer(const Range&)>
template <class Array>
void init_tiles(Array &a, py::object f) {
  using T = typename Array::element_type;
  py::gil_scoped_release gil;
  auto op = [f](const Range &range) {
    Tensor<T> tile;
    {
      py::gil_scoped_acquire acquire;
      py::buffer buffer = f(range);
      tile = make_tile<T>(buffer);
    }
    return tile;
  };
  a.init_tiles(op);
  a.world().gop.fence();
}

template <class Array, class... Trange>
std::shared_ptr<Array> make_array(const Trange &... args, World *world,
                                  py::object op) {
  if (!world) {
    world = &get_default_world();
  }
  auto array = std::make_shared<Array>(*world, trange::make_trange(args...));
  if (!op.is_none()) {
    init_tiles(*array, op);
  }
  return array;
}

template <class Array>
std::shared_ptr<Array> make_array_from_trange(const TiledRange &tr,
                                              World *world, py::object op) {
  if (!world) {
    world = &get_default_world();
  }
  auto array = std::make_shared<Array>(*world, tr);
  if (!op.is_none()) {
    init_tiles(*array, op);
  }
  return array;
}

template <class Array, class S = std::vector<size_t> >
inline S shape(const Array &a) {
  auto e = a.elements_range().extent();
  S shape(e.size());
  for (size_t i = 0; i < e.size(); ++i) {
    shape[i] = e[i];
  }
  return shape;
}

template <class Array>
inline std::vector<std::vector<int64_t> > trange(const Array &a) {
  return trange::list(a.trange());
}

template <typename T>
py::buffer_info make_buffer_info(Tensor<T> &tile) {
  std::vector<py::ssize_t> strides;
  for (auto s : tile.range().stride()) {
    strides.push_back(static_cast<py::ssize_t>(sizeof(T) * s));
  }
  return py::buffer_info(
      tile.data(),                        /* Pointer to buffer */
      sizeof(T),                          /* Size of one scalar */
      py::format_descriptor<T>::format(), /* Python struct-style format
                                             descriptor */
      tile.range().rank(),                /* Number of dimensions */
      tile.range().extent(),              /* Buffer dimensions */
      strides /* Strides (in bytes) for each index */
  );
}

template <class Array>
inline py::iterator make_iterator(Array &array) {
  return py::make_iterator(array.begin(), array.end());
}

template <class Array>
inline void setitem(Array &array, std::vector<int64_t> idx, py::buffer data) {
  using T = typename Array::element_type;
  auto tile = make_tile<T>(data);
  array.set(idx, tile);
}

template <class Array, class Idx>
inline py::array getitem(const Array &array, Idx idx) {
  auto tile = array.find(idx);
  if (!tile.probe()) {
    auto str = py::str(py::cast(idx));
    throw std::runtime_error("TArray[" + py::cast<std::string>(str) +
                             "] tile is not set");
  }
  return py::array(make_buffer_info(tile.get()));
}

template <class Array>
py::buffer_info make_buffer(Array &a) {
  typedef typename Array::element_type T;
  auto buffer = py::array_t<T>(shape(a));
  // Sparse arrays omit below-threshold tiles entirely (e.g. fill(0) yields an
  // empty array), so zero-initialize and only copy the tiles that are present.
  std::memset(buffer.mutable_data(), 0, buffer.nbytes());
  for (size_t i = 0; i < a.size(); ++i) {
    if (a.is_zero(i)) continue;
    auto range = range::slice(a.trange().make_tile_range(i));
    buffer[range] = getitem(a, i);
  }
  return buffer.request();
}

template <class Array>
using TileReference = typename Array::reference;

template <class Array>
py::array get_reference_data(TileReference<Array> &r) {
  using T = typename Array::element_type;
  auto tile = r.get();
  auto shape = tile.range().extent();
  auto base = py::cast(r);
  return py::array_t<T>(shape, tile.data(), base);
}

template <class Array>
void set_reference_data(TileReference<Array> &r, py::buffer data) {
  using T = typename Array::element_type;
  r = make_tile<T>(data);
}

template <class Array>
std::shared_ptr<Array> from_numpy(py::array np_data,
                                  std::vector<std::vector<int64_t>> trange_list,
                                  World *world) {
  using T = typename Array::element_type;
  if (!world) world = &get_default_world();
  auto tr = trange::make_trange(trange_list);
  auto arr = std::make_shared<Array>(*world, tr);
  py::gil_scoped_release gil;
  arr->init_tiles([np_data](const Range &range) mutable -> Tensor<T> {
    Tensor<T> tile;
    {
      py::gil_scoped_acquire acquire;
      py::list slices;
      const auto &lo = range.lobound();
      const auto &hi = range.upbound();
      for (size_t i = 0; i < range.rank(); ++i) {
        slices.append(py::slice(static_cast<py::ssize_t>(lo[i]),
                                static_cast<py::ssize_t>(hi[i]),
                                py::ssize_t{1}));
      }
      py::array sliced = np_data[py::tuple(slices)];
      tile = make_tile<T>(sliced);
    }
    return tile;
  });
  arr->world().gop.fence();
  return arr;
}

template <class Array>
void py_init_elements(Array &a, py::object f) {
  using T = typename Array::element_type;
  py::gil_scoped_release gil;
  a.init_elements([f](const auto &idx) -> T {
    T result;
    {
      py::gil_scoped_acquire acquire;
      using IndexVec = std::vector<int64_t>;
      IndexVec vidx(idx.begin(), idx.end());
      result = py::cast<T>(f(vidx));
    }
    return result;
  });
  a.world().gop.fence();
}

template <class Array>
void make_array_class(py::object m, const char *name) {
  auto PyArray =
      py::class_<Array, std::shared_ptr<Array> >(m, name, py::buffer_protocol())
          .def(py::init())
          .def(py::init(&make_array<Array, std::vector<int64_t>, size_t>),
               py::arg("shape"), py::arg("block"), py::arg("world") = nullptr,
               py::arg("op") = py::none())
          .def(
              py::init(&array::make_array<Array,
                                          std::vector<std::vector<int64_t> > >),
              py::arg("trange"), py::arg("world") = nullptr,
              py::arg("op") = py::none())
          .def(
              py::init(&array::make_array_from_trange<Array>),
              py::arg("trange"), py::arg("world") = nullptr,
              py::arg("op") = py::none())
          .def_buffer(&array::make_buffer<Array>)
          .def_property_readonly("world", &Array::world,
                                 py::return_value_policy::reference)
          .def_property_readonly("trange", &array::trange<Array>)
          .def_property_readonly("shape", &array::shape<Array, py::tuple>)
          .def("fill", &Array::template fill<>, py::arg("value"),
               py::arg("skip_set") = false)
          .def("init", &array::init_tiles<Array>)
          // Array object needs be alive while iterator is used */
          .def("__iter__", &array::make_iterator<Array>, py::keep_alive<0, 1>())
          .def("__getitem__", &expression::getitem<Array>)
          .def("__setitem__", &expression::setitem_contraction<Array>)
          .def("__setitem__", &expression::setitem<Array>)
          .def("__getitem__", &array::getitem<Array, std::vector<int64_t> >)
          .def("__setitem__", &array::setitem<Array>)
          .def("clone", [](const Array &a) {
            return std::make_shared<Array>(a.clone());
          })
          .def("truncate", [](Array &a) { a.truncate(); })
          .def("is_zero", [](const Array &a, std::vector<int64_t> idx) {
            return a.is_zero(idx);
          })
          .def_property_readonly("is_dense", &Array::is_dense)
          .def_static("from_array", &array::from_numpy<Array>,
                      py::arg("data"), py::arg("trange"),
                      py::arg("world") = nullptr)
          .def("init_elements", &array::py_init_elements<Array>)
      ;

  py::class_<typename Array::reference>(PyArray, "Reference",
                                        py::module_local())
      .def_property_readonly("index", &TileReference<Array>::index)
      .def_property_readonly("range", &TileReference<Array>::make_range)
      .def_property("data", &get_reference_data<Array>,
                    &set_reference_data<Array>);
}

void __init__(py::module m) {
  make_array_class<TArray<double> >(m, "TArray");
  make_array_class<TSpArray<double> >(m, "TSpArray");
  make_array_class<TArray<float> >(m, "TArrayF");
  make_array_class<TSpArray<float> >(m, "TSpArrayF");
  make_array_class<TArray<std::complex<double> > >(m, "TArrayZ");
  make_array_class<TSpArray<std::complex<double> > >(m, "TSpArrayZ");
  make_array_class<TArray<std::complex<float> > >(m, "TArrayC");
  make_array_class<TSpArray<std::complex<float> > >(m, "TSpArrayC");

  // Module-level sparse threshold control
  m.def("get_sparse_threshold",
        []() { return TiledArray::SparseShape<float>::threshold(); });
  m.def("set_sparse_threshold",
        [](float t) { TiledArray::SparseShape<float>::threshold(t); });
}

}  // namespace array
}  // namespace python
}  // namespace TiledArray

#endif  // TA_PYTHON_ARRAY_H
