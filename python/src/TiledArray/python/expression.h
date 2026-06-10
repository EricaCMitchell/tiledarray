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

#ifndef TA_PYTHON_EXPRESSION_H
#define TA_PYTHON_EXPRESSION_H

#include "python.h"

#include <tiledarray.h>

#include <complex>
#include <vector>
#include <string>

namespace TiledArray {
namespace python {
namespace expression {

  template<class Array>
  struct Expression {

    struct Term {
      std::shared_ptr<Array> array;
      std::string index;
      double factor = 1;
      auto evaluate() const {
        const auto &array = *this->array;
        return factor*(array(index));
      }
    };

    explicit Expression(std::vector<Term> terms)
      : terms(terms)
    {
    }

    Expression add(const Expression &e) const {
      auto r = this->terms;
      for (const auto &t : e.terms) {
        r.push_back({t.array, t.index, t.factor});
      }
      return Expression{r};
    }

    Expression sub(const Expression &e) const {
      auto r = this->terms;
      for (const auto &t :  e.terms) {
        r.push_back({t.array, t.index, -t.factor});
      }
      return Expression{r};
    }

    Expression mul(double f) const {
      auto r = this->terms;
      for (auto &t :  r) {
        t.factor *= f;
      }
      return Expression{r};
    }

    Expression div(double f) const {
      auto r = this->terms;
      for (auto &t :  r) {
        t.factor /= f;
      }
      return Expression{r};
    }

    template<size_t ... Idx>
    auto operator[](std::integer_sequence<size_t,Idx...>) const {
      return (terms.at(Idx).evaluate() + ...);
    }

  public:
    const std::vector<Term> terms;

  };

  // ContractionExpression: holds two Expression objects for A*B contractions
  template<class Array>
  struct ContractionExpression {
    Expression<Array> lhs;
    Expression<Array> rhs;
  };

  template<size_t N, class Array, size_t ... Idx>
  auto index(const Expression<Array> &e, std::integer_sequence<size_t,Idx...> idx) {
    using Index = std::variant< std::make_integer_sequence<size_t,1+Idx>... >;
    if (N == e.terms.size()) {
      return Index(std::make_integer_sequence<size_t,N>());
    }
    if constexpr (N < TA_PYTHON_MAX_EXPRESSION) {
      return index<N+1>(e, idx);
    }
    throw std::domain_error(
      "Expression exceeds TA_PYTHON_MAX_EXPRESSION=" + std::to_string(TA_PYTHON_MAX_EXPRESSION)
    );
  }

  template<class Array>
  auto index(const Expression<Array> &e) {
    return index<1>(e, std::make_integer_sequence<size_t,TA_PYTHON_MAX_EXPRESSION>());
  }

  template<class F, class Array>
  auto evaluate(F &&f, const Expression<Array> &a) {
    auto visitor = [&](auto &&A) {
      return f(a[A]);
    };
    return std::visit(visitor, index(a));
  }

  template<class F, class Array>
  auto evaluate(F &&f, const Expression<Array> &a, const Expression<Array> &b) {
    auto visitor = [&](auto &&A, auto &&B) {
      return f(a[A], b[B]);
    };
    return std::visit(visitor, index(a), index(b));
  }

#define TA_PYTHON_EXPRESSION_REDUCE(EXPRESSION, OP)                     \
  [](const EXPRESSION &e) {                                             \
    auto op = [](auto &&e) { return e.OP(); };                          \
    return evaluate(op, e).get();                                       \
  }

#define TA_PYTHON_EXPRESSION_REDUCE2(EXPRESSION, OP)                    \
  [](const EXPRESSION &a, const EXPRESSION &b) {                        \
    auto op = [](auto &&a, auto &&b) { return a.OP(b); };               \
    return evaluate(op, a, b).get();                                    \
  }

  template<class Array>
  inline Expression<Array> getitem(std::shared_ptr<Array> array, std::string idx) {
    return Expression<Array>({{array, idx}});
  }

  template<class Array>
  inline void setitem(Array &array, std::string idx, const Expression<Array> &e) {
    auto op = [&](auto &&e) {
      array(idx) = e;
    };
    evaluate(op, e);
  }

  template<class Array>
  inline void setitem_contraction(Array &array, std::string idx,
                                  const ContractionExpression<Array> &ce) {
    auto op = [&array, &idx](auto &&lhs_eval, auto &&rhs_eval) {
      array(idx) = lhs_eval * rhs_eval;
    };
    evaluate(op, ce.lhs, ce.rhs);
  }

  template<class Array>
  void make_array_expression_class(py::module m, const char *name,
                                   const char *contraction_name) {
    using ExprType = Expression<Array>;
    using ContrType = ContractionExpression<Array>;
    using T = typename Array::element_type;

    py::class_<ContrType>(m, contraction_name)
      ;  // just needs to exist as a Python type

    auto cls = py::class_<ExprType>(m, name)
      .def("__add__", &ExprType::add)
      .def("__sub__", &ExprType::sub)
      // mul_expr (Expression * Expression -> ContractionExpression) FIRST
      .def("__mul__", [](const ExprType &self, const ExprType &other) {
        return ContrType{self, other};
      })
      // Then scalar mul
      .def("__mul__", &ExprType::mul)
      .def("__rmul__", &ExprType::mul)
      .def("__truediv__", &ExprType::div)
      .def("__neg__", [](const ExprType &e) { return e.mul(-1.0); })
      .def("norm",        TA_PYTHON_EXPRESSION_REDUCE(ExprType, norm))
      .def("dot",         TA_PYTHON_EXPRESSION_REDUCE2(ExprType, dot))
      .def("squared_norm",TA_PYTHON_EXPRESSION_REDUCE(ExprType, squared_norm))
      .def("sum",         TA_PYTHON_EXPRESSION_REDUCE(ExprType, sum))
      ;

    // Real-valued-only reductions
    if constexpr (std::is_floating_point_v<T>) {
      cls
        .def("min",     TA_PYTHON_EXPRESSION_REDUCE(ExprType, min))
        .def("max",     TA_PYTHON_EXPRESSION_REDUCE(ExprType, max))
        .def("abs_min", TA_PYTHON_EXPRESSION_REDUCE(ExprType, abs_min))
        .def("abs_max", TA_PYTHON_EXPRESSION_REDUCE(ExprType, abs_max))
        .def("trace",   TA_PYTHON_EXPRESSION_REDUCE(ExprType, trace))
        ;
    }
  }

  inline void __init__(py::module m) {
    make_array_expression_class< TArray<double> >(
        m, "Expression", "ContractionExpression");
    make_array_expression_class< TSpArray<double> >(
        m, "SparseExpression", "SparseContractionExpression");
    make_array_expression_class< TArray<float> >(
        m, "ExpressionF", "ContractionExpressionF");
    make_array_expression_class< TSpArray<float> >(
        m, "SparseExpressionF", "SparseContractionExpressionF");
    make_array_expression_class< TArray<std::complex<double>> >(
        m, "ExpressionZ", "ContractionExpressionZ");
    make_array_expression_class< TSpArray<std::complex<double>> >(
        m, "SparseExpressionZ", "SparseContractionExpressionZ");
    make_array_expression_class< TArray<std::complex<float>> >(
        m, "ExpressionC", "ContractionExpressionC");
    make_array_expression_class< TSpArray<std::complex<float>> >(
        m, "SparseExpressionC", "SparseContractionExpressionC");
  }

}
}
}

#endif // TA_PYTHON_EXPRESSION_H
