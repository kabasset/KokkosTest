// @copyright 2022-2024, Antoine Basset (CNES)
// This file is part of Linx <github.com/kabasset/Linx>
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXBASE_MIXINS_MATH_H
#define _LINXBASE_MIXINS_MATH_H

#include "Linx/Base/Types.h"

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <cmath>
#include <numeric> // inner_product

namespace Linx {

/**
 * @brief &pi;
 */
template <typename T>
T pi()
{
  static const T out = std::acos(T(-1));
  return out;
}

/**
 * @brief Compute the absolute value of an integral power.
 */
template <int P, typename T>
KOKKOS_INLINE_FUNCTION T abspow(T x)
{
  if constexpr (P == 0) {
    return bool(x);
  }
  if constexpr (P == 1) {
    return std::abs(x);
  }
  if constexpr (P == 2) {
    return x * x;
  }
  if constexpr (P > 2) {
    return x * x * abspow<P - 2>(x);
  }
}

/**
 * @ingroup pixelwise
 * @ingroup mixins
 * @brief Mixin to provide mathematical operations and transforms to a container.
 * 
 * Implements element-wise mathematical functions which may take a range or scalar argument (or none).
 * In the former case, the number of elements in the range must match that of the container.
 * @see pixelwise
 * @see https://en.cppreference.com/w/cpp/header/cmath for functions description
 */
template <typename T, typename TDerived>
struct MathFunctionsMixin {
#define LINX_MATH_UNARY_INPLACE(function) \
  /** @brief Apply std::##function##(). */ \
  const TDerived& function() const \
  { \
    return LINX_CRTP_CONST_DERIVED.apply( \
        #function, \
        KOKKOS_LAMBDA(auto e) { return std::function(e); }); \
  }

#define LINX_MATH_BINARY_INPLACE(function) \
  /** @brief Apply std::##function##(). */ \
  const TDerived& function(const TDerived& other) const \
  { \
    return LINX_CRTP_CONST_DERIVED.apply( \
        #function, \
        KOKKOS_LAMBDA(auto e, auto f) { return std::function(e, f); }, \
        other); \
  }

#define LINX_MATH_BINARY_SCALAR_INPLACE(function) \
  /** @brief Apply std::##function##(). */ \
  const TDerived& function(T other) const \
  { \
    return LINX_CRTP_CONST_DERIVED.apply( \
        #function, \
        KOKKOS_LAMBDA(auto e) { return std::function(e, other); }); \
  } // TODO rm enable_if and merge with previous function thanks to if constexpr

  LINX_MATH_UNARY_INPLACE(abs)
  LINX_MATH_BINARY_INPLACE(max)
  LINX_MATH_BINARY_SCALAR_INPLACE(max)
  LINX_MATH_BINARY_INPLACE(min)
  LINX_MATH_BINARY_SCALAR_INPLACE(min)
  LINX_MATH_BINARY_INPLACE(fdim)
  LINX_MATH_BINARY_SCALAR_INPLACE(fdim)
  LINX_MATH_UNARY_INPLACE(ceil)
  LINX_MATH_UNARY_INPLACE(floor)
  LINX_MATH_BINARY_INPLACE(fmod)
  LINX_MATH_BINARY_SCALAR_INPLACE(fmod)
  LINX_MATH_UNARY_INPLACE(trunc)
  LINX_MATH_UNARY_INPLACE(round)

  LINX_MATH_UNARY_INPLACE(cos)
  LINX_MATH_UNARY_INPLACE(sin)
  LINX_MATH_UNARY_INPLACE(tan)
  LINX_MATH_UNARY_INPLACE(acos)
  LINX_MATH_UNARY_INPLACE(asin)
  LINX_MATH_UNARY_INPLACE(atan)
  LINX_MATH_BINARY_INPLACE(atan2)
  LINX_MATH_BINARY_SCALAR_INPLACE(atan2)
  LINX_MATH_UNARY_INPLACE(cosh)
  LINX_MATH_UNARY_INPLACE(sinh)
  LINX_MATH_UNARY_INPLACE(tanh)
  LINX_MATH_UNARY_INPLACE(acosh)
  LINX_MATH_UNARY_INPLACE(asinh)
  LINX_MATH_UNARY_INPLACE(atanh)

  LINX_MATH_UNARY_INPLACE(exp)
  LINX_MATH_UNARY_INPLACE(exp2)
  LINX_MATH_UNARY_INPLACE(expm1)
  LINX_MATH_UNARY_INPLACE(log)
  LINX_MATH_UNARY_INPLACE(log2)
  LINX_MATH_UNARY_INPLACE(log10)
  LINX_MATH_UNARY_INPLACE(logb)
  LINX_MATH_UNARY_INPLACE(ilogb)
  LINX_MATH_UNARY_INPLACE(log1p)
  LINX_MATH_BINARY_INPLACE(pow)
  LINX_MATH_BINARY_SCALAR_INPLACE(pow)
  LINX_MATH_UNARY_INPLACE(sqrt)
  LINX_MATH_UNARY_INPLACE(cbrt)
  LINX_MATH_BINARY_INPLACE(hypot)
  LINX_MATH_BINARY_SCALAR_INPLACE(hypot)

  LINX_MATH_UNARY_INPLACE(erf)
  LINX_MATH_UNARY_INPLACE(erfc)
  LINX_MATH_UNARY_INPLACE(tgamma)
  LINX_MATH_UNARY_INPLACE(lgamma)

#undef LINX_MATH_UNARY_INPLACE
#undef LINX_MATH_BINARY_INPLACE
};

#define LINX_MATH_UNARY_NEWINSTANCE(function) \
  /** @relatesalso MathFunctionsMixin @brief Apply std::##function##() (new instance). */ \
  template <typename T, typename TDerived> \
  TDerived function(const MathFunctionsMixin<T, TDerived>& in) \
  { \
    const auto& derived = static_cast<const TDerived&>(in); \
    TDerived out(compose_label(#function, derived), derived.shape()); \
    Kokkos::deep_copy(out.container(), derived.container()); \
    out.function(); \
    return out; \
  }

#define LINX_MATH_BINARY_NEWINSTANCE(function) \
  /** @relatesalso MathFunctionsMixin @brief Apply std::##function##() (new instance). */ \
  template <typename T, typename TDerived, typename TOther> \
  TDerived function(const MathFunctionsMixin<T, TDerived>& in, const TOther& other) \
  { \
    const auto& derived = static_cast<const TDerived&>(in); \
    TDerived out(compose_label(#function, derived), derived.shape()); \
    Kokkos::deep_copy(out.container(), derived.container()); \
    out.function(other); \
    return out; \
  }

LINX_MATH_UNARY_NEWINSTANCE(abs)
LINX_MATH_BINARY_NEWINSTANCE(max)
LINX_MATH_BINARY_NEWINSTANCE(min)
LINX_MATH_BINARY_NEWINSTANCE(fdim)
LINX_MATH_UNARY_NEWINSTANCE(ceil)
LINX_MATH_UNARY_NEWINSTANCE(floor)
LINX_MATH_BINARY_NEWINSTANCE(fmod)
LINX_MATH_UNARY_NEWINSTANCE(trunc)
LINX_MATH_UNARY_NEWINSTANCE(round)

LINX_MATH_UNARY_NEWINSTANCE(cos)
LINX_MATH_UNARY_NEWINSTANCE(sin)
LINX_MATH_UNARY_NEWINSTANCE(tan)
LINX_MATH_UNARY_NEWINSTANCE(acos)
LINX_MATH_UNARY_NEWINSTANCE(asin)
LINX_MATH_UNARY_NEWINSTANCE(atan)
LINX_MATH_BINARY_NEWINSTANCE(atan2)
LINX_MATH_UNARY_NEWINSTANCE(cosh)
LINX_MATH_UNARY_NEWINSTANCE(sinh)
LINX_MATH_UNARY_NEWINSTANCE(tanh)
LINX_MATH_UNARY_NEWINSTANCE(acosh)
LINX_MATH_UNARY_NEWINSTANCE(asinh)
LINX_MATH_UNARY_NEWINSTANCE(atanh)

LINX_MATH_UNARY_NEWINSTANCE(exp)
LINX_MATH_UNARY_NEWINSTANCE(exp2)
LINX_MATH_UNARY_NEWINSTANCE(expm1)
LINX_MATH_UNARY_NEWINSTANCE(log)
LINX_MATH_UNARY_NEWINSTANCE(log2)
LINX_MATH_UNARY_NEWINSTANCE(log10)
LINX_MATH_UNARY_NEWINSTANCE(logb)
LINX_MATH_UNARY_NEWINSTANCE(ilogb)
LINX_MATH_UNARY_NEWINSTANCE(log1p)
LINX_MATH_BINARY_NEWINSTANCE(pow)
LINX_MATH_UNARY_NEWINSTANCE(sqrt)
LINX_MATH_UNARY_NEWINSTANCE(cbrt)
LINX_MATH_BINARY_NEWINSTANCE(hypot)

LINX_MATH_UNARY_NEWINSTANCE(erf)
LINX_MATH_UNARY_NEWINSTANCE(erfc)
LINX_MATH_UNARY_NEWINSTANCE(tgamma)
LINX_MATH_UNARY_NEWINSTANCE(lgamma)

#undef LINX_MATH_UNARY_NEWINSTANCE
#undef LINX_MATH_BINARY_NEWINSTANCE

} // namespace Linx

#endif
