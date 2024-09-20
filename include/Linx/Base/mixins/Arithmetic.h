// @copyright 2022-2024, Antoine Basset (CNES)
// This file is part of Linx <github.com/kabasset/Linx>
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXBASE_MIXINS_ARITHMETIC_H
#define _LINXBASE_MIXINS_ARITHMETIC_H

#include "Linx/Base/Functional.h"
#include "Linx/Base/Types.h" // LINX_FORWARD

#include <Kokkos_Core.hpp>
#include <functional>
#include <type_traits>

namespace Linx {

#define LINX_SCALAR_OPERATOR_INPLACE(op, Func) \
  const TDerived& operator op##=(const T& rhs) const \
  { \
    return LINX_CRTP_CONST_DERIVED.apply(compose_label(#op, LINX_CRTP_CONST_DERIVED, rhs), Func(rhs)); \
  }

#define LINX_SCALAR_OPERATOR_NEWINSTANCE(op) \
  friend TDerived operator op(const TDerived& lhs, const T& rhs) \
  { \
    TDerived out = lhs.copy_as(compose_label(#op, lhs, rhs)); \
    out op## = rhs; \
    return out; \
  }

#define LINX_SCALAR_OPERATOR(op, Func) \
  LINX_SCALAR_OPERATOR_INPLACE(op, Func) \
  LINX_SCALAR_OPERATOR_NEWINSTANCE(op)

#define LINX_VECTOR_OPERATOR_INPLACE(op, Func) \
  template <typename USpecs, typename U, typename UDerived> \
  const TDerived& operator op##=(const ArithmeticMixin<USpecs, U, UDerived>& rhs) const \
  { \
    const auto& derived_rhs = static_cast<const UDerived&>(rhs); \
    return LINX_CRTP_CONST_DERIVED \
        .apply(compose_label(#op, LINX_CRTP_CONST_DERIVED, derived_rhs), Func(), derived_rhs); \
  }

#define LINX_VECTOR_OPERATOR_NEWINSTANCE(op) \
  template <typename USpecs, typename U, typename UDerived> \
  friend TDerived operator op(const TDerived& lhs, const ArithmeticMixin<USpecs, U, UDerived>& rhs) \
  { \
    const auto& derived_rhs = static_cast<const UDerived&>(rhs); \
    TDerived out = lhs.copy_as(compose_label(#op, lhs, derived_rhs)); \
    out op## = derived_rhs; \
    return out; \
  }

#define LINX_VECTOR_OPERATOR(op, Func) \
  LINX_VECTOR_OPERATOR_INPLACE(op, Func) \
  LINX_VECTOR_OPERATOR_NEWINSTANCE(op)

#define LINX_OPERATOR(op, Func) \
  LINX_SCALAR_OPERATOR(op, Func) \
  LINX_VECTOR_OPERATOR(op, Func)

/**
 * @ingroup concepts
 * @requirements{VectorArithmetic}
 * @brief Vector space arithmetic requirements.
 * 
 * Implements vector space arithmetic operators
 * (uppercase letters are for vectors, lowercase letters are for scalars):
 * - Vector-additive: V += U, W = V + U, V -= U, W = V - U;
 * - Scalar-additive: V += a, V = U + a, V = a + U, V -= a, V = U + a, V = a - U, V++, ++V, V--, --V;
 * - Scalar-multiplicative: V *= a, V = U * a, V = a * U, V /= a, V = U / a.
 */
class VectorArithmetic;

/**
 * @ingroup concepts
 * @requirements{EuclidArithmetic}
 * @brief Euclidean ring arithmetic requirements.
 * 
 * Adds the following operators to `VectorArithmetic`:
 * - Vector-multiplicative: V *= U, W = U * V, V /= U, W = V / U;
 * - Scalar-modable: V %= a, V = U % a;
 * - Vector-modable: V %= U, W = V % U;
 */
class EuclidArithmetic;

/**
 * @ingroup pixelwise
 * @ingroup mixins
 * @brief Mixin to provide arithmetics operators to a container.
 * @tparam TSpecs The operators specifications, can be `void`
 * @tparam TDerived The container which inherits this class
 * 
 * @tspecialization{VectorArithmetic}
 * @tspecialization{EuclidArithmetic}
 */
template <typename TSpecs, typename T, typename TDerived>
struct ArithmeticMixin {
  TDerived copy_as(const std::string& label) const
  {
    TDerived out(label, LINX_CRTP_CONST_DERIVED.shape());
    Kokkos::deep_copy(out.container(), LINX_CRTP_CONST_DERIVED.container());
    return out;
  }

  /**
   * @brief Copy.
   */
  TDerived operator+() const
  {
    // FIXME if container use_count() <= 1, return this to optimize out temporary objects
    return copy_as(compose_label("copy", LINX_CRTP_CONST_DERIVED));
  }

  /**
   * @brief Compute the opposite.
   */
  TDerived operator-() const
  {
    TDerived res = copy_as(compose_label("negate", LINX_CRTP_CONST_DERIVED));
    res.apply("-", std::negate());
    return res;
  }
};

/// @cond

/**
 * @ingroup pixelwise
 * @ingroup mixins
 * @brief `VectorArithmetic` specialization.
 * @satisfies{VectorArithmetic}
 */
template <typename T, typename TDerived>
struct ArithmeticMixin<VectorArithmetic, T, TDerived> : ArithmeticMixin<void, T, TDerived> {
  /// @{
  /// @group_modifiers

  LINX_OPERATOR(+, Plus)

  LINX_OPERATOR(-, Minus)

  LINX_SCALAR_OPERATOR(*, Multiplies)

  LINX_SCALAR_OPERATOR(/, Divides)

  LINX_SCALAR_OPERATOR(%, Modulus)

  /**
   * @brief ++V
   */
  const TDerived& operator++() const
  {
    return LINX_CRTP_CONST_DERIVED.apply("++", Plus(1));
  }

  /**
   * @brief --V
   */
  const TDerived& operator--() const
  {
    return LINX_CRTP_CONST_DERIVED.apply("--", Minus(1));
  }

  /// @}
};

/**
 * @ingroup pixelwise
 * @ingroup mixins
 * @brief `EuclidArithmetic` specialization.
 * @satisfies{EuclidArithmetic}
 */
template <typename T, typename TDerived>
struct ArithmeticMixin<EuclidArithmetic, T, TDerived> : ArithmeticMixin<void, T, TDerived> {
  /// @{
  /// @group_modifiers

  LINX_OPERATOR(+, Plus)

  LINX_OPERATOR(-, Minus)

  LINX_OPERATOR(*, Multiplies)

  LINX_OPERATOR(/, Divides)

  LINX_OPERATOR(%, Modulus)

  /**
   * @brief ++V
   */
  const TDerived& operator++() const
  {
    return LINX_CRTP_CONST_DERIVED.apply("++", Plus(1));
  }

  /**
   * @brief --V
   */
  const TDerived& operator--() const
  {
    return LINX_CRTP_CONST_DERIVED.apply("--", Minus(1));
  }

  /// @}
};

/// @endcond

#undef LINX_SCALAR_OPERATOR_INPLACE
#undef LINX_SCALAR_OPERATOR_NEWINSTANCE
#undef LINX_VECTOR_OPERATOR_INPLACE
#undef LINX_VECTOR_OPERATOR_NEWINSTANCE
#undef LINX_SCALAR_OPERATOR
#undef LINX_VECTOR_OPERATOR
#undef LINX_OPERATOR

} // namespace Linx

#endif
