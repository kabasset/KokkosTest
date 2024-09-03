// @copyright 2022-2024, Antoine Basset (CNES)
// This file is part of Linx <github.com/kabasset/Linx>
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXBASE_MIXINS_ARITHMETIC_H
#define _LINXBASE_MIXINS_ARITHMETIC_H

#include "Linx/Base/Types.h" // LINX_FORWARD

#include <Kokkos_Core.hpp>
#include <functional>
#include <type_traits>

namespace Linx {

#define LINX_DECLARE_OPERATOR_FUNCTOR(op, Func) \
  template <typename T> \
  struct Func { \
    T value; \
    constexpr T operator()(const auto& lhs) const \
    { \
      return lhs op value; \
    } \
  }; \
  \
  template <> \
  struct Func<void> { \
    constexpr auto operator()(const auto& lhs, const auto& rhs) const \
    { \
      return lhs op rhs; \
    } \
  }; \
  \
  Func() -> Func<void>; \
  template <typename T> Func(T) -> Func<T>;

LINX_DECLARE_OPERATOR_FUNCTOR(+, Plus)
LINX_DECLARE_OPERATOR_FUNCTOR(-, Minus)
LINX_DECLARE_OPERATOR_FUNCTOR(*, Multiplies)
LINX_DECLARE_OPERATOR_FUNCTOR(/, Divides)
LINX_DECLARE_OPERATOR_FUNCTOR(%, Modulus)

#define LINX_VECTOR_OPERATOR_INPLACE(op, Func) \
  const TDerived& operator op##=(const TDerived& rhs) const \
  { \
    return LINX_CRTP_CONST_DERIVED.apply( \
        compose_label(#op, *this, rhs), \
        Func(), \
        rhs); \
  }

#define LINX_SCALAR_OPERATOR_INPLACE(op, Func) \
  const TDerived& operator op##=(const T& rhs) const \
  { \
    return LINX_CRTP_CONST_DERIVED.apply( \
      compose_label(#op, *this, rhs), \
      Func(rhs)); \
  }

#define LINX_OPERATOR_NEWINSTANCE(op) \
  friend TDerived operator op(const TDerived& lhs, const auto& rhs) \
  { \
    TDerived out = +lhs.copy(#op); \
    out op##= rhs; \
    return out; \
  }

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

  TDerived copy(const std::string& label) const
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
    return copy();
  }

  /**
   * @brief Compute the opposite.
   */
  TDerived operator-() const
  {
    TDerived res = copy("negate");
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

  LINX_VECTOR_OPERATOR_INPLACE(+, Plus) ///< V += U
  LINX_SCALAR_OPERATOR_INPLACE(+, Plus) ///< V += a
  LINX_OPERATOR_NEWINSTANCE(+) ///< W = U + V, V = U + a

  LINX_VECTOR_OPERATOR_INPLACE(-, Minus) ///< V -= U
  LINX_SCALAR_OPERATOR_INPLACE(-, Minus) ///< V -= a
  LINX_OPERATOR_NEWINSTANCE(-) ///< W = U - V, V = U - a

  LINX_SCALAR_OPERATOR_INPLACE(*, Multiplies) ///< V *= a
  LINX_OPERATOR_NEWINSTANCE(*) ///< V = U * a

  LINX_SCALAR_OPERATOR_INPLACE(/, Divides) ///< V /= a
  LINX_OPERATOR_NEWINSTANCE(/) ///< V = U / a

  LINX_SCALAR_OPERATOR_INPLACE(%, Modulus) ///< V %= a
  LINX_OPERATOR_NEWINSTANCE(%) ///< V = U % a

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

  LINX_VECTOR_OPERATOR_INPLACE(+, Plus) ///< V += U
  LINX_SCALAR_OPERATOR_INPLACE(+, Plus) ///< V += a
  LINX_OPERATOR_NEWINSTANCE(+) ///< W = U + V, V = U + a

  LINX_VECTOR_OPERATOR_INPLACE(-, Minus) ///< V -= U
  LINX_SCALAR_OPERATOR_INPLACE(-, Minus) ///< V -= a
  LINX_OPERATOR_NEWINSTANCE(-) ///< W = U - V, V = U - a

  LINX_VECTOR_OPERATOR_INPLACE(*, Multiplies) ///< V *= U
  LINX_SCALAR_OPERATOR_INPLACE(*, Multiplies) ///< V *= a
  LINX_OPERATOR_NEWINSTANCE(*) ///< W = U * V, V = U * a

  LINX_VECTOR_OPERATOR_INPLACE(/, Divides) ///< V /= U
  LINX_SCALAR_OPERATOR_INPLACE(/, Divides) ///< V /= a
  LINX_OPERATOR_NEWINSTANCE(/) ///< W = U / V, V = U / a

  LINX_VECTOR_OPERATOR_INPLACE(%, Modulus) ///< V %= U
  LINX_SCALAR_OPERATOR_INPLACE(%, Modulus) ///< V %= a
  LINX_OPERATOR_NEWINSTANCE(%) ///< W = U % V, V = U % a

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

#undef LINX_VECTOR_OPERATOR_INPLACE
#undef LINX_SCALAR_OPERATOR_INPLACE
#undef LINX_OPERATOR_NEWINSTANCE

} // namespace Linx

#endif
