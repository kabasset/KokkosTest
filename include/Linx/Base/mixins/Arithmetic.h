// @copyright 2022-2024, Antoine Basset (CNES)
// This file is part of Linx <github.com/kabasset/Linx>
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXBASE_MIXINS_ARITHMETIC_H
#define _LINXBASE_MIXINS_ARITHMETIC_H

#include "Linx/Base/Types.h" // LINX_FORWARD

#include <Kokkos_Core.hpp>
#include <boost/operators.hpp>
#include <functional>
#include <type_traits>

namespace Linx {

// FIXME handle const instances

#define LINX_VECTOR_OPERATOR_INPLACE(op) \
  TDerived& operator op##=(const TDerived& rhs) \
  { \
    LINX_CRTP_DERIVED.apply( \
        #op, \
        KOKKOS_LAMBDA(auto e, auto f) { return e op f; }, \
        rhs); \
    return LINX_CRTP_DERIVED; \
  }

#define LINX_SCALAR_OPERATOR_INPLACE(op) \
  TDerived& operator op##=(const T& rhs) \
  { \
    LINX_CRTP_DERIVED.apply( \
        #op, \
        KOKKOS_LAMBDA(auto e) { return e op rhs; }); \
    return LINX_CRTP_DERIVED; \
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
 * @tparam T The contained element value type
 * @tparam TDerived The container which inherits this class
 * 
 * @tspecialization{VectorArithmetic}
 * @tspecialization{EuclidArithmetic}
 */
template <typename TSpecs, typename T, typename TDerived>
struct ArithmeticMixin {}; // Empy base class for void

/// @cond

/**
 * @ingroup pixelwise
 * @ingroup mixins
 * @brief `VectorArithmetic` specialization.
 * @satisfies{VectorArithmetic}
 */
template <typename T, typename TDerived>
struct ArithmeticMixin<VectorArithmetic, T, TDerived> :
    boost::additive<TDerived>,
    boost::additive<TDerived, T>,
    boost::subtractable2_left<TDerived, T>,
    boost::unit_steppable<TDerived>,
    boost::multiplicative<TDerived, T>,
    boost::modable<TDerived>,
    boost::modable<TDerived, T> {
  /// @{
  /// @group_modifiers

  LINX_VECTOR_OPERATOR_INPLACE(+) ///< V += U
  LINX_SCALAR_OPERATOR_INPLACE(+) ///< V += a

  LINX_VECTOR_OPERATOR_INPLACE(-) ///< V -= U
  LINX_SCALAR_OPERATOR_INPLACE(-) ///< V -= a

  LINX_SCALAR_OPERATOR_INPLACE(*) ///< V *= a

  LINX_SCALAR_OPERATOR_INPLACE(/) ///< V /= a

  LINX_VECTOR_OPERATOR_INPLACE(%) ///< V %= U
  LINX_SCALAR_OPERATOR_INPLACE(%) ///< V %= a

  /**
   * @brief ++V
   */
  TDerived& operator++()
  {
    LINX_CRTP_DERIVED.apply(
        "++",
        KOKKOS_LAMBDA(auto rhs) { return ++rhs; });
    return LINX_CRTP_DERIVED;
  }

  /**
   * @brief --V
   */
  TDerived& operator--()
  {
    LINX_CRTP_DERIVED.apply(
        "--",
        KOKKOS_LAMBDA(auto rhs) { return --rhs; });
    return LINX_CRTP_DERIVED;
  }

  /// @group_operations

  /**
   * @brief Copy.
   */
  TDerived operator+() const
  {
    return Kokkos::deep_copy(*this); // FIXME deep_copy?
  }

  /**
   * @brief Compute the opposite.
   */
  TDerived operator-() const
  {
    TDerived res = +(*this);
    res.apply(
        "-",
        KOKKOS_LAMBDA(auto r) { return -r; });
    return res;
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
struct ArithmeticMixin<EuclidArithmetic, T, TDerived> :
    boost::additive<TDerived>,
    boost::additive<TDerived, T>,
    boost::subtractable2_left<TDerived, T>,
    boost::unit_steppable<TDerived>,
    boost::multiplicative<TDerived>,
    boost::multiplicative<TDerived, T>,
    boost::modable<TDerived>,
    boost::modable<TDerived, T> {
  /// @{
  /// @group_modifiers

  LINX_VECTOR_OPERATOR_INPLACE(+) ///< V += U
  LINX_SCALAR_OPERATOR_INPLACE(+) ///< V += a

  LINX_VECTOR_OPERATOR_INPLACE(-) ///< V -= U
  LINX_SCALAR_OPERATOR_INPLACE(-) ///< V -= a

  LINX_VECTOR_OPERATOR_INPLACE(*) ///< V *= U
  LINX_SCALAR_OPERATOR_INPLACE(*) ///< V *= a

  LINX_VECTOR_OPERATOR_INPLACE(/) ///< V /= U
  LINX_SCALAR_OPERATOR_INPLACE(/) ///< V /= a

  LINX_VECTOR_OPERATOR_INPLACE(%) ///< V %= U
  LINX_SCALAR_OPERATOR_INPLACE(%) ///< V %= a

  /**
   * @brief ++V
   */
  TDerived& operator++()
  {
    LINX_CRTP_DERIVED.apply(
        "++",
        KOKKOS_LAMBDA(auto rhs) { return ++rhs; });
    return LINX_CRTP_DERIVED;
  }

  /**
   * @brief --V
   */
  TDerived& operator--()
  {
    LINX_CRTP_DERIVED.apply(
        "--",
        KOKKOS_LAMBDA(auto rhs) { return --rhs; });
    return LINX_CRTP_DERIVED;
  }

  /// @group_operations

  /**
   * @brief Deep copy.
   */
  TDerived operator+() const
  {
    TDerived out(LINX_CRTP_CONST_DERIVED.label(), LINX_CRTP_CONST_DERIVED.shape());
    Kokkos::deep_copy(out.container(), LINX_CRTP_CONST_DERIVED.container());
    return out;
  }

  /**
   * @brief Compute the opposite.
   */
  TDerived operator-() const
  {
    TDerived res = +(*this);
    res.apply(
        "-",
        KOKKOS_LAMBDA(auto r) { return -r; });
    return res;
  }

  /// @}
};

/// @endcond

#undef LINX_VECTOR_OPERATOR_INPLACE
#undef LINX_SCALAR_OPERATOR_INPLACE

} // namespace Linx

#endif
