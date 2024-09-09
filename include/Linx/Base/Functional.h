// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXBASE_FUNCTIONAL_H
#define _LINXBASE_FUNCTIONAL_H

#include <Kokkos_Core.hpp>

namespace Linx {

/**
 * @brief Functor which forwards its argument.
 */
struct Forward {
  KOKKOS_INLINE_FUNCTION constexpr decltype(auto) operator()(auto&& value) const
  {
    return LINX_FORWARD(value);
  }
};

/**
 * @brief Functor which always returns the same value.
 */
template <typename T>
struct Constant {
  using value_type = T;
  T value;
  KOKKOS_INLINE_FUNCTION constexpr const T& operator()(auto&&...) const
  {
    return value;
  }
};

#define LINX_DEFINE_BINARY_OPERATOR(op, Func) \
  template <typename TOut = Forward, typename TLhs = Forward, typename TRhs = Forward> \
  struct Func; \
\
  template <typename TOut, typename TRhs> \
  struct Func<TOut, Forward, TRhs> { \
    TRhs value; \
    Func(TRhs v) : value {v} {} \
    KOKKOS_INLINE_FUNCTION TOut operator()(const auto& lhs) const \
    { \
      return lhs op value; \
    } \
  }; \
\
  template <typename TOut, typename TLhs> \
  struct Func<TOut, TLhs, Forward> { \
    TLhs value; \
    Func(TLhs v) : value {v} {} \
    KOKKOS_INLINE_FUNCTION TOut operator()(const auto& rhs) const \
    { \
      return value op rhs; \
    } \
  }; \
\
  template <typename TOut> \
  struct Func<TOut, Forward, Forward> { \
    KOKKOS_INLINE_FUNCTION TOut operator()(const auto& lhs, const auto& rhs) const \
    { \
      return lhs op rhs; \
    } \
  }; \
\
  template <> \
  struct Func<Forward, Forward, Forward> { \
    KOKKOS_INLINE_FUNCTION auto operator()(const auto& lhs, const auto& rhs) const \
    { \
      return lhs op rhs; \
    } \
  }; \
\
  Func()->Func<Forward, Forward, Forward>; \
  template <typename T> \
  Func(T) -> Func<T, Forward, T>;

#define LINX_DEFINE_MONOID(op, Func, identity) \
  LINX_DEFINE_BINARY_OPERATOR(op, Func) \
\
  template <typename T, typename TOut, typename TLhs, typename TRhs> \
  KOKKOS_INLINE_FUNCTION T identity_element(const Func<TOut, TLhs, TRhs>&) \
  { \
    return identity; \
  }

LINX_DEFINE_MONOID(+, Plus, T {})
LINX_DEFINE_BINARY_OPERATOR(-, Minus)
LINX_DEFINE_MONOID(*, Multiplies, T {1})
LINX_DEFINE_BINARY_OPERATOR(/, Divides)
LINX_DEFINE_BINARY_OPERATOR(%, Modulus)
LINX_DEFINE_BINARY_OPERATOR(==, Equal)
LINX_DEFINE_BINARY_OPERATOR(!=, NotEqual)
LINX_DEFINE_MONOID(&&, And, true)
LINX_DEFINE_MONOID(||, Or, false)

#undef LINX_DEFINE_BINARY_OPERATOR
#undef LINX_DEFINE_MONOID

/**
 * @brief Compute the absolute value of an integral power.
 * @see `Abspow`
 */
template <int P, typename T>
KOKKOS_INLINE_FUNCTION constexpr T abspow(T x)
{
  if constexpr (P == 0) {
    return bool(x);
  }
  if constexpr (P == 1) {
    return x >= 0 ? x : -x;
  }
  if constexpr (P == 2) {
    return x * x;
  }
  if constexpr (P > 2) {
    return x * x * abspow<P - 2>(x);
  }
}

/**
 * @brief Functor which returns `abspow()`.
 */
template <int P, typename T>
struct Abspow {
  using value_type = T;
  KOKKOS_INLINE_FUNCTION constexpr T operator()(T lhs) const
  {
    return abspow<P>(lhs);
  }
  KOKKOS_INLINE_FUNCTION constexpr T operator()(T lhs, T rhs) const
  {
    return abspow<P>(rhs - lhs);
  }
};

struct IsNan {
  KOKKOS_INLINE_FUNCTION constexpr bool operator()(const auto& value) const
  {
    return value != value;
  }
};

} // namespace Linx

#endif
