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

#define LINX_DEFINE_BINARY_OPERATOR(Func, out) \
  template <typename TOut = Forward, typename TLhs = Forward, typename TRhs = Forward> \
  struct Func; \
\
  template <typename TOut, typename TRhs> \
  struct Func<TOut, Forward, TRhs> { \
    TRhs rhs; \
    Func(TRhs value) : rhs {value} {} \
    KOKKOS_INLINE_FUNCTION TOut operator()(const auto& lhs) const \
    { \
      return out; \
    } \
  }; \
\
  template <typename TOut, typename TLhs> \
  struct Func<TOut, TLhs, Forward> { \
    TLhs lhs; \
    Func(TLhs value) : lhs {value} {} \
    KOKKOS_INLINE_FUNCTION TOut operator()(const auto& rhs) const \
    { \
      return out; \
    } \
  }; \
\
  template <typename TOut> \
  struct Func<TOut, Forward, Forward> { \
    KOKKOS_INLINE_FUNCTION TOut operator()(const auto& lhs, const auto& rhs) const \
    { \
      return out; \
    } \
  }; \
\
  template <> \
  struct Func<Forward, Forward, Forward> { \
    KOKKOS_INLINE_FUNCTION auto operator()(const auto& lhs, const auto& rhs) const \
    { \
      return out; \
    } \
  }; \
\
  Func()->Func<Forward, Forward, Forward>; \
  template <typename T> \
  Func(T) -> Func<T, Forward, T>;

#define LINX_DEFINE_MONOID(Func, out, identity) \
  LINX_DEFINE_BINARY_OPERATOR(Func, out) \
\
  template <typename T, typename TOut, typename TLhs, typename TRhs> \
  KOKKOS_INLINE_FUNCTION T identity_element(const Func<TOut, TLhs, TRhs>&) \
  { \
    return identity; \
  }

LINX_DEFINE_MONOID(Plus, (lhs + rhs), T {})
LINX_DEFINE_BINARY_OPERATOR(Minus, (lhs - rhs))
LINX_DEFINE_MONOID(Multiplies, (lhs * rhs), T {1})
LINX_DEFINE_BINARY_OPERATOR(Divides, (lhs / rhs))
LINX_DEFINE_BINARY_OPERATOR(Modulus, (lhs % rhs))
LINX_DEFINE_BINARY_OPERATOR(Equal, (lhs == rhs))
LINX_DEFINE_BINARY_OPERATOR(NotEqual, (lhs != rhs))
LINX_DEFINE_MONOID(And, (lhs && rhs), true)
LINX_DEFINE_MONOID(Or, (lhs || rhs), false)
LINX_DEFINE_MONOID(Min, std::min(lhs, rhs), std::numeric_limits<T>::max())
LINX_DEFINE_MONOID(Max, std::max(lhs, rhs), std::numeric_limits<T>::lowest())

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
template <int P, typename T> // FIXME accept void
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

/**
 * @brief Functor which returns `true` iff `value != value`.
 */
struct IsNan {
  KOKKOS_INLINE_FUNCTION constexpr bool operator()(const auto& value) const
  {
    return value != value;
  }
};

} // namespace Linx

#endif
