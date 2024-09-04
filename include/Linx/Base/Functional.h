// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXBASE_FUNCTIONAL_H
#define _LINXBASE_FUNCTIONAL_H

#include <Kokkos_Core.hpp>

namespace Linx {

#define LINX_DECLARE_OPERATOR_FUNCTOR(op, Func) \
  template <typename T> \
  struct Func { \
    using value_type = T; \
    T value; \
    KOKKOS_INLINE_FUNCTION constexpr T operator()(const auto& lhs) const \
    { \
      return lhs op value; \
    } \
  }; \
\
  template <> \
  struct Func<void> { \
    KOKKOS_INLINE_FUNCTION constexpr auto operator()(const auto& lhs, const auto& rhs) const \
    { \
      return lhs op rhs; \
    } \
  }; \
\
  Func()->Func<void>; \
  template <typename T> \
  Func(T) -> Func<T>;

LINX_DECLARE_OPERATOR_FUNCTOR(+, Plus)
LINX_DECLARE_OPERATOR_FUNCTOR(-, Minus)
LINX_DECLARE_OPERATOR_FUNCTOR(*, Multiplies)
LINX_DECLARE_OPERATOR_FUNCTOR(/, Divides)
LINX_DECLARE_OPERATOR_FUNCTOR(%, Modulus)

#define LINX_DECLARE_FUNCTOR(op, Func) \
  struct Func { \
    template <typename T> \
    KOKKOS_INLINE_FUNCTION constexpr T operator()(const auto&... ins) const \
    { \
      return op(ins...); \
    } \
  };

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

} // namespace Linx

#endif
