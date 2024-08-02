// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXBASE_PACKS_H
#define _LINXBASE_PACKS_H

#include "Linx/Base/Types.h"

#include <tuple>
#include <type_traits>

namespace Linx {

/**
 * @brief Pack traits.
 */
template <typename T0, typename... Ts>
struct PackTraits {
  using First = T0;
  using Last = typename decltype((std::type_identity<Ts> {}, ...))::type;
};

template <typename T>
struct PackTraits<T> {
  using First = T;
  using Last = T;
};

/// @cond
namespace Internal {

template <typename TFunc, typename TTuple, std::size_t... Is>
KOKKOS_INLINE_FUNCTION auto apply_tuple_last_first(TFunc&& func, TTuple&& tuple, std::index_sequence<Is...>)
{
  constexpr auto N = std::tuple_size_v<TTuple>;
  return LINX_FORWARD(func)(std::get<N - 1>(LINX_FORWARD(tuple)), std::get<Is>(LINX_FORWARD(tuple))...);
}

template <typename TProj, typename TRed, typename TTuple, std::size_t... Is>
KOKKOS_INLINE_FUNCTION void
tuple_project_reduce_to(TProj&& projection, TRed&& reducer, TTuple&& tuple, std::index_sequence<Is...>)
{
  constexpr auto N = std::tuple_size_v<TTuple>;
  LINX_FORWARD(reducer).join(
      std::get<N - 1>(LINX_FORWARD(tuple)),
      LINX_FORWARD(projection)(std::get<Is>(LINX_FORWARD(tuple))...));
}

} // namespace Internal
/// @endcond

template <typename TFunc, typename... Ts>
KOKKOS_INLINE_FUNCTION auto apply_last_first(TFunc&& func, Ts&&... args)
{
  return Internal::apply_tuple_last_first(
      LINX_FORWARD(func),
      std::forward_as_tuple(LINX_FORWARD(args)...),
      std::make_index_sequence<sizeof...(Ts) - 1> {});
}

template <typename TProj, typename TRed, typename... Ts>
KOKKOS_INLINE_FUNCTION void project_reduce_to(TProj&& projection, TRed&& reducer, Ts&&... args)
{
  Internal::tuple_project_reduce_to(
      LINX_FORWARD(projection),
      LINX_FORWARD(reducer),
      std::forward_as_tuple(LINX_FORWARD(args)...),
      std::make_index_sequence<sizeof...(Ts) - 1> {});
}

} // namespace Linx

#endif
