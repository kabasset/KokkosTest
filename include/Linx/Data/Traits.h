// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXDATA_TRAITS_H
#define _LINXDATA_TRAITS_H

#include "Linx/Base/TypeUtils.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_DynRankView.hpp>

namespace Linx {

template <typename T, int N>
struct ContainerTraits {
  using Vector = Kokkos::View<T[N]>;
  using Image = Kokkos::View<typename ContainerTraits<T, N - 1>::Image::data_type*>;
  // FIXME fall back to Raster for N > 8
  // FIXME use DynRankView for N = -1 & dimension < 8
  // FIXME fall back to Raster for N = -1 & dimension > 7
};

template <typename T>
struct ContainerTraits<T, 1> {
  using Vector = Kokkos::View<T[1]>;
  using Image = Kokkos::View<T*>;
};

template <typename T>
struct ContainerTraits<T, 0> {
  using Vector = Kokkos::View<T*>;
  using Image = Kokkos::View<T*>;
};

template <typename T>
struct ContainerTraits<T, -1> {
  using Vector = Kokkos::View<T*>;
  using Image = Kokkos::DynRankView<T>;
};

template <typename T>
constexpr bool is_range()
{
  return false; // FIXME
}

template <typename T0, typename... Ts>
struct Pack {
  using First = T0;
  using Last = typename decltype((std::type_identity<Ts> {}, ...))::type;
};

/// @cond
namespace Internal {

template <typename TFunc, typename TTuple, std::size_t... Is>
KOKKOS_FORCEINLINE_FUNCTION auto apply_tuple_last_first(TFunc&& func, TTuple&& tuple, std::index_sequence<Is...>)
{
  constexpr auto N = std::tuple_size_v<TTuple>;
  return LINX_FORWARD(func)(std::get<N - 1>(LINX_FORWARD(tuple)), std::get<Is>(LINX_FORWARD(tuple))...);
}

template <typename TProj, typename TRed, typename TTuple, std::size_t... Is>
KOKKOS_FORCEINLINE_FUNCTION auto
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
KOKKOS_FORCEINLINE_FUNCTION auto apply_last_first(TFunc&& func, Ts&&... args)
{
  return Internal::apply_tuple_last_first(
      LINX_FORWARD(func),
      std::forward_as_tuple(LINX_FORWARD(args)...),
      std::make_index_sequence<sizeof...(Ts) - 1> {});
}

template <typename TProj, typename TRed, typename... Ts>
KOKKOS_FORCEINLINE_FUNCTION auto project_reduce_to(TProj&& projection, TRed&& reducer, Ts&&... args)
{
  Internal::tuple_project_reduce_to(
      LINX_FORWARD(projection),
      LINX_FORWARD(reducer),
      std::forward_as_tuple(LINX_FORWARD(args)...),
      std::make_index_sequence<sizeof...(Ts) - 1> {});
}

} // namespace Linx

#endif
