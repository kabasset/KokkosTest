// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXBASE_CONTAINERS_H
#define _LINXBASE_CONTAINERS_H

#include "Linx/Base/Types.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_DynRankView.hpp>

namespace Linx {

/**
 * @brief Mapping between type and rank and container classes.
 */
template <typename T, int N, typename... TArgs>
struct DefaultContainer {
  using Sequence = Kokkos::View<T[N], TArgs...>;
  using Image = Kokkos::View<typename DefaultContainer<T, N - 1>::Image::data_type*, TArgs...>;
  // FIXME fall back to Raster for N > 8
  // FIXME fall back to Raster for N = -1 & dimension > 7
};

/**
 * @brief Rank-1 specialization.
 */
template <typename T, typename... TArgs>
struct DefaultContainer<T, 1, TArgs...> {
  using Sequence = Kokkos::View<T[1], TArgs...>;
  using Image = Kokkos::View<T*, TArgs...>;
};

/**
 * @brief Rank-0 specialization.
 */
template <typename T, typename... TArgs>
struct DefaultContainer<T, 0, TArgs...> {
  using Sequence = Kokkos::View<T*, TArgs...>;
  using Image = Kokkos::View<T*, TArgs...>;
};

/**
 * @brief Dynamic rank specialization.
 */
template <typename T, typename... TArgs>
struct DefaultContainer<T, -1, TArgs...> {
  using Sequence = Kokkos::View<T*, TArgs...>;
  using Image = Kokkos::DynRankView<T, TArgs...>;
};

/**
 * @brief Traits to rebind containers.
 */
template <typename T>
struct Rebind {
  using AsReadonly = const T;
};

/**
 * @brief Pointer specialization.
 */
template <typename T>
struct Rebind<T*> {
  using AsReadonly = typename Rebind<T>::AsReadonly*;
};

/**
 * @brief `View` specialization.
 */
template <typename T, typename... TArgs>
struct Rebind<Kokkos::View<T, TArgs...>> {
  using AsReadonly = Kokkos::View<typename Rebind<T>::AsReadonly, TArgs...>;
};

/**
 * @brief `DynRankView` specialization.
 */
template <typename T, typename... TArgs>
struct Rebind<Kokkos::DynRankView<T, TArgs...>> {
  using AsReadonly = Kokkos::DynRankView<typename Rebind<T>::AsReadonly, TArgs...>;
};

} // namespace Linx

#endif
