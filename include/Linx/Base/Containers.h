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
template <typename T, int N>
struct ContainerTraits {
  using Vector = Kokkos::View<T[N]>;
  using Image = Kokkos::View<typename ContainerTraits<T, N - 1>::Image::data_type*>;
  // FIXME fall back to Raster for N > 8
  // FIXME fall back to Raster for N = -1 & dimension > 7
};

/**
 * @brief Rank-1 specialization.
 */
template <typename T>
struct ContainerTraits<T, 1> {
  using Vector = Kokkos::View<T[1]>;
  using Image = Kokkos::View<T*>;
};

/**
 * @brief Rank-0 specialization.
 */
template <typename T>
struct ContainerTraits<T, 0> {
  using Vector = Kokkos::View<T*>;
  using Image = Kokkos::View<T*>;
};

/**
 * @brief Dynamic rank specialization.
 */
template <typename T>
struct ContainerTraits<T, -1> {
  using Vector = Kokkos::View<T*>;
  using Image = Kokkos::DynRankView<T>;
};

} // namespace Linx

#endif
