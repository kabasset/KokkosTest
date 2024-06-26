// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXDATA_TRAITS_H
#define _LINXDATA_TRAITS_H

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

} // namespace Linx

#endif
