// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXDATA_RASTER_H
#define _LINXDATA_RASTER_H

#include "Linx/Base/mixins/Range.h"
#include "Linx/Data/Image.h"

namespace Linx {

template <typename T, int N, typename TContainer>
  requires(is_contiguous<TContainer>())
auto begin(const Image<T, N, TContainer>& image)
{
  return image.data();
}

template <typename T, int N, typename TContainer>
  requires(is_contiguous<TContainer>())
auto end(const Image<T, N, TContainer>& image)
{
  return begin(image) + image.size();
}

/**
 * @brief Contiguous image with row-major ordering.
 */
template <typename T, int N> // FIXME foward TArgs to DefaultContainers?
using Raster = Image<T, N, typename DefaultContainer<T, N, Kokkos::LayoutLeft>::Image>;

/**
 * @brief Raster on host for legacy software.
 */
template <typename T, int N>
using HostRaster = Image<T, N, typename DefaultContainer<T, N, Kokkos::LayoutLeft, Kokkos::HostSpace>::Image>;

} // namespace Linx

#endif
