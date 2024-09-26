// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXDATA_TILING_H
#define _LINXDATA_TILING_H

#include "Linx/Data/Image.h"
#include "Linx/Data/Line.h"
#include "Linx/Data/Patch.h"
#include "Linx/Data/Sequence.h"

#include <Kokkos_Core.hpp>

namespace Linx {

template <int I, typename TIn>
using Profile = Patch<TIn, Line<int, I, TIn::Rank>>;

/**
 * @brief Get the collection of all the profiles of an image along a given axis.
 * @tparam I The profile axis.
 * 
 * A profile is a line-based patch.
 * 
 * \code
 * for (const auto& column : profiles<1>(image)) {
 *   // Do something with the column
 * }
 * \endcode
 * 
 * @see `rows()`
 */
template <int I, typename TIn>
std::vector<Profile<I, TIn>> profiles(const TIn& in)
{
  static constexpr int N = TIn::Rank;
  using Domain = Line<int, I, N>;
  const auto& domain = in.domain();
  const auto& start = domain.start();
  auto shape = domain.shape();
  shape[I] = 1;
  auto stop = domain.stop(I);
  auto size = product(shape);
  std::vector<Profile<I, TIn>> vec;
  vec.reserve(size);
  for (int i = 0; i < size; ++i) {
    vec.emplace_back(in, Domain(+start, stop)); // Shallow-copy is not enough
  }
  Raster<Profile<I, TIn>, N> out(Wrap(vec.data()), shape); // FIXME owning raster somehow?
  Linx::for_each<Kokkos::Serial>(
      "profiles",
      Box<int, N> {Position<int, N> {}, shape}, // FIXME handle potential offset
      [&](auto... is) {
        out(is...).shift(is...);
      }); // This is serial for now, no KOKKOS_LAMBDA needed
  return vec; // FIXME return out somehow?
}

/**
 * @brief Get the collection of all the rows of an image.
 * 
 * \code
 * for (const auto& row : rows(image)) {
 *   // Do something with the row
 * }
 * \endcode
 * 
 * @see `profiles()`
 */
template <typename TIn>
auto rows(const TIn& in)
{
  return profiles<0>(in);
}

} // namespace Linx

#endif
