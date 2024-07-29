// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXDATA_DISTRIBUTION_H
#define _LINXDATA_DISTRIBUTION_H

#include "Linx/Data/Sequence.h"

namespace Linx {

/**
 * @brief Compute the histogram of a container.
 * 
 * @param in The container
 * @param bins The bin bounds
 * 
 * The bins are half-open intervals, such that `histogram[i]` is the number of values
 * both greater or equal to `bins[i]` and strictly less than `bin[i+1]`.
 * Values outside the bins are not counted.
 */
template <typename TOut = int, typename TBins>
auto histogram(const auto& in, const TBins& bins)
{
  constexpr auto N = std::max(TBins::Rank - 1, -1);
  const auto bin_count = bins.size() - 1;
  Sequence<TOut, N> out(compose_label("histogram", in), bin_count);
  const auto& atomic_out = as_atomic(out.container());
  const auto& readonly_in = as_readonly(in);

  for_each(
      "histogram()",
      in.domain(),
      KOKKOS_LAMBDA(auto... is) {
        auto value = readonly_in(is...);
        if (value >= bins[0] && value < bins[bin_count]) {
          std::size_t index = 0;
          while (index < bin_count && value >= bins[index + 1]) {
            ++index;
          };
          ++atomic_out(index);
        }
      });
  Kokkos::fence();
  return out;
}

/**
 * @relatesalo RangeMixin
 * @brief Create a `DataDistribution` from the container.
 */
// template <typename TRange>
// DataDistribution<typename TRange::value_type> distribution(const TRange& in)
// {
//   return DataDistribution<typename TRange::value_type>(in);
// }

} // namespace Linx

#endif