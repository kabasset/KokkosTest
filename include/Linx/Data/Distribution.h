// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXDATA_DISTRIBUTION_H
#define _LINXDATA_DISTRIBUTION_H

#include "Linx/Data/Sequence.h"

namespace Linx {

/// @cond
namespace Internal {

template <typename TIn, typename TBins, typename TOut>
struct HistogramBinFinder {
  TIn m_in;
  TBins m_bins;
  TOut m_out;
  int m_bin_count;
  
  HistogramBinFinder(TIn in, TBins bins, TOut out) : m_in(LINX_MOVE(in)), m_bins(LINX_MOVE(bins)), m_out(LINX_MOVE(out)), m_bin_count(m_out.size())
  {}
  
  KOKKOS_INLINE_FUNCTION void operator()(auto... is) const
  {
    auto value = m_in(is...);
    if (value >= m_bins[0] && value < m_bins[m_bin_count]) {
      int index = 0;
      while (index < m_bin_count && value >= m_bins[index + 1]) {
        ++index;
      };
      ++m_out(index);
    }
  }
};

}
/// @endcond

template <typename TIn, typename TBins, typename TOut>
void histogram_to(const TIn& in, const TBins& bins, TOut& out)
{
  const auto& atomic_out = as_atomic(out.container());
  const auto& readonly_in = as_readonly(in);

  for_each(
      "histogram()",
      in.domain(),
      Internal::HistogramBinFinder(readonly_in, bins, atomic_out));
  Kokkos::fence();
}

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
  Sequence<TOut, N> out(compose_label("histogram", in), bins.size() - 1);
  histogram_to(in, bins, out);
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