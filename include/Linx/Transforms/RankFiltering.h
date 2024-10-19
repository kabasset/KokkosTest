// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXTRANSFORMS_RANKFILTERING_H
#define _LINXTRANSFORMS_RANKFILTERING_H

#include "Linx/Base/Algorithm.h"
#include "Linx/Data/Image.h"
#include "Linx/Data/Sequence.h"

#include <concepts>
#include <string>

namespace Linx {

namespace Impl {

template <typename TIn, typename TOut>
struct Median { // FIXME OddMedian and EvenMedian
  Median(const Sequence<std::ptrdiff_t, -1>& offsets, const TIn& in, const TOut& out) :
      m_offsets(offsets), m_neighbors(m_offsets.size()), m_in(in), m_out(out), m_size(m_offsets.size())
  {}

  Median(const Median& rhs) :
      m_offsets(rhs.m_offsets), m_neighbors(m_offsets.size()), m_in(rhs.m_in), m_out(rhs.m_out), m_size(rhs.m_size)
  {}

  KOKKOS_INLINE_FUNCTION void operator()(const std::integral auto&... is) const
  {
    auto in_ptr = &m_in(is...);
    for (std::size_t i = 0; i < m_size; ++i) {
      m_neighbors[i] = in_ptr[m_offsets[i]];
    }
    m_out(is...) = median(m_neighbors);
  }

  Sequence<std::ptrdiff_t, -1> m_offsets;
  Sequence<typename TIn::element_type, -1> m_neighbors;
  TIn m_in;
  TOut m_out;
  std::size_t m_size;
};

} // namespace Impl

template <typename TIn, typename TStrel>
Sequence<std::ptrdiff_t, -1> compute_offsets(const TIn& in, const TStrel& strel)
{
  Sequence<std::ptrdiff_t, -1> offsets("offsets", strel.size());
  auto offsets_on_host = on_host(offsets);
  auto index = std::make_shared<Index>(0);
  auto data = &in[Position<TIn::Rank>()];
  for_each<Kokkos::Serial>(
      "compute_offsets()", // FIXME analytic through strides?
      strel,
      KOKKOS_LAMBDA(std::integral auto... is) {
        offsets_on_host[*index] = &in(is...) - data;
        ++(*index);
      });
  copy_to(offsets_on_host, offsets); // FIXME offsets_on_host.copy_to(offsets)
  return offsets;
}

/**
 * @brief Correlate two data containers
 * 
 * @param name The output name
 * @param in The input container
 * @param strel The structuring element
 * @param out The output container
 * 
 * Without extraplation, the output container is generally smaller than the input.
 * In this case, the output extent along axis `i` is `in.extent(i) - strel.extent(i) + 1`.
 */
template <typename TIn, typename TStrel, typename TOut>
void median_filter_to(const TIn& in, const TStrel& strel, TOut& out)
{
  for_each("median_filter_to()", out.domain(), Impl::Median(compute_offsets(in, strel), in, out));
  // FIXME as_readonly() anywhere relevant
}

/**
 * @copydoc median_filter_to()
 */
template <typename TIn, typename TStrel>
auto median_filter(const std::string& label, const TIn& in, const TStrel& strel)
{
  TIn out(label, in.shape() - strel.shape() + 1); // FIXME box(strel).shape()
  median_filter_to(in, strel, out);
  return out;
}

/**
 * @copydoc median_filter_to()
 */
template <typename TIn>
auto median_filter(const std::string& label, const TIn& in, Index radius)
{
  constexpr auto N = TIn::Rank;
  const auto rank = in.rank();
  auto strel = Box(Position<N>(Constant(-radius), rank), Position<N>(Constant(radius + 1), rank));
  return median_filter(label, in, strel);
}

} // namespace Linx

#endif
