// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXTRANSFORMS_CORRELATION_H
#define _LINXTRANSFORMS_CORRELATION_H

#include "Linx/Data/Image.h"
#include "Linx/Data/Sequence.h"

#include <Kokkos_StdAlgorithms.hpp>
#include <concepts>
#include <string>

namespace Linx {

/// @cond
namespace Internal {

template <typename TKernel, typename TIn>
struct OffsetValueMap {
  OffsetValueMap(
      const TKernel& kernel,
      const TIn& in,
      const Sequence<std::ptrdiff_t, -1>& offsets,
      const Sequence<typename TKernel::value_type, -1>& values) :
      m_kernel(kernel),
      m_in(in), m_kernel_data(kernel.data()), m_in_data(in.data()), // FIXME This is not necessarily in(0...)
      m_offsets(offsets), m_values(values)
  {}

  KOKKOS_INLINE_FUNCTION void operator()(const std::integral auto&... is) const
  {
    auto kernel_ptr = &m_kernel(is...);
    auto in_ptr = &m_in(is...);
    auto index = kernel_ptr - m_kernel_data; // Assume a contiguous span
    m_values[index] = *kernel_ptr;
    m_offsets[index] = in_ptr - m_in_data;
  }

  TKernel m_kernel;
  TIn m_in;
  typename TKernel::pointer m_kernel_data;
  typename TIn::pointer m_in_data;
  Sequence<std::ptrdiff_t, -1> m_offsets;
  Sequence<typename TKernel::value_type, -1> m_values;
};

template <typename TValues, typename TIn, typename TOut>
struct OffsetValueDot {
  OffsetValueDot(const Sequence<std::ptrdiff_t, -1> offsets, const TValues& values, const TIn& in, const TOut& out) :
      m_offsets(offsets), m_values(values), m_in(in), m_out(out), m_size(m_offsets.size())
  {}

  KOKKOS_INLINE_FUNCTION void operator()(const std::integral auto&... is) const
  {
    auto in_ptr = &m_in(is...);
    typename TOut::value_type res {};
    for (std::size_t i = 0; i < m_size; ++i) {
      res += m_values[i] * in_ptr[m_offsets[i]];
    }
    m_out(is...) = res;
  }

  Sequence<std::ptrdiff_t, -1> m_offsets;
  TValues m_values;
  TIn m_in;
  TOut m_out;
  std::size_t m_size;
};

} // namespace Internal
/// @endcond

/**
 * @brief Correlate two data containers
 * 
 * @param name The output name
 * @param in The input container
 * @param kernel The kernel container
 * @param out The output container
 * 
 * Without extraplation, the output container is generally smaller than the input and kernel containers.
 * In this case, the output extent along axis `i` is `in.extent(i) - kernel.extent(i) + 1`.
 */
template <typename TIn, typename TKernel, typename TOut>
void correlate_to(const TIn& in, const TKernel& kernel, TOut& out)
{
  assert(kernel.container().span_is_contiguous() && "As of today, correlate_to() only accepts contiguous kernels.");
  // FIXME copy kernel to Raster if not contiguous (see below)

  auto kernel_size = kernel.size();
  Sequence<std::ptrdiff_t, -1> offsets("correlate_to(): offsets", kernel_size);
  Sequence<typename TKernel::value_type, -1> values("correlate_to(): values", kernel_size);
  for_each(
      "correlate_to(): offsets computation", // FIXME analytic through strides?
      kernel.domain(),
      Internal::OffsetValueMap(kernel, in, offsets, values));
  for_each("correlate_to(): dot product", out.domain(), Internal::OffsetValueDot(offsets, values, in, out));
  // FIXME as_readonly() anywhere relevant
}

/**
 * @copydoc correlate_to()
 */
template <typename TIn, typename TKernel>
auto correlate(const std::string& label, const TIn& in, const TKernel& kernel)
{
  TKernel out(label, in.shape() - kernel.shape() + 1);
  correlate_to(in, kernel, out);
  return out;
}

} // namespace Linx

#endif
