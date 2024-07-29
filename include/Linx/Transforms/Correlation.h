// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXTRANSFORMS_CORRELATION_H
#define _LINXTRANSFORMS_CORRELATION_H

#include "Linx/Data/Image.h"
#include "Linx/Data/Sequence.h"

#include <Kokkos_StdAlgorithms.hpp>
#include <string>

namespace Linx {

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

  auto in_data = in.data(); // FIXME This is not necessarily in(0...)
  auto kernel_data = kernel.data();
  auto kernel_size = kernel.size();
  Sequence<std::ptrdiff_t, -1> offsets("correlate_to(): offsets", kernel_size);
  auto offsets_data = offsets.data();
  Sequence<typename TKernel::value_type, -1> values("correlate_to(): values", kernel_size);
  auto values_data = values.data();
  for_each(
      "correlate_to(): offsets computation", // FIXME analytic through strides?
      kernel.domain(),
      KOKKOS_LAMBDA(auto... is) {
        auto kernel_ptr = &kernel(is...);
        auto in_ptr = &in(is...);
        auto index = kernel_ptr - kernel_data; // Assume a contiguous span
        values[index] = *kernel_ptr;
        offsets[index] = in_ptr - in_data;
      });
  for_each(
      "correlate_to(): dot product",
      out.domain(),
      KOKKOS_LAMBDA(auto... is) {
        auto in_ptr = &in(is...);
        typename TOut::value_type res {};
        for (std::size_t i = 0; i < kernel_size; ++i) {
          res += values_data[i] * in_ptr[offsets_data[i]];
        }
        out(is...) = res;
      });
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
