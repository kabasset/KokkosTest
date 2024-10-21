// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXTRANSFORMS_CORRELATION_H
#define _LINXTRANSFORMS_CORRELATION_H

#include "Linx/Data/Image.h"
#include "Linx/Data/Sequence.h"
#include "Linx/Transforms/mixins/FilterMixin.h"

#include <Kokkos_StdAlgorithms.hpp>
#include <concepts>
#include <string>

namespace Linx {

template <typename TKernel, typename TIn>
class Correlation : public WeightedFilterMixin<TKernel, TIn, Correlation<TKernel, TIn>> {
public:

  using value_type = typename TKernel::value_type;
  using element_type = typename TKernel::value_type;

  Correlation(const TKernel& kernel, const TIn& in) : WeightedFilterMixin<TKernel, TIn, Correlation>(kernel, in) {}

  std::string label() const
  {
    return "Correlation";
  }

  KOKKOS_INLINE_FUNCTION auto operator()(const std::integral auto&... is)
  {
    auto in_ptr = &this->m_in(is...);
    element_type res {};
    for (std::size_t i = 0; i < this->m_offsets.size(); ++i) {
      res += this->m_weights[i] * in_ptr[this->m_offsets[i]];
    }
    return res;
  }
};

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
void correlate_to(const TIn& in, const TKernel& kernel, TOut& out) // FIXME swap in and kernel args
{
  out.copy_from(Correlation(kernel, in));
}

/**
 * @copydoc correlate_to()
 */
template <typename TIn, typename TKernel>
auto correlate(const std::string& label, const TIn& in, const TKernel& kernel) // FIXME idem
{
  TKernel out(label, in.shape() - kernel.shape() + 1);
  correlate_to(in, kernel, out);
  return out;
}

} // namespace Linx

#endif
