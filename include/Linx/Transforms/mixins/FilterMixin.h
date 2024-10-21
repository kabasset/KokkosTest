// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXTRANSFORMS_FILTERMIXIN_H
#define _LINXTRANSFORMS_FILTERMIXIN_H

#include "Linx/Base/ArrayPool.h"
#include "Linx/Data/Sequence.h"

#include <string>

namespace Linx {

template <typename TIn, typename TDerived>
class MorphologyFilterMixin { // FIXME simply FilterMixin?
public:

  MorphologyFilterMixin(const auto& strel, const TIn& in) :
      MorphologyFilterMixin(Sequence<std::ptrdiff_t, -1>("offsets", strel.size()), in)
  {
    auto offsets_on_host = on_host(m_offsets);
    auto index = std::make_shared<Index>(0);
    auto front = &m_in.front();
    for_each<Kokkos::Serial>(
        "compute_offsets()", // FIXME analytic through strides?
        strel,
        KOKKOS_LAMBDA(std::integral auto... is) {
          offsets_on_host[*index] = &m_in(is...) - front;
          ++(*index);
        });
    copy_to(offsets_on_host, m_offsets); // FIXME offsets_on_host.copy_to(m_offsets)
  }

protected:

  MorphologyFilterMixin(const Sequence<std::ptrdiff_t, -1>& offsets, const TIn& in) :
      m_offsets(offsets), m_in(as_readonly(in))
  {}

  Sequence<std::ptrdiff_t, -1> m_offsets;
  decltype(as_readonly(std::declval<TIn>())) m_in;
};

template <typename TIn, typename TDerived>
const TDerived& as_readonly(const MorphologyFilterMixin<TIn, TDerived>& in)
{
  return static_cast<const TDerived&>(in);
}

template <typename TKernel, typename TIn, typename TDerived>
class WeightedFilterMixin : public MorphologyFilterMixin<TIn, TDerived> {
public:

  WeightedFilterMixin(const auto& kernel, const auto& in) :
      MorphologyFilterMixin<TIn, TDerived>(Sequence<std::ptrdiff_t, -1>("offsets", kernel.size()), in),
      m_weights("weights", kernel.size())
  {
    // FIXME delegate m_offsets computation to Morphology?

    auto offsets_on_host = on_host(this->m_offsets);
    auto weights_on_host = on_host(m_weights);
    auto index = std::make_shared<Index>(0);
    auto front = &this->m_in.front();
    for_each<Kokkos::Serial>(
        "compute_offsets()",
        kernel.domain(),
        KOKKOS_LAMBDA(std::integral auto... is) {
          offsets_on_host[*index] = &this->m_in(is...) - front;
          weights_on_host[*index] = kernel(is...);
          ++(*index);
        });
    copy_to(offsets_on_host, this->m_offsets); // FIXME offsets_on_host.copy_to(m_offsets)
    copy_to(weights_on_host, m_weights); // FIXME
  }

protected:

  Sequence<typename TKernel::element_type, -1> m_weights;
};

} // namespace Linx

#endif
