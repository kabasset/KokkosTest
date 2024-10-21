// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXTRANSFORMS_MORPHOLOGY_H
#define _LINXTRANSFORMS_MORPHOLOGY_H

#include "Linx/Base/Algorithm.h"
#include "Linx/Base/ArrayPool.h"
#include "Linx/Data/Image.h"
#include "Linx/Data/Sequence.h"
#include "Linx/Transforms/mixins/FilterMixin.h"

#include <concepts>
#include <string>

namespace Linx {

template <typename TStrel, typename TIn, typename TParity = Forward>
class Erosion : public MorphologyFilterMixin<TIn, Erosion<TStrel, TIn, TParity>> {
public:

  using value_type = bool;
  using element_type = bool;

  Erosion(const TStrel& strel, const TIn& in) : MorphologyFilterMixin<TIn, Erosion>(strel, in) {}

  // TODO Erosion(std::integral auto radius, const TIn& in)

  std::string label() const
  {
    return "Erosion";
  }

  KOKKOS_INLINE_FUNCTION auto operator()(const std::integral auto&... is) const
  {
    auto in_ptr = &this->m_in(is...);
    for (std::size_t i = 0; i < this->m_offsets.size(); ++i) {
      if (not in_ptr[this->m_offsets[i]]) {
        return false;
      }
    }
    return true;
  }
};

template <typename TStrel, typename TIn, typename TParity = Forward>
class Dilation : public MorphologyFilterMixin<TIn, Dilation<TStrel, TIn, TParity>> {
public:

  using value_type = bool;
  using element_type = bool;

  Dilation(const TStrel& strel, const TIn& in) : MorphologyFilterMixin<TIn, Dilation>(strel, in) {}

  // TODO Dilation(std::integral auto radius, const TIn& in)

  std::string label() const
  {
    return "Dilation";
  }

  KOKKOS_INLINE_FUNCTION auto operator()(const std::integral auto&... is) const
  {
    auto in_ptr = &this->m_in(is...);
    for (std::size_t i = 0; i < this->m_offsets.size(); ++i) {
      if (in_ptr[this->m_offsets[i]]) {
        return true;
      }
    }
    return false;
  }
};

template <typename TIn>
auto erode(const std::string& label, Index radius, const TIn& in)
{
  constexpr auto N = TIn::Rank;
  const auto rank = in.rank();
  auto strel = Box(Position<N>(Constant(-radius), rank), Position<N>(Constant(radius + 1), rank));

  auto bbox = +strel; // FIXME box(strel)
  TIn out(label, in.shape() - bbox.shape() + 1);
  out.copy_from(Erosion(strel - bbox.start(), in));
  return out;
}

template <typename TIn>
auto dilate(const std::string& label, Index radius, const TIn& in)
{
  constexpr auto N = TIn::Rank;
  const auto rank = in.rank();
  auto strel = Box(Position<N>(Constant(-radius), rank), Position<N>(Constant(radius + 1), rank));

  auto bbox = +strel; // FIXME box(strel)
  TIn out(label, in.shape() - bbox.shape() + 1);
  out.copy_from(Dilation(strel - bbox.start(), in));
  return out;
}

} // namespace Linx

#endif
