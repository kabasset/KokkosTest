// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXTRANSFORMS_RANKFILTERING_H
#define _LINXTRANSFORMS_RANKFILTERING_H

#include "Linx/Base/Algorithm.h"
#include "Linx/Base/ArrayPool.h"
#include "Linx/Data/Image.h"
#include "Linx/Data/Sequence.h"
#include "Linx/Transforms/mixins/FilterMixin.h"

#include <concepts>
#include <string>

namespace Linx {

template <typename TStrel, typename TIn, typename TParity = Forward>
class MedianFilter : public MorphologyFilterMixin<TIn, MedianFilter<TStrel, TIn, TParity>> {
public:

  using value_type = typename TIn::value_type;
  using element_type = std::remove_cvref_t<value_type>;

  MedianFilter(const TStrel& strel, const TIn& in) :
      MorphologyFilterMixin<TIn, MedianFilter>(strel, in), m_neighbors(this->m_offsets.size())
  {}

  MedianFilter(TParity, const TStrel& strel, const TIn& in) : MedianFilter(strel, in)
  {
    // FIXME test size
  }

  // TODO MedianFilter(std::integral auto radius, const TIn& in)

  std::string label() const
  {
    return "MedianFilter";
  }

  KOKKOS_INLINE_FUNCTION auto operator()(const std::integral auto&... is) const
  {
    auto array = m_neighbors.array();
    auto in_ptr = &this->m_in(is...);
    for (std::size_t i = 0; i < array.size(); ++i) {
      array[i] = in_ptr[this->m_offsets[i]];
    }
    return median<TParity>(array);
  }

private:

  ArrayPool<element_type> m_neighbors; // FIXME TSpace
};

template <typename TStrel, typename TIn, typename TParity = Forward>
class MinFilter : public MorphologyFilterMixin<TIn, MinFilter<TStrel, TIn, TParity>> {
public:

  using value_type = typename TIn::value_type;
  using element_type = std::remove_cvref_t<value_type>;

  MinFilter(const TStrel& strel, const TIn& in) : MorphologyFilterMixin<TIn, MinFilter>(strel, in) {}

  // TODO MinFilter(std::integral auto radius, const TIn& in)

  std::string label() const
  {
    return "MinFilter";
  }

  KOKKOS_INLINE_FUNCTION auto operator()(const std::integral auto&... is) const
  {
    auto in_ptr = &this->m_in(is...);
    auto out = identity_element<element_type>(Min());
    for (std::size_t i = 0; i < this->m_offsets.size(); ++i) {
      out = std::min<element_type>(out, in_ptr[this->m_offsets[i]]);
    }
    return out;
  }
};

template <typename TStrel, typename TIn, typename TParity = Forward>
class MaxFilter : public MorphologyFilterMixin<TIn, MaxFilter<TStrel, TIn, TParity>> {
public:

  using value_type = typename TIn::value_type;
  using element_type = std::remove_cvref_t<value_type>;

  MaxFilter(const TStrel& strel, const TIn& in) : MorphologyFilterMixin<TIn, MaxFilter>(strel, in) {}

  // TODO MaxFilter(std::integral auto radius, const TIn& in)

  std::string label() const
  {
    return "MaxFilter";
  }

  KOKKOS_INLINE_FUNCTION auto operator()(const std::integral auto&... is) const
  {
    auto in_ptr = &this->m_in(is...);
    auto out = identity_element<element_type>(Max());
    for (std::size_t i = 0; i < this->m_offsets.size(); ++i) {
      out = std::max<element_type>(out, in_ptr[this->m_offsets[i]]);
    }
    return out;
  }
};

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
void median_filter_to(const TStrel& strel, const TIn& in, TOut& out)
{
  if (strel.size() % 2 == 0) {
    out.copy_from(MedianFilter(EvenNumber(), strel, in));
  } else {
    out.copy_from(MedianFilter(OddNumber(), strel, in));
  }
}

/**
 * @copydoc median_filter_to()
 */
template <typename TIn, typename TStrel>
auto median_filter(const std::string& label, const TStrel& strel, const TIn& in)
{
  auto bbox = +strel; // FIXME box(strel)
  TIn out(label, in.shape() - bbox.shape() + 1);
  median_filter_to(strel - bbox.start(), in, out);
  return out;
}

/**
 * @copydoc median_filter_to()
 */
template <typename TIn>
auto median_filter(const std::string& label, Index radius, const TIn& in)
{
  constexpr auto N = TIn::Rank;
  const auto rank = in.rank();
  auto strel = Box(Position<N>(Constant(-radius), rank), Position<N>(Constant(radius + 1), rank));
  return median_filter(label, strel, in);
}

template <typename TIn>
auto min_filter(const std::string& label, Index radius, const TIn& in)
{
  constexpr auto N = TIn::Rank;
  const auto rank = in.rank();
  auto strel = Box(Position<N>(Constant(-radius), rank), Position<N>(Constant(radius + 1), rank));

  auto bbox = +strel; // FIXME box(strel)
  TIn out(label, in.shape() - bbox.shape() + 1);
  out.copy_from(MinFilter(strel - bbox.start(), in));
  return out;
}

template <typename TIn>
auto max_filter(const std::string& label, Index radius, const TIn& in)
{
  constexpr auto N = TIn::Rank;
  const auto rank = in.rank();
  auto strel = Box(Position<N>(Constant(-radius), rank), Position<N>(Constant(radius + 1), rank));

  auto bbox = +strel; // FIXME box(strel)
  TIn out(label, in.shape() - bbox.shape() + 1);
  out.copy_from(MaxFilter(strel - bbox.start(), in));
  return out;
}

} // namespace Linx

#endif
