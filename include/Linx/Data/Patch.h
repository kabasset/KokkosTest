// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXDATA_PATCH_H
#define _LINXDATA_PATCH_H

#include "Linx/Data/Image.h"
#include "Linx/Data/Slice.h"

#include <Kokkos_Core.hpp>

namespace Linx {

/// @cond
namespace Internal {

template <typename TView, typename TSlice, std::size_t... Is>
auto slice_impl(const TView& view, const TSlice& slice, std::index_sequence<Is...>)
{
  return Kokkos::subview(view, get<Is>(slice).kokkos_slice()...);
}

template <typename TView, typename TBox, std::size_t... Is>
auto box_patch_impl(const TView& view, const TBox& box, std::index_sequence<Is...>)
{
  using Subview = decltype(Kokkos::subview(view, Kokkos::pair(get<Is>(box).start(), get<Is>(box).stop())...));
  return typename Rebind<Subview>::Offset(
      Kokkos::subview(view, Kokkos::pair(get<Is>(box).start(), get<Is>(box).stop())...),
      {get<Is>(box).start()...});
}

} // namespace Internal
/// @endcond

/**
 * @brief Slice an image.
 * 
 * As opposed to patches:
 * - If the slice contains singletons, the associated axes are droped;
 * - Coordinates along all axis start at index 0.
 * 
 * @see patch()
 */
template <typename T, int N, typename TContainer, typename U, SliceType... TSlices>
auto slice(const Image<T, N, TContainer>& in, const Slice<U, TSlices...>& slice)
{
  const auto domain = clamp(slice, in.domain()); // Resolve Kokkos::ALL to drop offsets with subview
  using Container =
      decltype(Internal::slice_impl(in.container(), domain, std::make_index_sequence<sizeof...(TSlices)>()));
  return Image<T, Container::rank(), Container>(
      ForwardTag {},
      Internal::slice_impl(in.container(), domain, std::make_index_sequence<sizeof...(TSlices)>()));
}

/**
 * @brief Make a patch of an image.
 * 
 * @param in The input container
 * @param domain The patch domain
 * 
 * A patch is a restriction of an image to some domain.
 * As opposed to an image slice, an image patch always has the same rank as the image, and its domain is the input domain.
 * 
 * @see slice()
 */
template <typename T, int N, typename TContainer, typename U, SliceType... TSlices>
auto patch(const Image<T, N, TContainer>& in, const Slice<U, TSlices...>& domain)
{
  return patch(in, box(clamp(domain, in.domain())));
}

/**
 * @copydoc patch()
 */
template <typename T, int N, typename TContainer, typename U>
auto patch(const Image<T, N, TContainer>& in, const Box<U, N>& domain)
{
  using Container = decltype(Internal::box_patch_impl(in.container(), domain, std::make_index_sequence<N>()));
  return Image<T, Container::Rank, Container>(
      ForwardTag {},
      Internal::box_patch_impl(in.container(), domain, std::make_index_sequence<N>()));
}

// FIXME Mask-based patch
// FIXME Sequence-based patch

} // namespace Linx

#endif
