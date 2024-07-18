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
auto subview_impl(const TView& view, const TSlice& slice, std::index_sequence<Is...>)
{
  return Kokkos::subview(view, get<Is>(slice).kokkos_slice()...);
}

} // namespace Internal
/// @endcond

template <typename T, int N, typename TContainer, typename U, SliceType... TTypes>
auto patch(const std::string& label, const Image<T, N, TContainer>& in, const Slice<U, TTypes...>& slice)
{
  // FIXME label
  using Container =
      decltype(Internal::subview_impl(in.container(), slice, std::make_index_sequence<sizeof...(TTypes)>()));
  return Image<T, Container::rank(), Container>(
      ForwardTag {},
      Internal::subview_impl(in.container(), slice, std::make_index_sequence<sizeof...(TTypes)>()));
  // FIXME OffsetView?
}

} // namespace Linx

#endif
