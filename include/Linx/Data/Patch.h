// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXDATA_PATCH_H
#define _LINXDATA_PATCH_H

#include "Linx/Base/Types.h"
#include "Linx/Data/Image.h"
#include "Linx/Data/Slice.h"

#include <Kokkos_Core.hpp>

namespace Linx {

/**
 * @brief An image patch.
 * 
 * A patch is a restriction of an image to some domain.
 * As opposed to an image slice, an image patch always has the same rank as the image, and its domain is the input domain.
 * The patch itself contains no data: it only points to the parent image, so that the lifespan of the image must exceed that of the patch.
 */
template <typename TParent, typename TDomain>
class Patch : public DataMixin<typename TParent::value_type, EuclidArithmetic, Patch<TParent, TDomain>> {
public:

  using Parent = TParent; ///< The parent, which may be a patch
  using Domain = TDomain; ///< The domain
  static constexpr int Rank = Domain::Rank;

  using value_type = typename Parent::value_type; ///< The value type
  using reference = typename Parent::reference; ///< The reference type

  /**
   * @brief Constructor.
   */
  Patch(const TParent& parent, TDomain region) : m_parent(&parent), m_region(LINX_MOVE(region)) {}

  /**
   * @brief The parent.
   */
  KOKKOS_INLINE_FUNCTION const Parent& parent() const
  {
    return *m_parent;
  }

  /**
   * @brief The domain.
   */
  KOKKOS_INLINE_FUNCTION const Domain& domain() const
  {
    return m_region;
  }

  /**
   * @brief Forward to parent's `operator[]`.
   */
  KOKKOS_INLINE_FUNCTION reference operator[](auto&& arg) const
  {
    return *m_parent[LINX_FORWARD(arg)];
  }

  /**
   * @brief Forward to parent's `operator()`.
   */
  KOKKOS_INLINE_FUNCTION reference operator()(auto... args) const
  {
    return (*m_parent)(args...);
  }

  /**
   * @copydoc Image::generate_with_side_effects()
   */
  const Patch& generate_with_side_effects(const std::string& label, auto&& func, const auto&... others) const
  {
    m_region.iterate(
        label,
        KOKKOS_LAMBDA(auto... is) { *m_parent(is...) = func(others(is...)...); });
    return *this;
  }

private:

  const TParent* m_parent; ///< Pointer to the parent
  Domain m_region; ///< Domain
};

template <typename T>
concept AnyPatch = is_specialization<Patch, T>;

/**
 * @relatesalso Patch
 * @brief Get the root data container.
 */
KOKKOS_INLINE_FUNCTION const auto& root(const AnyPatch auto& patch)
{
  return root(patch.parent());
}

/**
 * @relatesalso Patch
 * @brief Identity for compatibility with `Patch`.
 */
KOKKOS_INLINE_FUNCTION const auto& root(const AnyImage auto& image)
{
  return image;
}

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
 * @relatesalso Image
 * @brief Slice an image.
 * 
 * As opposed to patches:
 * - If the slice contains singletons, the associated axes are droped;
 * - Coordinates along all axis start at index 0;
 * - The image can safely be destroyed.
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
 * @relatesalso Image
 * @relatesalso Patch
 * @brief Make a patch of an image.
 * 
 * @param in The input container
 * @param domain The patch domain
 * 
 * If the domain is larger than the image domain, then their intersection is used.
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
  return Patch<Image<T, N, TContainer>, Box<U, N>>(in, domain);
}

// FIXME Mask-based patch
// FIXME Sequence/Path-based patch

} // namespace Linx

#endif
