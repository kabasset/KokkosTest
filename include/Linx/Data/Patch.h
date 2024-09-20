// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXDATA_PATCH_H
#define _LINXDATA_PATCH_H

#include "Linx/Base/Functional.h"
#include "Linx/Base/Types.h"
#include "Linx/Data/Image.h"
#include "Linx/Data/Sequence.h"
#include "Linx/Data/Slice.h"

#include <Kokkos_Core.hpp>

namespace Linx {

/**
 * @brief An image patch.
 * 
 * A patch is a restriction of an image to some domain.
 * As opposed to an image slice, an image patch always has the same rank as the image, and its domain is the input domain.
 */
template <typename TParent, typename TDomain>
class Patch : public DataMixin<typename TParent::value_type, EuclidArithmetic, Patch<TParent, TDomain>> {
public:

  using Parent = TParent; ///< The parent, which may be a patch
  using Domain = TDomain; ///< The domain
  static constexpr int Rank = Domain::Rank;

  using memory_space = typename Parent::memory_space;
  using execution_space = typename Parent::execution_space;

  using value_type = typename Parent::value_type; ///< The value type
  using reference = typename Parent::reference; ///< The reference type

  struct ConstructTag {};

  /**
   * @brief Default constructor.
   */
  Patch() : m_parent(nullptr), m_domain() {}

  /**
   * @brief Constructor.
   */
  Patch(const TParent& parent, TDomain region) : m_parent(&parent), m_domain(LINX_MOVE(region)) {}

  /**
   * @brief The parent.
   */
  KOKKOS_INLINE_FUNCTION const Parent& parent() const
  {
    return *m_parent; // FIXME Dereference on device?
  }

  /**
   * @brief The domain.
   */
  KOKKOS_INLINE_FUNCTION const Domain& domain() const
  {
    return m_domain;
  }

  /**
   * @brief The underlying container.
   */
  KOKKOS_INLINE_FUNCTION const auto& container() const
  {
    return root(*this).container();
  }

  /**
   * @brief The domain size.
   */
  KOKKOS_INLINE_FUNCTION auto size() const
  {
    return domain().size();
  }

  /**
   * @brief Forward to parent's `operator[]`.
   */
  KOKKOS_INLINE_FUNCTION reference operator[](auto&& arg) const
  {
    return (*m_parent)[LINX_FORWARD(arg)];
  }

  /**
   * @brief Forward to parent's `operator()`.
   */
  KOKKOS_INLINE_FUNCTION reference operator()(auto&&... args) const
  {
    return (*m_parent)(LINX_FORWARD(args)...);
  }

  /**
   * @brief Get the element at a given domain-local position.
   * 
   * The arguments are forwarded to the domain,
   * such that the method returns `parent[domain(args...)]`.
   */
  reference local(auto&&... args) const
  {
    return (*m_parent)[m_domain(LINX_FORWARD(args)...)];
  }

  /**
   * @brief Translate the patch by a given vector.
   */
  KOKKOS_INLINE_FUNCTION Patch& operator>>=(const auto& vector) // FIXME constain
  {
    m_domain += vector;
    return *this;
  }

  /**
   * @brief Translate the patch by the opposite of a given vector.
   */
  KOKKOS_INLINE_FUNCTION Patch& operator<<=(const auto& vector) // FIXME constrain
  {
    m_domain -= vector;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION Patch& shift(auto... is)
  {
    m_domain.add(is...);
    return *this;
  }

  KOKKOS_INLINE_FUNCTION Patch& ishift(auto... is)
  {
    m_domain.subtract(is...);
    return *this;
  }

private:

  const Parent* m_parent; ///< The parent
  Domain m_domain; ///< The domain
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
  return patch(in, box(domain & in.domain()));
}

/**
 * @copydoc patch()
 */
template <typename T, int N, typename TContainer, typename U>
auto patch(const Image<T, N, TContainer>& in, const Box<U, N>& domain)
{
  return Patch<Image<T, N, TContainer>, Box<U, N>>(in, domain & in.domain());
}

/**
 * @copydoc patch()
 */
template <typename TParent, typename TDomain, typename U>
auto patch(const Patch<TParent, TDomain>& in, const Box<U, TParent::Rank>& domain)
{
  return Patch<TParent, TDomain>(root(in), domain & in.domain());
}

// FIXME Mask-based patch
// FIXME Sequence/Path-based patch

namespace Impl {

template <typename TView, typename TType, std::size_t... Is>
auto slice_impl(const TView& view, const TType& slice, std::index_sequence<Is...>)
{
  return Kokkos::subview(view, get<Is>(slice).kokkos_slice()...);
}

template <typename TView, typename TType, std::size_t... Is>
auto slice_impl(const TView& view, std::index_sequence<Is...>, const TType& slice)
{
  using Prepend = std::array<Kokkos::ALL_t, sizeof...(Is)>;
  return Kokkos::subview(view, (typename std::tuple_element<Is, Prepend>::type {})..., slice.kokkos_slice());
}

} // namespace Impl

/**
 * @relatesalso Image
 * @relatesalso Sequence
 * @brief Slice an image or sequence.
 * @param in The image or sequence
 * @param region The slicing region as a `Slice` or `Box`
 * @param start, stop, index The slicing indices
 * 
 * As opposed to patches:
 * - If the slice contains singletons, the associated axes are droped;
 * - Coordinates along all axis start at index 0;
 * - The image can safely be destroyed.
 * 
 * @see patch()
 */
template <typename T, int N, typename TContainer, typename U, SliceType... TSlices>
auto slice(const Image<T, N, TContainer>& in, const Slice<U, TSlices...>& region)
{
  const auto& domain = region & in.domain(); // Resolve Kokkos::ALL to drop offsets with subview
  using Container = decltype(Impl::slice_impl(in.container(), domain, std::make_index_sequence<sizeof...(TSlices)>()));
  return Image<T, Container::rank(), Container>(
      Forward {},
      Impl::slice_impl(in.container(), domain, std::make_index_sequence<sizeof...(TSlices)>()));
}

/**
 * @copydoc slice()
 */
template <typename T, int N, typename TContainer, typename U, int M>
auto slice(const Image<T, N, TContainer>& in, const Box<U, M>& region)
{
  const auto& domain = region & in.domain();
  using Container = decltype(Impl::slice_impl(in.container(), domain, std::make_index_sequence<M>()));
  return Image<T, Container::rank(), Container>(
      Forward {},
      Impl::slice_impl(in.container(), domain, std::make_index_sequence<M>()));
}

/**
 * @copydoc slice()
 */
template <typename T, int N, typename TContainer>
auto slice(const Image<T, N, TContainer>& in, int start, int stop)
{
  using Container = decltype(Impl::slice_impl(in.container(), std::make_index_sequence<N - 1>(), Slice(start, stop)));
  return Image<T, Container::rank(), Container>(
      Forward {},
      Impl::slice_impl(in.container(), std::make_index_sequence<N - 1>(), Slice(start, stop)));
}

/**
 * @copydoc slice()
 */
template <typename T, int N, typename TContainer>
auto slice(const Image<T, N, TContainer>& in, int index)
{
  using Container = decltype(Impl::slice_impl(in.container(), std::make_index_sequence<N - 1>(), Slice(index)));
  return Image<T, Container::rank(), Container>(
      Forward {},
      Impl::slice_impl(in.container(), std::make_index_sequence<N - 1>(), Slice(index)));
}

/**
 * @copydoc slice()
 */
template <typename T, int N, typename TContainer, typename U, SliceType TSlice>
auto slice(const Sequence<T, N, TContainer>& in, const Slice<U, TSlice>& region)
{
  const auto& domain = region & in.domain(); // Resolve Kokkos::ALL to drop offsets with subview
  using Container = decltype(Impl::slice_impl(in.container(), domain, std::index_sequence<0>()));
  return Sequence<T, Container::rank(), Container>(
      Forward {},
      Impl::slice_impl(in.container(), domain, std::index_sequence<0>()));
}

} // namespace Linx

#endif
