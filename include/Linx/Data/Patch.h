// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXDATA_PATCH_H
#define _LINXDATA_PATCH_H

#include "Linx/Base/Functional.h"
#include "Linx/Base/Types.h"
#include "Linx/Data/Image.h"
#include "Linx/Data/Sequence.h"

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
  Patch(const TParent& parent, TDomain region) : m_parent(parent), m_domain(LINX_MOVE(region)) {}

  /**
   * @brief The parent.
   */
  KOKKOS_INLINE_FUNCTION const Parent& parent() const
  {
    return m_parent; // FIXME Dereference on device?
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
    return m_parent[LINX_FORWARD(arg)];
  }

  /**
   * @brief Forward to parent's `operator()`.
   */
  KOKKOS_INLINE_FUNCTION reference operator()(auto&&... args) const
  {
    return m_parent(LINX_FORWARD(args)...);
  }

  /**
   * @brief Get the element at a given domain-local position.
   * 
   * The arguments are forwarded to the domain,
   * such that the method returns `parent[domain(args...)]`.
   */
  reference local(auto&&... args) const
  {
    return m_parent[m_domain(LINX_FORWARD(args)...)];
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

  Parent m_parent; ///< The parent
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
auto patch(const Image<T, N, TContainer>& in, const GBox<U, N>& domain)
{
  return Patch<Image<T, N, TContainer>, GBox<U, N>>(in, domain & in.domain());
}

/**
 * @copydoc patch()
 */
template <typename TParent, typename TDomain, typename U>
auto patch(const Patch<TParent, TDomain>& in, const GBox<U, TParent::Rank>& domain)
{
  return Patch<TParent, TDomain>(root(in), domain & in.domain());
}

// FIXME Mask-based patch
// FIXME Sequence/Path-based patch

} // namespace Linx

#endif
