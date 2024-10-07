// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXDATA_BOX_H
#define _LINXDATA_BOX_H

#include "Linx/Base/Containers.h"
#include "Linx/Base/Exceptions.h"
#include "Linx/Base/Functional.h"
#include "Linx/Base/Packs.h"
#include "Linx/Base/Types.h"
#include "Linx/Base/concepts/Array.h"
#include "Linx/Data/Sequence.h"
#include "Linx/Data/Slice.h"

#include <Kokkos_Core.hpp>
#include <concepts>
#include <string>

namespace Linx {

template <typename T, int N>
using GPosition = Sequence<T, N, typename DefaultContainer<T, N, Kokkos::HostSpace>::Sequence>;

template <int N>
using Position = GPosition<Index, N>;

template <int M, typename T, int N>
auto pad(const GPosition<T, N>& in)
{
  using U = std::decay_t<T>;
  GPosition<U, M> out; // FIXME label
  copy_to(in, out);
  return out;
}

template <int M, typename T, int N>
auto pad(const GPosition<T, N>& in, const T& value)
{
  using U = std::decay_t<T>;
  GPosition<U, M> out; // FIXME label
  out.fill(value);
  copy_to(in, out);
  return out;
}

template <typename T, int N>
struct Shape : StrongType<GPosition<T, N>, struct ShapeTag> { // FIXME const GPosition&?
  using StrongType<GPosition<T, N>, ShapeTag>::StrongType;

  /**
   * @brief Compute the shape size.
   */
  T size() const
  {
    return product(this->value);
  }
};

template <typename T, int N>
Shape(T (&&)[N]) -> Shape<T, N>;

template <typename T, int N>
Shape(const GPosition<T, N>&) -> Shape<T, N>;

/**
 * @relatesalso Window
 * @brief An ND bounding box, defined by its start (inclusive) and stop (exclusive) bounds.
 * 
 * @tparam T The coordinate type
 * @tparam N The dimension parameter
 * 
 * If `T` is integral, the box can be iterated with `for_each()` and `kokkos_reduce()`,
 * and patches can be created from the box.
 * 
 * @see `Patch`
 */
template <typename T, int N>
class GBox {
public:

  static constexpr int Rank = N; ///< The dimension parameter
  using size_type = T; ///< The coordinate type, which may be non-integral
  using value_type = GPosition<T, N>; ///< The position type

  /**
   * @brief Constructor.
   */
  GBox() : GBox(std::abs(Rank)) {}

  /**
   * @copydoc GBox()
   */
  explicit GBox(std::integral auto size) : m_start("start", size), m_stop("stop", size) {}

  /**
   * @copydoc GBox()
   */
  GBox(const ArrayLike auto& start, const ArrayLike auto& stop) : GBox(std::size(start))
  {
    SizeMismatch::may_throw("bounds", rank(), start, stop);
    for (std::size_t i = 0; i < rank(); ++i) {
      m_start[i] = start[i];
      m_stop[i] = stop[i];
    }
  }

  /**
   * @copydoc GBox()
   */
  explicit GBox(const ArrayLike auto& stop) : GBox(value_type(std::size(stop)), stop) {} // FIXME GBox(Shape)

  /**
   * @copydoc GBox()
   */
  template <typename U>
  GBox(std::initializer_list<U> start, std::initializer_list<U> stop) : GBox(std::size(start))
  {
    SizeMismatch::may_throw("bounds", rank(), start, stop); // FIXME handle N = -1
    auto start_it = start.begin();
    auto stop_it = stop.begin();
    for (std::size_t i = 0; i < rank(); ++i, ++start_it, ++stop_it) {
      m_start[i] = *start_it;
      m_stop[i] = *stop_it;
    }
  }

  /**
   * @copydoc GBox()
   */
  GBox(GPosition<size_type, Rank> start, Shape<size_type, Rank> shape) :
      m_start(LINX_MOVE(start)), m_stop(shape.value + m_start)
  {}

  /**
   * @brief The box rank.
   */
  KOKKOS_INLINE_FUNCTION auto rank() const
  {
    return m_start.size();
  }

  /**
   * @brief The box shape.
   */
  auto shape() const
  {
    return m_stop - m_start;
  }

  /**
   * @brief The start bound, inclusive.
   */
  const auto& start() const
  {
    return m_start;
  }

  /**
   * @brief The stop bound, exclusive.
   */
  const auto& stop() const
  {
    return m_stop;
  }

  /**
   * @brief The start bound along given axis.
   */
  KOKKOS_INLINE_FUNCTION auto start(std::integral auto i) const
  {
    return m_start[i];
  }

  /**
   * @copybrief start()
   */
  KOKKOS_INLINE_FUNCTION auto& start(std::integral auto i)
  {
    return m_start[i];
  }

  /**
   * @brief The stop bound along given axis.
   */
  KOKKOS_INLINE_FUNCTION auto stop(std::integral auto i) const
  {
    return m_stop[i];
  }

  /**
   * @copybrief stop()
   */
  KOKKOS_INLINE_FUNCTION auto& stop(std::integral auto i)
  {
    return m_stop[i];
  }

  /**
   * @brief The extent along given axis.
   */
  KOKKOS_INLINE_FUNCTION auto extent(std::integral auto i) const
  {
    return m_stop[i] - m_start[i];
  }

  /**
   * @brief The product of the extents.
   */
  KOKKOS_INLINE_FUNCTION auto size() const
  {
    T out = 1;
    for (std::size_t i = 0; i < m_start.size(); ++i) {
      out *= extent(i);
    }
    return out;
  }

  /**
   * @brief Check whether two boxes are equal.
   */
  bool operator==(const auto& other) const
  {
    return m_start == other.start() && m_stop == other.stop();
  }

  /**
   * @brief Check whether two boxes are different.
   */
  bool operator!=(const auto& other) const
  {
    return not(*this == other);
  }

  /**
   * @brief Check whether a position lies inside the box.
   */
  bool contains(const ArrayLike auto& position) const
  {
    SizeMismatch::may_throw("position", rank(), position);
    for (std::size_t i = 0; i < rank(); ++i) {
      if (position[i] < m_start[i] || position[i] > m_stop[i]) {
        return false;
      }
    }
    return true;
  }

  /**
   * @copydoc contains()
   */
  bool contains(auto... is) const // FIXME accept convertible to size_type only
  {
    return contains(value_type {is...});
  }

  /**
   * @brief Shrink the box inside another box (i.e. get the intersection of both).
   */
  template <typename U, int M>
  GBox& operator&=(const GBox<U, M>& rhs)
  {
    // FIXME assert rank() == rhs.rank()
    for (std::size_t i = 0; i < rank(); ++i) {
      m_start[i] = std::max<size_type>(m_start[i], rhs.start(i));
      m_stop[i] = std::min<size_type>(m_stop[i], rhs.stop(i));
    }
    return *this;
  }

  /**
   * @brief Minimally grow the box to include another box (i.e. get the minimum box which contains both).
   */
  template <typename U, int M>
  GBox& operator|=(const GBox<U, M>& rhs)
  {
    // FIXME assert rank() == rhs.rank()
    for (std::size_t i = 0; i < rank(); ++i) {
      m_start[i] = std::min<size_type>(m_start[i], rhs.start(i));
      m_stop[i] = std::max<size_type>(m_stop[i], rhs.stop(i));
    }
    return *this;
  }

  /**
   * @brief Grow the box by a given margin.
   */
  template <typename U, int M>
  GBox& operator+=(const GBox<U, M>& margin)
  {
    // FIXME allow N=-1
    m_start += pad<Rank>(margin.start());
    m_stop += pad<Rank>(margin.stop());
    return *this;
  }

  /**
   * @brief Shrink the box by a given margin.
   */
  template <typename U, int M>
  GBox& operator-=(const GBox<U, M>& margin)
  {
    // FIXME allow N=-1
    m_start -= pad<Rank>(margin.start());
    m_stop -= pad<Rank>(margin.stop());
    return *this;
  }

  /**
   * @brief Translate the box by a given vector.
   */
  GBox& operator+=(const ArrayLike auto& vector)
  {
    // FIXME allow N=-1
    m_start += pad<Rank>(vector);
    m_stop += pad<Rank>(vector);
    return *this;
  }

  /**
   * @brief Translate the box by the opposite of a given vector.
   */
  GBox& operator-=(const ArrayLike auto& vector)
  {
    // FIXME allow N=-1
    m_start -= pad<Rank>(vector);
    m_stop -= pad<Rank>(vector);
    return *this;
  }

  /**
    * @brief Add a scalar to each coordinate.
    */
  GBox& operator+=(size_type scalar)
  {
    m_start += scalar;
    m_stop += scalar;
    return *this;
  }

  /**
   * @brief Subtract a scalar to each coordinate.
   */
  GBox& operator-=(size_type scalar)
  {
    m_start -= scalar;
    m_stop -= scalar;
    return *this;
  }

  /**
   * @brief Add 1 to each coordinate.
   */
  GBox& operator++()
  {
    return *this += 1;
  }

  GBox operator++(int)
  {
    GBox out = +(*this);
    ++(*this);
    return out;
  }

  /**
   * @brief Subtract 1 to each coordinate.
   */
  GBox& operator--()
  {
    return *this -= 1;
  }

  GBox operator--(int)
  {
    GBox out = +(*this);
    --(*this);
    return out;
  }

  /**
   * @brief Copy.
   */
  GBox operator+() const
  {
    return {+m_start, +m_stop};
  }

  /**
   * @brief Invert the sign of each coordinate.
   */
  GBox operator-() const
  {
    return {-m_start, -m_stop};
  }

  /**
   * @brief Equality.
   */
  template <typename U, int M>
  bool operator==(const GBox<U, M>& rhs) const
  {
    return m_start == rhs.m_start && m_stop == rhs.m_stop;
  }

  /**
   * @brief Inequality.
   */
  template <typename U, int M>
  bool operator!=(const GBox<U, M>& rhs) const
  {
    return not(*this == rhs);
  }

private:

  value_type m_start; ///< The start bound
  value_type m_stop; ///< The stop bound
};

GBox()->GBox<int, 0>;

template <typename T, int N>
GBox(T (&&)[N]) -> GBox<T, N>;

template <typename T, int N>
GBox(T (&&)[N], T (&&)[N]) -> GBox<T, N>;

template <typename T, int N>
GBox(const GPosition<T, N>&) -> GBox<T, N>;

template <typename T, int N>
GBox(const GPosition<T, N>&, const GPosition<T, N>&) -> GBox<T, N>;

template <typename T, int N>
GBox(const GPosition<T, N>&, const Shape<T, N>&) -> GBox<T, N>;

template <typename T, int N>
GBox(T (&&)[N], const Shape<T, N>&) -> GBox<T, N>;

template <int M, typename T, int N>
GBox<T, M> pad(const GBox<T, N>& in)
{
  return GBox<T, M>({pad<M>(in.start()), pad<M>(in.stop())});
}

/**
 * @relatesalso GBox
 */
template <typename T, int N>
GBox<T, N> operator+(const GBox<T, N>& lhs, const auto& rhs)
{
  auto out = +lhs;
  out += rhs;
  return out;
}

/**
 * @relatesalso GBox
 */
template <typename T, int N>
GBox<T, N> operator-(GBox<T, N> lhs, const auto& rhs)
{
  auto out = +lhs;
  out -= rhs;
  return out;
}

/**
 * @relatesalso GBox
 */
template <typename T, int N, typename U, int M>
GBox<T, N> operator&(GBox<T, N> lhs, const GBox<U, M>& rhs)
{
  auto out = +lhs;
  out &= rhs;
  return out;
}

/**
 * @brief Get the 1D slice along the i-th axis.
 */
template <int I, typename T, int N>
Slice<T, SliceType::RightOpen> get(const GBox<T, N>& box)
{
  return {box.start(I), box.stop(I)};
}

namespace Impl {

template <typename T>
T slice_start_impl(const Slice<T, SliceType::Singleton>& slice)
{
  return slice.value();
}

template <std::integral T>
T slice_stop_impl(const Slice<T, SliceType::Singleton>& slice)
{
  return slice.value() + 1;
}

template <typename T>
T slice_start_impl(const Slice<T, SliceType::RightOpen>& slice)
{
  return slice.start();
}

template <typename T>
T slice_stop_impl(const Slice<T, SliceType::RightOpen>& slice)
{
  return slice.stop();
}

template <typename TType, std::size_t... Is>
auto box_impl(const TType& slice, std::index_sequence<Is...>)
{
  using T = typename TType::size_type; // FIXME assert T is integral
  static constexpr int N = sizeof...(Is);
  return GBox<T, N>({slice_start_impl(get<Is>(slice))...}, {slice_stop_impl(get<Is>(slice))...});
}

} // namespace Impl

/**
 * @brief Get the bounding box of a slice.
 * 
 * @warning Unbounded slices are not supported, and singleton slices must be integral.
 */
template <typename T, SliceType... TTypes>
GBox<T, sizeof...(TTypes)> box(const Slice<T, TTypes...>& slice)
{
  static constexpr int N = sizeof...(TTypes);
  return Impl::box_impl(slice, std::make_index_sequence<N>());
}

/**
 * @brief Make a slice clamped by a box.
 */
template <typename T, typename U, int N, SliceType... TTypes>
auto operator&(const Slice<T, TTypes...>& slice, const GBox<U, N>& box)
{
  static constexpr auto Last = sizeof...(TTypes) - 1;
  return (slice.fronts() & box)(clamp(slice.back(), box.start(Last), box.stop(Last)));
}

/**
 * @brief Make a 1D slice clamped by a box.
 */
template <typename T, SliceType TType, typename U, int N>
auto operator&(const Slice<T, TType>& slice, const GBox<U, N>& box)
{
  return clamp(slice, box.start(0), box.stop(0));
}

namespace Impl {

template <typename TSpace, typename T, int N, std::size_t... Is>
auto kokkos_execution_policy_impl(const GBox<T, N>& domain, std::index_sequence<Is...>)
{
  using Policy = Kokkos::MDRangePolicy<TSpace, Kokkos::Rank<N>>;
  using Array = Policy::point_type;
  return Policy(Array {domain.start(Is)...}, Array {domain.stop(Is)...});
}

} // namespace Impl

/**
 * @brief Shortcut for indexing.
 */
template <int N>
using Box = GBox<Index, N>;

/**
 * @brief Get the execution policy of a box.
 */
template <typename TSpace, typename T, int N>
auto kokkos_execution_policy(const GBox<T, N>& domain)
{
  // FIXME support Properties
  if constexpr (N == 1) {
    return Kokkos::RangePolicy<TSpace>(domain.start(0), domain.stop(0));
  } else {
    return Impl::kokkos_execution_policy_impl<TSpace>(domain, std::make_index_sequence<N>());
  }
}

/**
 * @brief Apply a function to each position of a region.
 * 
 * @param label Some label for debugging
 * @param region The region
 * @param func The function
 * 
 * The coordinate type must be integral and the function must take integral coordinates as input.
 */
template <typename TSpace = Kokkos::DefaultExecutionSpace, typename T, int N, typename TFunc>
void for_each(const std::string& label, const GBox<T, N>& region, TFunc&& func)
{
#define LINX_CASE_RANK(n) \
  case n: \
    if constexpr (is_nadic<int, n, TFunc>()) { \
      return Kokkos::parallel_for(label, kokkos_execution_policy<TSpace>(pad<n>(region)), LINX_FORWARD(func)); \
    } else { \
      return; \
    }

  if constexpr (N == -1) {
    switch (region.rank()) {
      case 0:
        return;
        LINX_CASE_RANK(1)
        LINX_CASE_RANK(2)
        LINX_CASE_RANK(3)
        LINX_CASE_RANK(4)
        LINX_CASE_RANK(5)
        LINX_CASE_RANK(6)
      default:
        throw Linx::OutOfBounds<'[', ']'>("Dynamic rank", region.rank(), {0, 6});
    }
  } else {
    Kokkos::parallel_for(label, kokkos_execution_policy<TSpace>(region), LINX_FORWARD(func));
  }

#undef LINX_CASE_RANK
}

} // namespace Linx

#endif
