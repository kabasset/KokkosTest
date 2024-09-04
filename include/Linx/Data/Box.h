// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXDATA_BOX_H
#define _LINXDATA_BOX_H

#include "Linx/Base/Containers.h"
#include "Linx/Base/Exceptions.h"
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
using Position = Sequence<T, N, typename DefaultContainer<T, N, Kokkos::HostSpace>::Sequence>;

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
class Box {
public:

  static constexpr int Rank = N; ///< The dimension parameter
  using size_type = T; ///< The coordinate type, which may be non-integral
  using value_type = Position<T, N>; ///< The position type

  Box(std::integral auto /* size */) : m_start("Box start"), m_stop("Box stop") {} // FIXME handle N = -1

  /**
   * @brief Constructor.
   */
  Box(const ArrayLike auto& start, const ArrayLike auto& stop) : Box(std::size(start))
  {
    SizeMismatch::may_throw("bounds", rank(), start, stop);
    for (std::size_t i = 0; i < rank(); ++i) {
      m_start[i] = start[i];
      m_stop[i] = stop[i];
    }
  }

  /**
   * @brief Constructor.
   */
  template <typename U>
  Box(std::initializer_list<U> start, std::initializer_list<U> stop) : Box(std::size(start))
  {
    SizeMismatch::may_throw("bounds", rank(), start, stop); // FIXME handle N = -1
    auto start_it = start.begin();
    auto stop_it = stop.begin();
    for (std::size_t i = 0; i < rank(); ++i, ++start_it, ++stop_it) {
      m_start[i] = *start_it;
      m_stop[i] = *stop_it;
    }
  }

  auto rank() const
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
  auto start(std::integral auto i) const
  {
    return m_start[i];
  }

  /**
   * @copybrief start()
   */
  auto& start(std::integral auto i)
  {
    return m_start[i];
  }

  /**
   * @brief The stop bound along given axis.
   */
  auto stop(std::integral auto i) const
  {
    return m_stop[i];
  }

  /**
   * @copybrief stop()
   */
  auto& stop(std::integral auto i)
  {
    return m_stop[i];
  }

  /**
   * @brief The extent along given axis.
   */
  auto extent(std::integral auto i) const
  {
    return m_stop[i] - m_start[i];
  }

  /**
   * @brief The product of the extents.
   */
  auto size() const
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
    return contains(Kokkos::Array<size_type, sizeof...(is)> {is...});
  }

  /**
   * @brief Shrink the box inside another box (i.e. get the intersection of both).
   */
  template <typename U, int M>
  Box& operator&=(const Box<U, M>& rhs)
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
  Box& operator|=(const Box<U, M>& rhs)
  {
    // FIXME assert rank() == rhs.rank()
    for (std::size_t i = 0; i < rank(); ++i) {
      m_start[i] = std::min(m_start[i], rhs.start(i));
      m_stop[i] = std::max(m_stop[i], rhs.stop(i));
    }
    return *this;
  }

  /**
   * @brief Grow the box by a given margin.
   */
  template <typename U, int M>
  Box& operator+=(const Box<U, M>& margin)
  {
    // FIXME allow N=-1
    m_start += resize<Rank>(margin.start());
    m_stop += resize<Rank>(margin.stop());
    return *this;
  }

  /**
   * @brief Shrink the box by a given margin.
   */
  template <typename U, int M>
  Box& operator-=(const Box<U, M>& margin)
  {
    // FIXME allow N=-1
    m_start -= resize<Rank>(margin.start());
    m_stop -= resize<Rank>(margin.stop());
    return *this;
  }

  /**
   * @brief Translate the box by a given vector.
   */
  Box& operator+=(const ArrayLike auto& vector)
  {
    // FIXME allow N=-1
    m_start += resize<Rank>(vector);
    m_stop += resize<Rank>(vector);
    return *this;
  }

  /**
   * @brief Translate the box by the opposite of a given vector.
   */
  Box& operator-=(const ArrayLike auto& vector)
  {
    // FIXME allow N=-1
    m_start -= resize<Rank>(vector);
    m_stop -= resize<Rank>(vector);
    return *this;
  }

  /**
    * @brief Add a scalar to each coordinate.
    */
  Box& operator+=(size_type scalar)
  {
    m_start += scalar;
    m_stop += scalar;
    return *this;
  }

  /**
   * @brief Subtract a scalar to each coordinate.
   */
  Box& operator-=(size_type scalar)
  {
    m_start -= scalar;
    m_stop -= scalar;
    return *this;
  }

  /**
   * @brief Add 1 to each coordinate.
   */
  Box& operator++()
  {
    return *this += 1;
  }

  Box operator++(int)
  {
    Box out = +(*this);
    ++(*this);
    return out;
  }

  /**
   * @brief Subtract 1 to each coordinate.
   */
  Box& operator--()
  {
    return *this -= 1;
  }

  Box operator--(int)
  {
    Box out = +(*this);
    --(*this);
    return out;
  }

  /**
   * @brief Copy.
   */
  Box operator+() const
  {
    return {+m_start, +m_stop};
  }

  /**
   * @brief Invert the sign of each coordinate.
   */
  Box operator-() const
  {
    return {-m_start, -m_stop};
  }

private:

  value_type m_start; ///< The start bound
  value_type m_stop; ///< The stop bound
};

/// @cond
namespace Internal {

template <typename T, int N, std::size_t... Is>
auto kokkos_execution_policy_impl(const Box<T, N>& domain, std::index_sequence<Is...>)
{
  using Policy = Kokkos::MDRangePolicy<Kokkos::Rank<N>>;
  using Array = Policy::point_type;
  return Policy(Array {domain.start(Is)...}, Array {domain.stop(Is)...});
}

} // namespace Internal
/// @endcond

template <typename T, int N>
auto kokkos_execution_policy(const Box<T, N>& domain)
{
  // FIXME support Properties
  // FIXME support -1?
  if constexpr (N == 1) {
    return Kokkos::RangePolicy(domain.start(0), domain.stop(0));
  } else {
    return Internal::kokkos_execution_policy_impl(domain, std::make_index_sequence<N>());
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
template <typename T, int N>
void for_each(const std::string& label, const Box<T, N>& region, auto&& func)
{
  Kokkos::parallel_for(label, kokkos_execution_policy(region), LINX_FORWARD(func));
}

/**
 * @relatesalso Box
 */
template <typename T, int N>
Box<T, N> operator+(const Box<T, N>& lhs, const auto& rhs)
{
  auto out = +lhs;
  out += rhs;
  return out;
}

/**
 * @relatesalso Box
 */
template <typename T, int N>
Box<T, N> operator-(Box<T, N> lhs, const auto& rhs)
{
  auto out = +lhs;
  out -= rhs;
  return out;
}

template <typename T, int N, typename U, int M>
Box<T, N> operator&(Box<T, N> lhs, const Box<U, M>& rhs)
{
  auto out = +lhs;
  out &= rhs;
  return out;
}

/**
 * @brief Get the 1D slice along the i-th axis.
 */
template <int I, typename T, int N>
Slice<T, SliceType::RightOpen> get(const Box<T, N>& box)
{
  return {box.start(I), box.stop(I)};
}

/// @cond
namespace Internal {

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
  return Box<T, N>({slice_start_impl(get<Is>(slice))...}, {slice_stop_impl(get<Is>(slice))...});
}

} // namespace Internal
/// @endcond

/**
 * @brief Get the bounding box of a slice.
 * 
 * @warning Unbounded slices are not supported, and singleton slices must be integral.
 */
template <typename T, SliceType... TTypes>
Box<T, sizeof...(TTypes)> box(const Slice<T, TTypes...>& slice)
{
  static constexpr int N = sizeof...(TTypes);
  return Internal::box_impl(slice, std::make_index_sequence<N>());
}

/**
 * @brief Make a slice clamped by a box.
 */
template <typename T, typename U, int N, SliceType... TTypes>
auto operator&(const Slice<T, TTypes...>& slice, const Box<U, N>& box)
{
  static constexpr auto Last = sizeof...(TTypes) - 1;
  return (slice.fronts() & box)(clamp(slice.back(), box.start(Last), box.stop(Last)));
}

/**
 * @brief Make a 1D slice clamped by a box.
 */
template <typename T, SliceType TType, typename U, int N>
auto operator&(const Slice<T, TType>& slice, const Box<U, N>& box)
{
  return clamp(slice, box.start(0), box.stop(0));
}

} // namespace Linx

#endif
