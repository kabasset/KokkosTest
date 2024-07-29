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

#include <Kokkos_Core.hpp>
#include <string>

namespace Linx {

/**
 * @brief An ND bounding box, defined by its start (inclusive) and stop (exclusive) bounds.
 * 
 * @tparam T The coordinate type
 * @tparam N The dimension parameter
 * 
 * If `T` is integral, the box can be iterated with `for_each()` and `kokkos_reduce()`.
 */
template <typename T, int N>
class Box {
public:

  static constexpr int Rank = N; ///< The dimension parameter
  using Container = Kokkos::Array<T, Rank>; ///< The underlying container type

  using value_type = T; ///< The raw coordinate type
  using element_type = std::decay_t<T>; ///< The decayed coordinate type
  using size_type = typename Container::size_type; ///< The index and size type
  using difference_type = std::ptrdiff_t; ///< The index difference type
  using reference = typename Container::reference; ///< The reference type
  using pointer = typename Container::pointer; ///< The pointer type

  /**
   * @brief Constructor.
   */
  Box(const ArrayLike auto& start, const ArrayLike auto& stop)
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
  Box(std::initializer_list<U> start, std::initializer_list<U> stop)
  {
    SizeMismatch::may_throw("bounds", rank(), start, stop);
    auto start_it = start.begin();
    auto stop_it = stop.begin();
    for (std::size_t i = 0; i < rank(); ++i, ++start_it, ++stop_it) {
      m_start[i] = *start_it;
      m_stop[i] = *stop_it;
    }
  }

  KOKKOS_INLINE_FUNCTION auto rank() const
  {
    return m_start.size();
  }

  /**
   * @brief The box shape.
   */
  KOKKOS_INLINE_FUNCTION auto shape() const
  {
    return m_stop - m_start;
  }

  /**
   * @brief The start bound, inclusive.
   */
  KOKKOS_INLINE_FUNCTION const auto& start() const
  {
    return m_start;
  }

  /**
   * @brief The stop bound, exclusive.
   */
  KOKKOS_INLINE_FUNCTION const auto& stop() const
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
  KOKKOS_INLINE_FUNCTION bool operator==(const auto& other) const
  {
    return m_start == other.start() && m_stop == other.stop();
  }

  /**
   * @brief Check whether two boxes are different.
   */
  KOKKOS_INLINE_FUNCTION bool operator!=(const auto& other) const
  {
    return not(*this == other);
  }

  /**
   * @brief Check whether a position lies inside the box.
   */
  KOKKOS_INLINE_FUNCTION bool contains(const ArrayLike auto& position) const
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
  KOKKOS_INLINE_FUNCTION bool contains(auto... is) const // FIXME accept convertible to value_type only
  {
    return contains(Kokkos::Array<value_type, sizeof...(is)> {is...});
  }

  /**
   * @brief Shrink the box inside another box (i.e. get the intersection of both).
   */
  template <typename U, int M>
  Box& operator&=(const Box<U, M>& rhs)
  {
    // FIXME assert rank() == rhs.rank()
    for (std::size_t i = 0; i < rank(); ++i) {
      m_start[i] = std::max<value_type>(m_start[i], rhs.start(i));
      m_stop[i] = std::min<value_type>(m_stop[i], rhs.stop(i));
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
    m_start += extend<Rank>(margin.start());
    m_stop += extend<Rank>(margin.stop());
    return *this;
  }

  /**
   * @brief Shrink the box by a given margin.
   */
  template <typename U, int M>
  Box& operator-=(const Box<U, M>& margin)
  {
    // FIXME allow N=-1
    m_start -= extend<Rank>(margin.start());
    m_stop -= extend<Rank>(margin.stop());
    return *this;
  }

  /**
   * @brief Translate the box by a given vector.
   */
  Box& operator+=(const ArrayLike auto& vector)
  {
    // FIXME allow N=-1
    m_start += extend<Rank>(vector);
    m_stop += extend<Rank>(vector);
    return *this;
  }

  /**
   * @brief Translate the box by the opposite of a given vector.
   */
  Box& operator-=(const ArrayLike auto& vector)
  {
    // FIXME allow N=-1
    m_start -= extend<Rank>(vector);
    m_stop -= extend<Rank>(vector);
    return *this;
  }

  /**
    * @brief Add a scalar to each coordinate.
    */
  Box& operator+=(value_type scalar)
  {
    m_start += scalar;
    m_stop += scalar;
    return *this;
  }

  /**
   * @brief Subtract a scalar to each coordinate.
   */
  Box& operator-=(value_type scalar)
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

  /**
   * @brief Subtract 1 to each coordinate.
   */
  Box& operator--()
  {
    return *this -= 1;
  }

  /**
   * @brief Copy.
   */
  Box operator+()
  {
    return *this;
  }

  /**
   * @brief Invert the sign of each coordinate.
   */
  Box operator-()
  {
    return {-m_start, -m_stop};
  }

private:
private:

  Container m_start; ///< The start bound
  Container m_stop; ///< The stop bound
};

namespace Internal {

template <typename T, int N>
KOKKOS_INLINE_FUNCTION auto kokkos_execution_policy(const Box<T, N>& domain)
{
  // FIXME support Properties
  if constexpr (N == 1) {
    return Kokkos::RangePolicy(domain.start(0), domain.stop(0));
  } else {
    return Kokkos::MDRangePolicy<Kokkos::Rank<N>>(domain.start(), domain.stop());
  }
}

} // namespace Internal

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
  Kokkos::parallel_for(label, Internal::kokkos_execution_policy(region), LINX_FORWARD(func));
}

/**
 * @brief Apply a reduction to the box.
 * 
 * @param label Some label for debugging
 * @param region The region
 * @param projection The projection function
 * @param reducer The reduction function
 * 
 * The projection function takes as input a list of indices and outputs some value.
 * 
 * The reducer satisfies Kokkos' `ReducerConcept`.
 * The `join()` method of the reducer is used for both intra- and inter-thread reduction.
 */
template <typename T, int N>
auto kokkos_reduce(const std::string& label, const Box<T, N>& region, auto&& projection, auto&& reducer)
{
  Kokkos::parallel_reduce(
      label,
      Internal::kokkos_execution_policy(region),
      KOKKOS_LAMBDA(auto&&... args) {
        // args = is..., tmp
        // reducer.join(tmp, projection(is...))
        project_reduce_to(projection, reducer, LINX_FORWARD(args)...);
      },
      LINX_FORWARD(reducer));
  Kokkos::fence();
  return reducer.reference();
}

template <typename T, int N, typename U, int M>
Box<T, N> operator&(Box<T, N> lhs, const Box<U, M>& rhs)
{
  lhs &= rhs;
  return lhs;
}

} // namespace Linx

#endif
