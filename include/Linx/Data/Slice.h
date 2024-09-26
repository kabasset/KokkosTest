// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXDATA_SLICE_H
#define _LINXDATA_SLICE_H

#include "Linx/Base/Exceptions.h"
#include "Linx/Base/Types.h"

#include <Kokkos_Core.hpp>
#include <concepts>
#include <ostream>

namespace Linx {

/**
 * @brief Type of 1D slice.
 */
enum class SliceType : char {
  Unbounded = '*', ///< Unbounded
  Singleton = '=', ///< Single value
  Closed = ']', ///< Closed interval
  RightOpen = ')' ///< Right-open interval, a.k.a. span
};

/**
 * @brief Strong type for extent-based constructors.
 * 
 * \code
 * auto slice = Slice(10, Size(3));
 * \endcode
 * 
 */
template <typename T>
using Size = Wrap<T, struct ExtentTag>;

/// @cond

template <typename T, SliceType TTypeN, SliceType... TTypes>
class Slice;

/// @endcond

Slice()->Slice<int, SliceType::Unbounded>;

template <typename T>
Slice(const T&) -> Slice<T, SliceType::Singleton>;

template <typename T>
Slice(const T&, const T&) -> Slice<T, SliceType::RightOpen>;

template <typename T, typename U>
Slice(const T&, const Size<U>&) -> Slice<T, SliceType::RightOpen>;

/**
 * @brief Shortcut for right-open slice.
 */
template <typename T>
using Span = Slice<T, SliceType::RightOpen>;

/**
 * @brief Get the slice along i-th axis.
 */
template <int I, typename T, SliceType... TTypes>
KOKKOS_INLINE_FUNCTION constexpr auto& get(const Slice<T, TTypes...>& slice)
{
  if constexpr (sizeof...(TTypes) == 1) {
    return slice;
  } else {
    return slice.template get<I>();
  }
}

/**
 * @brief Append a 1D slice.
 */
template <typename T, SliceType TTypeN, SliceType... TTypes>
KOKKOS_INLINE_FUNCTION Slice<T, TTypeN, TTypes...> slice_push_back(Slice<T, TTypes...> slice, Slice<T, TTypeN> back)
{
  return Slice<T, TTypeN, TTypes...>(slice, back);
}

/**
 * @brief Emplace a 1D slice.
 */
template <typename T, SliceType... TTypes>
KOKKOS_INLINE_FUNCTION auto slice_emplace(Slice<T, TTypes...> slice, auto... args)
{
  return slice_push_back(slice, Slice(args...));
}

/**
 * @brief ND slice.
 * 
 * Slices are built iteratively by calling `operator()`.
 * For example, Python's `[:, 10, 3:14]` writes `Slice()(10)(3, 14)`.
 * 
 * Slices are similar to bounding boxes, except that:
 * - slices can be unbounded;
 * - slices are defined axis-by-axis while boxes are defined by two ND positions.
 */
template <typename T, SliceType TTypeN, SliceType... TTypes>
class Slice {
public:

  using size_type = T; ///< The value type
  static constexpr int Rank = sizeof...(TTypes) + 1; ///< The dimension

  /**
   * @brief Constructor.
   * 
   * Prefer creating slices using the `operator()` syntax.
   */
  KOKKOS_INLINE_FUNCTION Slice(Slice<T, TTypes...> fronts, Slice<T, TTypeN> back) :
      m_fronts(LINX_MOVE(fronts)), m_back(LINX_MOVE(back))
  {}

  /**
   * @brief Extend the slice.
   */
  KOKKOS_INLINE_FUNCTION auto operator()(auto... args) const&
  {
    return slice_emplace(*this, args...);
  }

  /**
   * @brief Extend the slice.
   */
  KOKKOS_INLINE_FUNCTION auto operator()(auto... args) &&
  {
    return slice_emplace(LINX_MOVE(*this), args...);
  }

  KOKKOS_INLINE_FUNCTION const auto& fronts() const
  {
    return m_fronts;
  }

  KOKKOS_INLINE_FUNCTION const auto& back() const
  {
    return m_back;
  }

  /**
   * @brief Get the 1D slice along i-th axis.
   */
  template <int I>
  KOKKOS_INLINE_FUNCTION constexpr auto& get() const
  {
    if constexpr (I == Rank - 1) {
      return m_back;
    } else {
      return Linx::get<I>(m_fronts);
    }
  }

  /**
   * @brief Stream insertion, following Python's syntax.
   * 
   * For example:
   * 
   * \code
   * std::cout << Slice(10)()(3, 14) << std::endl;
   * \endcode
   * 
   * prints `10, :, 3:14`.
   */
  friend std::ostream& operator<<(std::ostream& os, const Slice& slice)
  {
    os << slice.m_fronts << ", " << slice.m_back;
    return os;
  }

private:

  Slice<T, TTypes...> m_fronts; ///< The front slices
  Slice<T, TTypeN> m_back; ///< The back slice
};

/**
 * @brief 1D unbounded specialization.
 */
template <typename T>
class Slice<T, SliceType::Unbounded> {
public:

  using size_type = T;
  static constexpr int Rank = 1;
  static constexpr SliceType Type = SliceType::Unbounded;

  KOKKOS_INLINE_FUNCTION Slice() {}

  KOKKOS_INLINE_FUNCTION auto operator()(auto... args) const&
  {
    return slice_emplace(*this, args...);
  }

  KOKKOS_INLINE_FUNCTION auto operator()(auto... args) &&
  {
    return slice_emplace(LINX_MOVE(*this), args...);
  }

  KOKKOS_INLINE_FUNCTION auto kokkos_slice() const // FIXME free function
  {
    return Kokkos::ALL;
  }

  friend std::ostream& operator<<(std::ostream& os, const Slice&)
  {
    os << ':';
    return os;
  }
};

/**
 * @brief 1D singleton specialization.
 */
template <typename T>
class Slice<T, SliceType::Singleton> {
public:

  using size_type = T;
  static constexpr int Rank = 1;
  static constexpr SliceType Type = SliceType::Singleton;

  KOKKOS_INLINE_FUNCTION Slice(T value) : m_value(value) {}

  KOKKOS_INLINE_FUNCTION auto operator()(auto... args) const&
  {
    return slice_emplace(*this, args...);
  }

  KOKKOS_INLINE_FUNCTION auto operator()(auto... args) &&
  {
    return slice_emplace(LINX_MOVE(*this), args...);
  }

  KOKKOS_INLINE_FUNCTION T value() const
  {
    return m_value;
  }

  KOKKOS_INLINE_FUNCTION auto kokkos_slice() const // FIXME free function
  {
    return m_value;
  }

  friend std::ostream& operator<<(std::ostream& os, const Slice& slice)
  {
    os << slice.m_value;
    return os;
  }

private:

  T m_value;
};

/**
 * @brief 1D span specialization.
 */
template <typename T>
class Slice<T, SliceType::RightOpen> {
public:

  using size_type = T;
  static constexpr int Rank = 1;
  static constexpr SliceType Type = SliceType::RightOpen;

  KOKKOS_INLINE_FUNCTION Slice(const T& start, const T& stop) : m_start(start), m_stop(stop) {}

  KOKKOS_INLINE_FUNCTION Slice(const T& start, const Size<T>& size) : m_start(start), m_stop(m_start + size.value) {}

  KOKKOS_INLINE_FUNCTION auto operator()(auto... args) const&
  {
    return slice_emplace(*this, args...);
  }

  KOKKOS_INLINE_FUNCTION auto operator()(auto... args) &&
  {
    return slice_emplace(LINX_MOVE(*this), args...);
  }

  KOKKOS_INLINE_FUNCTION T start() const
  {
    return m_start;
  }

  KOKKOS_INLINE_FUNCTION T stop() const
  {
    return m_stop;
  }

  KOKKOS_INLINE_FUNCTION T size() const
  {
    return m_stop - m_start;
  }

  KOKKOS_INLINE_FUNCTION auto kokkos_slice() const // FIXME free function
  {
    return Kokkos::pair(m_start, m_stop);
  }

  friend std::ostream& operator<<(std::ostream& os, const Slice& slice)
  {
    os << slice.m_start << ':' << slice.m_stop;
    return os;
  }

private:

  T m_start;
  T m_stop;
};

template <typename TSpace, std::integral T>
auto kokkos_execution_policy(const Span<T>& region)
{
  return Kokkos::RangePolicy<TSpace>(region.start(), region.stop());
}

/**
 * @brief Apply a function to each element of the domain.
 * @tparam TSpace The execution space
 */
template <typename TSpace = Kokkos::DefaultExecutionSpace, std::integral T>
void for_each(const std::string& label, const Span<T>& region, auto&& func)
{
  Kokkos::parallel_for(label, kokkos_execution_policy<TSpace>(region), LINX_FORWARD(func));
}

/**
 * @brief Make a 1D slice clamped between bounds.
 */
template <typename T>
Slice<T, SliceType::RightOpen> clamp(const Slice<T, SliceType::Unbounded>&, auto start, auto stop)
{
  return {static_cast<T>(start), static_cast<T>(stop)};
}

/**
 * @brief Make a 1D slice clamped between bounds.
 */
template <typename T>
const Slice<T, SliceType::Singleton>& clamp(const Slice<T, SliceType::Singleton>& slice, auto start, auto stop)
{
  OutOfBounds<'[', ')'>::may_throw("slice index", slice.value(), {start, stop});
  return slice;
}

/**
 * @brief Make a 1D slice clamped between bounds.
 */
template <typename T>
Slice<T, SliceType::RightOpen> clamp(const Slice<T, SliceType::RightOpen>& slice, auto start, auto stop)
{
  return {std::max<T>(slice.start(), start), std::min<T>(slice.stop(), stop)};
}

} // namespace Linx

#endif
