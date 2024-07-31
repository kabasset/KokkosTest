// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXDATA_SLICE_H
#define _LINXDATA_SLICE_H

#include "Linx/Base/Exceptions.h"
#include "Linx/Base/Types.h"
#include "Linx/Data/Box.h"

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

/// @cond

template <typename T, SliceType TTypeN, SliceType... TTypes>
class Slice;

/// @endcond

Slice()->Slice<int, SliceType::Unbounded>;

template <typename T>
Slice(T&&) -> Slice<std::decay_t<T>, SliceType::Singleton>;

template <typename T>
Slice(T&&, T&&) -> Slice<std::decay_t<T>, SliceType::RightOpen>;

/**
 * @brief Shortcut for right-open slice.
 */
template <typename T>
using Span = Slice<T, SliceType::RightOpen>;

/**
 * @brief Get the slice along i-th axis.
 */
template <int I, typename T, SliceType... TTypes>
const auto& get(const Slice<T, TTypes...>& slice)
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
Slice<T, TTypeN, TTypes...> slice_push_back(Slice<T, TTypes...> slice, Slice<T, TTypeN> back)
{
  return Slice<T, TTypeN, TTypes...>(slice, back);
}

/**
 * @brief Emplace a 1D slice.
 */
template <typename T, SliceType... TTypes>
auto slice_emplace(Slice<T, TTypes...> slice, auto... args)
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
  Slice(Slice<T, TTypes...> fronts, Slice<T, TTypeN> back) : m_fronts(LINX_MOVE(fronts)), m_back(LINX_MOVE(back)) {}

  /**
   * @brief Extend the slice.
   */
  auto operator()(auto... args) const&
  {
    return slice_emplace(*this, args...);
  }

  /**
   * @brief Extend the slice.
   */
  auto operator()(auto... args) &&
  {
    return slice_emplace(LINX_MOVE(*this), args...);
  }

  const auto& fronts() const
  {
    return m_fronts;
  }

  const auto& back() const
  {
    return m_back;
  }

  /**
   * @brief Get the 1D slice along i-th axis.
   */
  template <int I>
  const auto& get() const
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

  Slice() {}

  auto operator()(auto... args) const&
  {
    return slice_emplace(*this, args...);
  }

  auto operator()(auto... args) &&
  {
    return slice_emplace(LINX_MOVE(*this), args...);
  }

  auto kokkos_slice() const
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

  Slice(T value) : m_value(value) {}

  auto operator()(auto... args) const&
  {
    return slice_emplace(*this, args...);
  }

  auto operator()(auto... args) &&
  {
    return slice_emplace(LINX_MOVE(*this), args...);
  }

  T value() const
  {
    return m_value;
  }

  auto kokkos_slice() const
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

  Slice(T start, T stop) : m_start(start), m_stop(stop) {}

  auto operator()(auto... args) const&
  {
    return slice_emplace(*this, args...);
  }

  auto operator()(auto... args) &&
  {
    return slice_emplace(LINX_MOVE(*this), args...);
  }

  T start() const
  {
    return m_start;
  }

  T stop() const
  {
    return m_stop;
  }

  auto kokkos_slice() const
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

/**
 * @brief Apply a function to each element of the domain.
 */
void for_each(const std::string& label, const Span<std::integral auto>& domain, auto&& func)
{
  Kokkos::parallel_for(label, Kokkos::RangePolicy(domain.start(), domain.stop()), LINX_FORWARD(func));
}

/**
 * @brief Get the 1D slice along the i-th axis.
 */
template <int I, typename T, int N>
Slice<T, SliceType::RightOpen> get(const Box<T, N>& box) // FIXME to Box.h
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

/// @cond
namespace Internal {

template <typename TView, typename TType, std::size_t... Is>
auto slice_impl(const TView& view, const TType& slice, std::index_sequence<Is...>)
{
  return Kokkos::subview(view, get<Is>(slice).kokkos_slice()...);
}

} // namespace Internal
/// @endcond

} // namespace Linx

#endif
