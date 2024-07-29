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
  Span = ')' ///< Right-open interval
};

/**
 * @brief ND slice.
 * 
 * Slices are built iteratively by calling `operator()`.
 * For example, Python's `[:, 10, 3:14]` writes `Slice()(10)(3, 14)`.
 */
template <typename T, SliceType TSlice0, SliceType... TSlices>
class Slice {
private:

  friend class Slice<T, TSlices...>;

  template <typename... TArgs>
  Slice(Slice<T, TSlices...> tail, TArgs... args) : m_head(args...), m_tail(LINX_MOVE(tail))
  {}

public:

  using value_type = T; ///< The value type
  static constexpr int Rank = sizeof...(TSlices) + 1; ///< The dimension

  /**
   * @brief Extend the slice over unbounded axis.
   */
  Slice<T, SliceType::Unbounded, TSlice0, TSlices...> operator()() const
  {
    return {*this};
  }

  /**
   * @brief Extend the slice at given value.
   */
  Slice<T, SliceType::Singleton, TSlice0, TSlices...> operator()(T value) const
  {
    return {*this, value};
  }

  /**
   * @brief Extend the slice over given span.
   */
  Slice<T, SliceType::Span, TSlice0, TSlices...> operator()(T start, T stop) const
  {
    return {*this, start, stop};
  }

  /**
   * @brief Extend the slice.
   */
  template <SliceType UType>
  Slice<T, UType, TSlice0, TSlices...> operator()(Slice<T, UType> slice) const
  {
    return {*this, LINX_MOVE(slice)};
  }

  /**
   * @brief Get the 1D slice along i-th axis.
   */
  template <int I>
  constexpr const auto& get() const
  {
    if constexpr (I == Rank - 1) {
      return m_head;
    } else {
      return m_tail.template get<I>();
    }
  }

  /**
   * @brief Make a slice clamped inside a bounding box.
   */
  template <typename U, int N>
  auto operator&(const Box<U, N>& box) const
  {
    return (m_tail & box)(clamp(m_head, box.start(Rank - 1), box.stop(Rank - 1)));
  }

  /**
   * @brief Stream insertion, following Python's syntax.
   * 
   * For example:
   * 
   * \code
   * std::cout << Slice(10)(3, 14) << std::endl;
   * \endcode
   * 
   * prints `10, 3:14`.
   */
  friend std::ostream& operator<<(std::ostream& os, const Slice& slice)
  {
    os << slice.m_tail << ", " << slice.m_head;
    return os;
  }

private:

  Slice<T, TSlice0> m_head; ///< The 1D slice along highest axis value
  Slice<T, TSlices...> m_tail; ///< The other slices in descending axis value
};

/**
 * @brief 1D unbounded specialization.
 */
template <typename T>
class Slice<T, SliceType::Unbounded> {
public:

  using value_type = T;
  static constexpr int Rank = 1;
  static constexpr SliceType Type = SliceType::Unbounded;

  Slice() {}

  Slice<T, SliceType::Unbounded, Type> operator()() const
  {
    return {*this};
  }

  Slice<T, SliceType::Singleton, Type> operator()(T value) const
  {
    return {*this, value};
  }

  Slice<T, SliceType::Span, Type> operator()(T start, T stop) const
  {
    return {*this, start, stop};
  }

  template <SliceType UType>
  Slice<T, UType, Type> operator()(Slice<T, UType> slice) const
  {
    return {*this, LINX_MOVE(slice)};
  }

  template <int I>
  constexpr const auto& get() const
  {
    return *this;
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
 * @brief 1D singleton specialization
 */
template <typename T>
class Slice<T, SliceType::Singleton> {
public:

  using value_type = T;
  static constexpr int Rank = 1;
  static constexpr SliceType Type = SliceType::Singleton;

  Slice(T value) : m_value(value) {}

  Slice<T, SliceType::Unbounded, Type> operator()() const
  {
    return {*this};
  }

  Slice<T, SliceType::Singleton, Type> operator()(T value) const
  {
    return {*this, value};
  }

  Slice<T, SliceType::Span, Type> operator()(T start, T stop) const
  {
    return {*this, start, stop};
  }

  template <SliceType UType>
  Slice<T, UType, Type> operator()(Slice<T, UType> slice) const
  {
    return {*this, LINX_MOVE(slice)};
  }

  template <int I>
  constexpr const auto& get() const
  {
    return *this;
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
class Slice<T, SliceType::Span> {
public:

  using value_type = T;
  static constexpr int Rank = 1;
  static constexpr SliceType Type = SliceType::Span;

  Slice(T start, T stop) : m_start(start), m_stop(stop) {}

  Slice<T, SliceType::Unbounded, Type> operator()() const
  {
    return {*this};
  }

  Slice<T, SliceType::Singleton, Type> operator()(T value) const
  {
    return {*this, value};
  }

  Slice<T, SliceType::Span, Type> operator()(T start, T stop) const
  {
    return {*this, start, stop};
  }

  template <SliceType UType>
  Slice<T, UType, Type> operator()(Slice<T, UType> slice) const
  {
    return {*this, LINX_MOVE(slice)};
  }

  template <int I>
  constexpr const auto& get() const
  {
    return *this;
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

Slice()->Slice<int, SliceType::Unbounded>;

template <typename T>
Slice(T) -> Slice<T, SliceType::Singleton>;

template <typename T>
Slice(T, T) -> Slice<T, SliceType::Span>;

template <int I, typename T, SliceType... TSlices>
const auto& get(const Slice<T, TSlices...>& slice)
{
  return slice.template get<I>();
}

template <int I, typename T, int N>
Slice<T, SliceType::Span> get(const Box<T, N>& box)
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

template <typename T>
T slice_stop_impl(const Slice<T, SliceType::Singleton>& slice)
{
  return slice.value() + 1;
}

template <typename T>
T slice_start_impl(const Slice<T, SliceType::Span>& slice)
{
  return slice.start();
}

template <typename T>
T slice_stop_impl(const Slice<T, SliceType::Span>& slice)
{
  return slice.stop();
}

template <typename TSlice, std::size_t... Is>
auto box_impl(const TSlice& slice, std::index_sequence<Is...>)
{
  using T = typename TSlice::value_type; // FIXME assert T is integral
  static constexpr int N = sizeof...(Is);
  return Box<T, N>({slice_start_impl(get<Is>(slice))...}, {slice_stop_impl(get<Is>(slice))...});
}

} // namespace Internal
/// @endcond

/**
 * @brief Get the bounding box of a slice.
 * 
 * @warning Unbounded slices are not supported.
 */
template <typename T, SliceType... TSlices>
Box<T, sizeof...(TSlices)> box(const Slice<T, TSlices...>& slice)
{
  static constexpr int N = sizeof...(TSlices);
  return Internal::box_impl(slice, std::make_index_sequence<N>());
}

/**
 * @brief Make a 1D slice clamped by a box.
 */
template <typename T, SliceType TSlice, typename U, int N>
auto operator&(const Slice<T, TSlice>& slice, const Box<U, N>& box)
{
  return clamp(slice, box.start(0), box.stop(0));
}

/**
 * @brief Make a 1D slice clamped between bounds.
 */
template <typename T>
Slice<T, SliceType::Span> clamp(const Slice<T, SliceType::Unbounded>&, auto start, auto stop)
{
  return {static_cast<T>(start), static_cast<T>(stop)};
}

/**
 * @brief Make a 1D slice clamped between bounds.
 */
template <typename T>
const Slice<T, SliceType::Singleton>& clamp(const Slice<T, SliceType::Singleton>& slice, auto start, auto stop)
{
  OutOfBoundsError<'[', ')'>::may_throw("slice index", slice.value(), {start, stop});
  return slice;
}

/**
 * @brief Make a 1D slice clamped between bounds.
 */
template <typename T>
Slice<T, SliceType::Span> clamp(const Slice<T, SliceType::Span>& slice, auto start, auto stop)
{
  return {std::max<T>(slice.start(), start), std::min<T>(slice.stop(), stop)};
}

} // namespace Linx

#endif
