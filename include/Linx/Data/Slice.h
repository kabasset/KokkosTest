// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXDATA_SLICE_H
#define _LINXDATA_SLICE_H

#include "Linx/Base/Types.h"
#include "Linx/Data/Box.h"

#include <Kokkos_Core.hpp>
#include <concepts>
#include <ostream>

namespace Linx {

/**
 * @brief Type of 1D slicing.
 */
enum class SliceType {
  Unbounded = 0, ///< Unbounded
  Singleton = 1, ///< Single value
  Span = 2 ///< Contiguous range
};

/**
 * @brief ND slice.
 * 
 * Slices are built iteratively by calling `operator()`.
 * For example, Python's `[:, 10, 3:14]` writes `Slice()(10)(3, 14)`.
 */
template <typename T, SliceType TType0, SliceType... TTypes>
class Slice {
private:

  friend class Slice<T, TTypes...>;

  template <typename... TArgs>
  Slice(Slice<T, TTypes...> tail, TArgs... args) : m_head(args...), m_tail(LINX_MOVE(tail))
  {}

public:

  using value_type = T; ///< The value type
  static constexpr int Rank = sizeof...(TTypes) + 1; ///< The dimension

  /**
   * @brief Extend the slice over unbounded axis.
   */
  Slice<T, SliceType::Unbounded, TType0, TTypes...> operator()() const
  {
    return {*this};
  }

  /**
   * @brief Extend the slice at given index.
   */
  Slice<T, SliceType::Singleton, TType0, TTypes...> operator()(T index) const
  {
    return {*this, index};
  }

  /**
   * @brief Extend the slice over given span.
   */
  Slice<T, SliceType::Span, TType0, TTypes...> operator()(T start, T stop) const
  {
    return {*this, start, stop};
  }

  /**
   * @brief Extend the slice.
   */
  template <SliceType UType>
  Slice<T, UType, TType0, TTypes...> operator()(Slice<T, UType> slice) const
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
  friend auto clamp(const Slice slice, const Box<U, N>& box)
  {
    return clamp(slice.m_tail, box)(clamp(slice.m_head, box.start(Rank - 1), box.stop(Rank - 1)));
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

  Slice<T, TType0> m_head; ///< The 1D slice along highest axis index
  Slice<T, TTypes...> m_tail; ///< The other slices in descending axis index
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

  Slice<T, SliceType::Singleton, Type> operator()(T index) const
  {
    return {*this, index};
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

  Slice(T index) : m_index(index) {}

  Slice<T, SliceType::Unbounded, Type> operator()() const
  {
    return {*this};
  }

  Slice<T, SliceType::Singleton, Type> operator()(T index) const
  {
    return {*this, index};
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
    return m_index;
  }

  T stop() const
  {
    return m_index + 1; // FIXME + epsilon
  }

  auto kokkos_slice() const
  {
    return m_index;
  }

  friend std::ostream& operator<<(std::ostream& os, const Slice& slice)
  {
    os << slice.m_index;
    return os;
  }

private:

  T m_index;
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

  Slice<T, SliceType::Singleton, Type> operator()(T index) const
  {
    return {*this, index};
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

template <int I, typename T, SliceType... TTypes>
const auto& get(const Slice<T, TTypes...>& slice)
{
  return slice.template get<I>();
}

/// @cond
namespace Internal {

template <typename TSlice, std::size_t... Is>
auto box_impl(const TSlice& slice, std::index_sequence<Is...>)
{
  using T = typename TSlice::value_type;
  static constexpr int N = sizeof...(Is);
  return Box<T, N>({get<Is>(slice).start()...}, {get<Is>(slice).stop()...});
}

} // namespace Internal
/// @endcond

/**
 * @brief Get the bounding box of a slice.
 * 
 * @warning Unbounded slices are not supported.
 */
template <typename T, SliceType... TTypes>
Box<T, sizeof...(TTypes)> box(const Slice<T, TTypes...>& slice)
{
  static constexpr int N = sizeof...(TTypes);
  return Internal::box_impl(slice, std::make_index_sequence<N>());
}

/**
 * @brief Make a 1D slice clamped by a box.
 */
template <typename T, SliceType TType, typename U, int N>
auto clamp(const Slice<T, TType>& slice, const Box<U, N>& box)
{
  return clamp(slice, box.start(0), box.stop(0));
}

/**
 * @brief Make a 1D slice clamped between bounds.
 */
template <typename T, SliceType TType>
Slice<T, SliceType::Span> clamp(const Slice<T, TType>& slice, auto start, auto stop)
{
  if constexpr (TType == SliceType::Unbounded) {
    return {static_cast<T>(start), static_cast<T>(stop)};
  } else {
    return {std::max<T>(slice.start(), start), std::min<T>(slice.stop(), stop)};
  }
}

} // namespace Linx

#endif
