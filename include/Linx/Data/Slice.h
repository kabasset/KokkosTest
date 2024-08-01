// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXDATA_SLICE_H
#define _LINXDATA_SLICE_H

#include "Linx/Base/Exceptions.h"
#include "Linx/Base/Types.h"

#include <Kokkos_Core.hpp>
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
Slice(const T&) -> Slice<T, SliceType::Singleton>;

template <typename T>
Slice(const T&, const T&) -> Slice<T, SliceType::RightOpen>;

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
void for_each(const std::string& label, const Span<std::integral auto>& region, auto&& func)
{
  Kokkos::parallel_for(label, Kokkos::RangePolicy(region.start(), region.stop()), LINX_FORWARD(func));
}

/**
 * @brief Apply a reduction to the span.
 */
auto kokkos_reduce(const std::string& label, const Span<std::integral auto>& region, auto&& projection, auto&& reducer)
{
  Kokkos::parallel_reduce(
      label,
      Kokkos::RangePolicy(region.start(), region.stop()),
      KOKKOS_LAMBDA(auto&&... args) {
        // args = is..., tmp
        // reducer.join(tmp, projection(is...))
        project_reduce_to(projection, reducer, LINX_FORWARD(args)...);
      },
      LINX_FORWARD(reducer));
  return reducer.reference();
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
