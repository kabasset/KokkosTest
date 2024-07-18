// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXDATA_SLICE_H
#define _LINXDATA_SLICE_H

#include "Linx/Base/Types.h"

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
 */
template <typename T, SliceType TType0, SliceType... TTypes>
class Slice {
private:

  friend class Slice<T, TTypes...>;

  template <typename... TArgs>
  Slice(Slice<T, TTypes...> tail, TArgs... args) : m_head(args...), m_tail(LINX_MOVE(tail))
  {}

public:

  using value_type = T;
  static constexpr int Rank = sizeof...(TTypes) + 1;

  Slice<T, SliceType::Unbounded, TType0, TTypes...> operator()() const
  {
    return {*this};
  }

  Slice<T, SliceType::Singleton, TType0, TTypes...> operator()(T index) const
  {
    return {*this, index};
  }

  Slice<T, SliceType::Span, TType0, TTypes...> operator()(T start, T stop) const
  {
    return {*this, start, stop};
  }

  template <int I>
  constexpr const auto& get() const
  {
    if constexpr (I == Rank - 1) {
      return m_head;
    } else {
      return m_tail.template get<I>();
    }
  }

  friend std::ostream& operator<<(std::ostream& os, const Slice& slice)
  {
    os << slice.m_tail << ", " << slice.m_head;
    return os;
  }

private:

  Slice<T, TType0> m_head;
  Slice<T, TTypes...> m_tail;
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
    return m_index + 1;
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

} // namespace Linx

#endif
