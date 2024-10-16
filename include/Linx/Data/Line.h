// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXDATA_LINE_H
#define _LINXDATA_LINE_H

#include "Linx/Base/Packs.h"
#include "Linx/Base/Types.h"

#include <Kokkos_Core.hpp>

namespace Linx {

template <typename T, int I, int N>
class GLine {
public:

  static constexpr int Rank = N;
  static constexpr int Axis = I;

  using size_type = T;
  using value_type = GPosition<size_type, Rank>;

  GLine() : m_start {}, m_stop(0), m_step(1) {}

  GLine(value_type start, size_type stop, size_type step = 1) : m_start(+start), m_stop(stop), m_step(step) {}

  KOKKOS_INLINE_FUNCTION constexpr const value_type& start() const
  {
    return m_start;
  }

  KOKKOS_INLINE_FUNCTION constexpr size_type size() const
  {
    return m_stop - m_start[Axis]; // FIXME m_step
  }

  KOKKOS_INLINE_FUNCTION constexpr size_type start(int i) const
  {
    return m_start[i];
  }

  KOKKOS_INLINE_FUNCTION constexpr size_type stop(int = 0) const
  {
    return m_stop;
  }

  value_type operator()(int i) const
  {
    auto out = +m_start;
    out[Axis] += i * m_step;
    return out;
  }

  KOKKOS_INLINE_FUNCTION GLine& add(auto... values)
  {
    add_impl(forward_as_tuple(values...), std::make_index_sequence<sizeof...(values)>());
    return *this;
  }

  KOKKOS_INLINE_FUNCTION GLine& subtract(auto... values)
  {
    subtract_impl(forward_as_tuple(values...), std::make_index_sequence<sizeof...(values)>());
    return *this;
  }

  KOKKOS_INLINE_FUNCTION GLine& operator+=(const auto& delta)
  {
    m_start += delta;
    m_stop += delta[Axis];
    return *this;
  }

  KOKKOS_INLINE_FUNCTION GLine& operator-=(const auto& delta)
  {
    m_start -= delta;
    m_stop -= delta[Axis];
    return *this;
  }

private:

  template <std::size_t... Is>
  KOKKOS_INLINE_FUNCTION void add_impl(const auto& values, std::index_sequence<Is...>)
  {
    ((m_start[Is] += get<Is>(values)), ...);
    m_stop += get<Axis>(values);
  }

  template <std::size_t... Is>
  KOKKOS_INLINE_FUNCTION void subtract_impl(const auto& values, std::index_sequence<Is...>)
  {
    ((m_start[Is] -= get<Is>(values)), ...);
    m_stop -= get<Axis>(values);
  }

  value_type m_start;
  size_type m_stop;
  size_type m_step;
};

template <typename T, int I, int N>
using Line = GLine<Index, I, N>;

} // namespace Linx

#endif
