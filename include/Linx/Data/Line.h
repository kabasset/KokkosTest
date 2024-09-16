// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXDATA_LINE_H
#define _LINXDATA_LINE_H

#include "Linx/Base/Types.h"

#include <Kokkos_Core.hpp>

namespace Linx {

template <typename T, int I, int N>
class Line {
public:

  static constexpr int Rank = N;
  static constexpr int Axis = I;

  using size_type = T;
  using value_type = Position<size_type, Rank>;

  Line(value_type start, T stop, T step = 1) : m_start(LINX_MOVE(start)), m_stop(stop), m_step(step) {}

  KOKKOS_INLINE_FUNCTION T size() const
  {
    return m_stop - m_start[I]; // FIXME m_step
  }

  KOKKOS_INLINE_FUNCTION value_type operator()(int i) const
  {
    auto out = +m_start;
    out[I] += i * m_step;
    return out;
  }

private:

  value_type m_start;
  T m_stop;
  T m_step;
};

} // namespace Linx

#endif
