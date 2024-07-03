// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXDATA_BOX_H
#define _LINXDATA_BOX_H

#include "Linx/Base/TypeUtils.h"
#include "Linx/Data/Traits.h"
#include "Linx/Data/Vector.h"

#include <Kokkos_Core.hpp>
#include <string>

namespace Linx {

template <typename T, int N>
class Box {
public:

  static constexpr int Rank = N;
  using Container = Kokkos::Array<std::int64_t, Rank>; // FIXME Vector<T, N>?

  using value_type = T;
  using element_type = std::decay_t<T>;
  using size_type = typename Container::size_type;
  using difference_type = std::ptrdiff_t;
  using reference = typename Container::reference;
  using pointer = typename Container::pointer;
  using iterator = pointer;

  template <typename TContainer> // FIXME range?
  Box(TContainer&& f, TContainer&& b)
  {
    for (int i = 0; i < Rank; ++i) {
      m_front[i] = f[i];
      m_back[i] = b[i];
    }
  }

  const auto& front() const
  {
    return m_front;
  }
  const auto& back() const
  {
    return m_back;
  }

  template <typename TFunc>
  KOKKOS_INLINE_FUNCTION void iterate(const std::string& name, TFunc&& func) const
  {
    Kokkos::parallel_for(name, Kokkos::MDRangePolicy<Kokkos::Rank<Rank>>(m_front, m_back), LINX_FORWARD(func));
  }

private:

  Container m_front;
  Container m_back;
};

} // namespace Linx

#endif
