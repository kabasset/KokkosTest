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
  Box(const TContainer& f, const TContainer& b)
  {
    for (int i = 0; i < Rank; ++i) {
      m_front[i] = f[i];
      m_back[i] = b[i];
    }
  }

  KOKKOS_FORCEINLINE_FUNCTION const auto& front() const
  {
    return m_front;
  }

  KOKKOS_FORCEINLINE_FUNCTION const auto& back() const
  {
    return m_back;
  }

  KOKKOS_FORCEINLINE_FUNCTION auto extent(std::integral auto i) const
  {
    return m_back[i] - m_front[i];
  }

  KOKKOS_FORCEINLINE_FUNCTION auto size() const
  {
    auto out = 1;
    for (std::size_t i = 0; i < m_front.size(); ++i) {
      out *= extent(i);
    }
    return out;
  }

  template <typename TFunc>
  KOKKOS_INLINE_FUNCTION void iterate(const std::string& name, TFunc&& func) const
  {
    Kokkos::parallel_for(name, Kokkos::MDRangePolicy<Kokkos::Rank<Rank>>(m_front, m_back), LINX_FORWARD(func));
  }

  template <typename TProj, typename TRed>
  KOKKOS_INLINE_FUNCTION TRed::value_type reduce(const std::string& name, TProj&& projection, TRed&& reducer) const
  {
    Kokkos::parallel_reduce(
        name,
        Kokkos::MDRangePolicy<Kokkos::Rank<Rank>>(m_front, m_back),
        KOKKOS_LAMBDA(auto&&... args) {
          // args = is..., tmp
          // reducer.join(tmp, projection(is...))
          project_reduce_to(projection, reducer, LINX_FORWARD(args)...);
        },
        LINX_FORWARD(reducer));
    return reducer.reference();
  }

private:

  Container m_front;
  Container m_back;
};

} // namespace Linx

#endif
