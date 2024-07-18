// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXDATA_BOX_H
#define _LINXDATA_BOX_H

#include "Linx/Base/Containers.h"
#include "Linx/Base/Packs.h"
#include "Linx/Base/Types.h"

#include <Kokkos_Core.hpp>
#include <string>

namespace Linx {

template <typename T, int N>
class Box {
public:

  static constexpr int Rank = N;
  using Container = Kokkos::Array<std::int64_t, Rank>; // FIXME Sequence<T, N>?

  using value_type = T;
  using element_type = std::decay_t<T>;
  using size_type = typename Container::size_type;
  using difference_type = std::ptrdiff_t;
  using reference = typename Container::reference;
  using pointer = typename Container::pointer;
  using iterator = pointer;

  template <typename TContainer> // FIXME range?
  Box(const TContainer& start, const TContainer& stop)
  {
    for (int i = 0; i < Rank; ++i) {
      m_start[i] = start[i];
      m_stop[i] = stop[i];
    }
  }

  template <typename U>
  Box(std::initializer_list<U> start, std::initializer_list<U> stop)
  {
    auto start_it = start.begin();
    auto stop_it = stop.begin();
    for (int i = 0; i < Rank; ++i, ++start_it, ++stop_it) {
      m_start[i] = *start_it;
      m_stop[i] = *stop_it;
    }
  }

  KOKKOS_INLINE_FUNCTION const auto& start() const // FIXME start
  {
    return m_start;
  }

  KOKKOS_INLINE_FUNCTION const auto& stop() const // FIXME stop
  {
    return m_stop;
  }

  KOKKOS_INLINE_FUNCTION auto start(std::integral auto i) const
  {
    return m_start[i];
  }

  KOKKOS_INLINE_FUNCTION auto& start(std::integral auto i)
  {
    return m_start[i];
  }

  KOKKOS_INLINE_FUNCTION auto stop(std::integral auto i) const
  {
    return m_stop[i];
  }

  KOKKOS_INLINE_FUNCTION auto& stop(std::integral auto i)
  {
    return m_stop[i];
  }

  KOKKOS_INLINE_FUNCTION auto extent(std::integral auto i) const
  {
    return m_stop[i] - m_start[i];
  }

  KOKKOS_INLINE_FUNCTION auto size() const
  {
    auto out = 1;
    for (std::size_t i = 0; i < m_start.size(); ++i) {
      out *= extent(i);
    }
    return out;
  }

  template <typename TFunc>
  KOKKOS_INLINE_FUNCTION void iterate(const std::string& name, TFunc&& func) const
  {
    Kokkos::parallel_for(name, kokkos_execution_policy(), LINX_FORWARD(func));
  }

  template <typename TProj, typename TRed>
  KOKKOS_INLINE_FUNCTION TRed::value_type reduce(const std::string& name, TProj&& projection, TRed&& reducer) const
  {
    Kokkos::parallel_reduce(
        name,
        kokkos_execution_policy(),
        KOKKOS_LAMBDA(auto&&... args) {
          // args = is..., tmp
          // reducer.join(tmp, projection(is...))
          project_reduce_to(projection, reducer, LINX_FORWARD(args)...);
        },
        LINX_FORWARD(reducer));
    // FIXME fence?
    return reducer.reference();
  }

  auto kokkos_execution_policy() const
  {
    if constexpr (Rank == 1) {
      return Kokkos::RangePolicy(m_start[0], m_stop[0]);
    } else {
      return Kokkos::MDRangePolicy<Kokkos::Rank<Rank>>(m_start, m_stop);
    }
  }

private:

  Container m_start;
  Container m_stop;
};

} // namespace Linx

#endif
