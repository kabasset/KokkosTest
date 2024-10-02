// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXDATA_RANDOM_H
#define _LINXDATA_RANDOM_H

#include "Linx/Base/Types.h"
#include "Linx/Data/Slice.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

namespace Linx {

/**
 * @brief Uniform noise generator.
 * 
 * \code
 * auto noise = Linx::Sequence<int, 100>("noise").generate("random", Linx::UniformNoise(0, 1000, 42));
 * auto noise = Linx::random<100>("noise", Linx::Span(0, 1000), 42);
 * \endcode
 */
template <typename T>
class UniformNoise {
public:

  UniformNoise(T min = Limits<T>::half_min(), T max = Limits<T>::half_max(), Index seed = -1) :
      m_min(min), m_max(max), m_pool(seed + 1) // Random seed iff m_pool(0)
  {}

  UniformNoise(const Span<T>& bouds, Index seed = -1) : UniformNoise(bouds.start(), bouds.stop(), seed) {}

  KOKKOS_INLINE_FUNCTION T operator()() const
  {
    auto generator = m_pool.get_state();
    auto out = Kokkos::rand<decltype(generator), T>::draw(generator, m_min, m_max);
    m_pool.free_state(generator);
    return out;
  }

private:

  T m_min;
  T m_max;
  Kokkos::Random_XorShift64_Pool<> m_pool; // FIXME tparam
};

} // namespace Linx

#endif
