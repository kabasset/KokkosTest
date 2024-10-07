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
 * auto noise = Linx::generate<100>("noise", Linx::UniformNoise(0, 10));
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

/**
 * @brief Gaussian noise generator.
 * 
 * \code
 * auto noise = Linx::generate<100>("noise", Linx::GaussianNoise(100, 15));
 * \endcode
 */
template <typename T>
class GaussianNoise {
public:

  GaussianNoise(T mu = Limits<T>::zero(), T sigma = Limits<T>::one(), Index seed = -1) :
      m_mu(mu), m_sigma(sigma), m_pool(seed + 1) // Random seed iff m_pool(0)
  {}

  KOKKOS_INLINE_FUNCTION T operator()() const
  {
    // Box-Muller method

    auto generator = m_pool.get_state();
    auto u = Kokkos::rand<decltype(generator), double>::draw(generator, -1, 0); // [-1, 0) excludes 0
    auto theta = Kokkos::rand<decltype(generator), double>::draw(generator, 0, 2 * std::numbers::pi);
    m_pool.free_state(generator);

    const double r = Kokkos::sqrt(-2 * Kokkos::log(-u)); // -u in (0, 1]
    const double x = r * Kokkos::cos(theta);

    if constexpr (is_complex<T>()) { // Get two variables at once
      const double y = r * Kokkos::sin(theta);
      return T(x, y) * m_sigma + m_mu;
    } else {
      return x * m_sigma + m_mu;
    }
  }

  KOKKOS_INLINE_FUNCTION T operator()(const T& in) const
  {
    return in + operator()();
  }

private:

  T m_mu;
  T m_sigma;
  Kokkos::Random_XorShift64_Pool<> m_pool; // FIXME tparam
};

/**
 * @brief Poisson noise generator.
 */
class PoissonNoise {
public:

  PoissonNoise(Index seed = -1) : m_pool(seed + 1) // Random seed iff m_pool(0)
  {}

  template <typename T>
  KOKKOS_INLINE_FUNCTION T operator()(const T& in) const
  {
    // For stability, generate u even when in <= 0
    auto generator = m_pool.get_state();
    auto u = Kokkos::rand<decltype(generator), double>::draw(generator, 0, 1);
    m_pool.free_state(generator);

    if (in <= 0 || u == 0) { // FIXME needed?
      return 0;
    }

    // FIXME support complex?
    auto mu = static_cast<double>(in);
    auto p = std::exp(-mu);
    auto cp = 0.0;
    T k {};
    while (cp < u) {
      cp += p;
      ++k;
      p *= mu / k;
    }

    return k - 1;
  }

private:

  Kokkos::Random_XorShift64_Pool<> m_pool; // FIXME tparam
};

} // namespace Linx

#endif
