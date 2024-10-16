// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXBASE_RANDOM_H
#define _LINXBASE_RANDOM_H

#include "Linx/Base/Slice.h"
#include "Linx/Base/Types.h"
#include "Linx/Base/mixins/Arithmetic.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <boost/math/special_functions.hpp>
#include <chrono> // For random seed

namespace Linx {

/**
 * @brief Pool of random number generators for parallel usage.
 */
template <typename T, typename TPool = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>>
class RngPool {
public:

  /**
   * @brief Generator, to be used with RAII.
   */
  class Generator {
  public:

    /**
     * @brief Constructor (acquire state).
     */
    KOKKOS_INLINE_FUNCTION explicit Generator(const auto& pool) : m_pool(pool), m_gen(m_pool.get_state()) {}

    /**
     * @brief Destructor (free state).
     */
    KOKKOS_INLINE_FUNCTION ~Generator()
    {
      m_pool.free_state(m_gen);
    }

    /**
     * @brief Draw a single, uniformly distributed sample.
     */
    template <typename U>
    KOKKOS_INLINE_FUNCTION U uniform(U start, U stop)
    {
      return Kokkos::rand<decltype(m_gen), U>::draw(m_gen, start, stop);
    }

  private:

    const TPool& m_pool; ///< The Kokkos pool
    typename TPool::generator_type m_gen; ///< The Kokkos generator
  };

  /**
   * @brief Constructor.
   * @param seed The random seed, or -1 to get a time-dependent seed
   */
  RngPool(Index seed = -1) : m_pool(seed != -1 ? seed : std::chrono::system_clock::now().time_since_epoch().count()) {}

  /**
   * @brief Draw a single, uniformly distributed sample.
   * 
   * For more complex cases, use `generator()`.
   */
  KOKKOS_INLINE_FUNCTION T uniform(T start, T stop) const
  {
    return generator().uniform(start, stop);
  }

  /**
   * @brief Get a generator.
   * 
   * The generator acquires and releases pool a state through RAII, for example:
   * 
   * \code
   * double u;
   * {
   *   auto gen = pool.generator();
   *   u = gen.uniform(0, 1);
   * }
   * \endcode
   */
  KOKKOS_INLINE_FUNCTION auto generator() const
  {
    return Generator(m_pool);
  }

private:

  TPool m_pool; // FIXME tparam
};

/**
 * @brief Uniform probability distribution.
 */
template <typename T>
class UniformDistribution {
public:

  /**
   * @brief Constructor.
   */
  KOKKOS_INLINE_FUNCTION UniformDistribution(Span<T> span) : m_span(LINX_MOVE(span)) {}

  /**
   * @brief Constructor.
   */
  KOKKOS_INLINE_FUNCTION UniformDistribution(T start, T stop) : m_span(start, stop) {}

  /**
   * @brief Lower bound (inclusive).
   */
  KOKKOS_INLINE_FUNCTION T start() const
  {
    return m_span.start();
  }

  /**
   * @brief Upper bound (exclusive).
   */
  KOKKOS_INLINE_FUNCTION T stop() const
  {
    return m_span.stop();
  }

  /**
   * @brief Probability density function.
   */
  KOKKOS_INLINE_FUNCTION double pdf(auto x) const
  {
    const auto& a = m_span.start();
    const auto& b = m_span.stop();
    return m_span.contains(x) ? 1. / (b - a) : 0.;
  }

  /**
   * @brief Cumulative density function.
   */
  KOKKOS_INLINE_FUNCTION double cdf(auto x) const
  {
    if (x <= m_span.start()) {
      return 0.;
    }
    if (x >= m_span.stop()) {
      return 1.;
    }
    const auto& a = m_span.start();
    const auto& b = m_span.stop();
    return double(x - a) / (b - a);
  }

private:

  Span<T> m_span; ///< Bounds
};

/**
 * @brief Uniform random number generator.
 * 
 * \code
 * auto noise = Linx::generate<100>("noise", Linx::UniformRng(0., 1.));
 * \endcode
 */
template <typename T, typename TSpace = Kokkos::DefaultExecutionSpace>
class UniformRng : public ArithmeticMixin<void, const T, UniformRng<T, TSpace>> {
public:

  using value_type = const T;

  /**
   * @brief Constructor.
   */
  UniformRng(UniformDistribution<T> distribution, Index seed = -1) :
      m_distribution(LINX_MOVE(distribution)), m_pool(seed)
  {}

  std::string label() const
  {
    return "Uniform"; // FIXME parameters
  }

  /**
   * @brief The distribution.
   */
  KOKKOS_INLINE_FUNCTION const auto& distribution() const
  {
    return m_distribution;
  }

  /**
   * @brief Sample. 
   */
  KOKKOS_INLINE_FUNCTION T operator()(auto&&...) const
  {
    return m_pool.uniform(m_distribution.start(), m_distribution.stop());
  }

private:

  UniformDistribution<T> m_distribution; ///< Distribution parameters
  RngPool<T, Kokkos::Random_XorShift64_Pool<TSpace>> m_pool; ///< RNG pool
};

template <typename T>
UniformRng(T (&&)[2]) -> UniformRng<T>;

template <typename T>
UniformRng(const Span<T>&) -> UniformRng<T>;

template <typename T>
UniformRng(T (&&)[2], Index) -> UniformRng<T>;

template <typename T>
UniformRng(const Span<T>&, Index) -> UniformRng<T>;

/**
 * @brief Gaussian probability distribution.
 * 
 * \code
 * auto noise = Linx::generate<100>("noise", Linx::GaussianRng(100, 15));
 * \endcode
 */
template <typename T>
class GaussianDistribution {
public:

  /**
   * @brief Constructor.
   */
  KOKKOS_INLINE_FUNCTION GaussianDistribution(T mu = 0, T sigma = 1) : m_mu(mu), m_sigma(sigma) {}

  /**
   * @brief Mean.
   */
  KOKKOS_INLINE_FUNCTION T mean() const
  {
    return m_mu;
  }

  /**
   * @brief Standard deviation. 
   */
  KOKKOS_INLINE_FUNCTION T stddev() const
  {
    return m_sigma;
  }

  /**
   * @brief Probability density function.
   */
  KOKKOS_INLINE_FUNCTION double pdf(const auto& x) const
  {
    const auto u = x - m_mu;
    const auto two_var = 2 * m_sigma * m_sigma;
    return std::numbers::inv_sqrtpi / (std::numbers::sqrt2 * m_sigma) * std::exp(-u * u / two_var);
  }

  /**
   * @brief Cumulative density function.
   */
  KOKKOS_INLINE_FUNCTION double cdf(const auto& x) const
  {
    return .5 * (1. + std::erf((x - m_mu) / (std::numbers::sqrt2 * m_sigma)));
  }

private:

  T m_mu; ///< Mean
  T m_sigma; ///< Standard deviation
};

/**
 * @brief Gaussian random number generator.
 */
template <typename T, typename TSpace = Kokkos::DefaultExecutionSpace>
class GaussianRng : public ArithmeticMixin<void, const T, GaussianRng<T, TSpace>> {
public:

  using value_type = const T;

  /**
   * @brief Constructor.
   */
  GaussianRng(GaussianDistribution<T> distribution, Index seed = -1) :
      m_distribution(LINX_MOVE(distribution)), m_pool(seed)
  {}

  std::string label() const
  {
    return "Gaussian"; // FIXME parameters
  }

  /**
   * @brief The distribution.
   */
  KOKKOS_INLINE_FUNCTION const auto& distribution() const
  {
    return m_distribution;
  }

  /**
   * @brief Sample using the Box-Muller method.
   */
  KOKKOS_INLINE_FUNCTION T operator()(auto&&...) const
  {
    double u;
    double theta;
    {
      auto gen = m_pool.generator();
      u = gen.uniform(-1., 0.); // for sampling over (0, 1] instead of [0, 1)
      theta = gen.uniform(0., 2 * std::numbers::pi);
    }

    const double r = Kokkos::sqrt(-2 * Kokkos::log(-u)); // -u in (0, 1]
    const double x = r * Kokkos::cos(theta);

    if constexpr (is_complex<T>()) { // Get two variables at once
      const double y = r * Kokkos::sin(theta);
      return T(x, y) * m_distribution.stddev() + m_distribution.mean();
    } else {
      return x * m_distribution.stddev() + m_distribution.mean();
    }
  }

private:

  GaussianDistribution<T> m_distribution; ///< Distribution parameters
  RngPool<T, Kokkos::Random_XorShift64_Pool<TSpace>> m_pool; ///< RNG pool
};

template <typename T>
GaussianRng(T (&&)[2]) -> GaussianRng<T>;

template <typename T>
GaussianRng(T (&&)[2], Index) -> GaussianRng<T>;

/**
 * @brief Poisson probability distribution.
 */
template <typename T>
class PoissonDistribution {
public:

  /**
   * @brief Constructor.
   */
  KOKKOS_INLINE_FUNCTION PoissonDistribution(T lambda) : m_lambda(lambda) {}

  /**
   * @brief Mean.
   */
  KOKKOS_INLINE_FUNCTION T mean() const
  {
    return m_lambda;
  }

  /**
   * @brief Probability mass function.
   */
  KOKKOS_INLINE_FUNCTION double pmf(const std::integral auto& k) const
  {
    return std::pow(m_lambda, k) * std::exp(-m_lambda) / boost::math::factorial(k);
  }

  /**
   * @brief Cumulative density function.
   */
  KOKKOS_INLINE_FUNCTION double cdf(const auto& x) const
  {
    auto floor = std::floor(x);
    return boost::math::gamma_q(floor + 1, m_lambda);
  }

private:

  T m_lambda; ///< Mean
};

/**
 * @brief Poisson random number generator.
 */
template <typename T, typename TSpace = Kokkos::DefaultExecutionSpace>
class PoissonRng : public ArithmeticMixin<void, const T, PoissonRng<T, TSpace>> {
public:

  using value_type = const T;

  /**
   * @brief Constructor.
   */
  PoissonRng(T lambda, Index seed = -1) : m_lambda(lambda), m_pool(seed) {}

  std::string label() const
  {
    return "Poisson"; // FIXME parameters
  }

  /**
   * @brief Sample.
   */
  KOKKOS_INLINE_FUNCTION T operator()(auto&&...) const
  {
    // For stability, generate u even when in <= 0
    auto u = m_pool.uniform(0., 1.);

    if (m_lambda <= 0 || u == 0) {
      return 0;
    }

    // FIXME support complex?
    auto p = std::exp(-m_lambda);
    auto cp = 0.0;
    T k {};
    while (cp < u) {
      cp += p;
      ++k;
      p *= m_lambda / k;
    }

    return k - 1;
  }

private:

  double m_lambda;
  RngPool<T, Kokkos::Random_XorShift64_Pool<TSpace>> m_pool; ///< RNG pool
};

/**
 * @brief Poisson noise generator.
 * 
 * As opposed to many implementation, this generator draws only once to generate one value,
 * and is therefore stable, i.e.:
 * 
 * \code
 * auto a = Linx::Sequence<int>({1, 10, 100, 1000}).apply(Linx::PoissonNoise(42));
 * auto b = Linx::Sequence<int>({1, 10, 0, 1000}).apply(Linx::PoissonNoise(42));
 * assert(a[0] == b[0]);
 * assert(a[1] == b[1]);
 * assert(a[3] == b[3]); // This fails in many implementations
 * \endcode
 */
template <typename TSpace = Kokkos::DefaultExecutionSpace> // FIXME PoissonRng<Forward, TSpace>?
class PoissonNoise {
public:

  /**
   * @brief Constructor.
   */
  PoissonNoise(Index seed = -1) : m_pool(seed) {}

  /**
   * @brief Sample from a given mean value.
   */
  KOKKOS_INLINE_FUNCTION Index operator()(double lambda) const
  {
    // For stability, generate u even when lambda <= 0
    auto u = m_pool.uniform(0., 1.);

    if (lambda <= 0 || u == 0) {
      return 0;
    }

    // FIXME support complex?
    auto p = std::exp(-lambda);
    auto cp = 0.0;
    Index k {};
    while (cp < u) {
      cp += p;
      ++k;
      p *= lambda / k;
    }

    return k - 1;
  }

private:

  RngPool<double, Kokkos::Random_XorShift64_Pool<TSpace>> m_pool; ///< RNG pool
};

template <typename T>
const T& as_readonly(const T& in) requires(std::is_const_v<typename T::value_type>) // FIXME to functional
{
  return in;
}

} // namespace Linx

#endif
