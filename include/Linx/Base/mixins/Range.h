// @copyright 2022-2024, Antoine Basset (CNES)
// This file is part of Linx <github.com/kabasset/Linx>
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXBASE_MIXINS_RANGE_H
#define _LINXBASE_MIXINS_RANGE_H

// #include "Linx/Base/DataDistribution.h" // FIXME

#include <Kokkos_StdAlgorithms.hpp>
#include <algorithm>
#include <numeric> // accumulate

namespace Linx {

/**
 * @ingroup mixins
 * @brief Base class to provide range operations.
 * @tparam T The value type
 * @tparam TDerived The child class which implements required methods
 */
template <typename T, typename TDerived>
struct RangeMixin {
  /// @{
  /// @group_modifiers

  /**
   * @brief Fill the container with a single value.
   */
  const TDerived& fill(const T& value) const
  {
    return LINX_CRTP_CONST_DERIVED.generate(
        "fill",
        KOKKOS_LAMBDA() { return value; });
  }

  /**
   * @brief Fill the container with evenly spaced value.
   * 
   * The difference between two adjacent values is _exactly_ `step`,
   * i.e. `container[i + 1] = container[i] + step`.
   * This means that rounding errors may sum up,
   * as opposed to `linspace()`.
   * @see `linspace()`
   */
  TDerived& range(const T& min = Limits<T>::zero(), const T& step = Limits<T>::one())
  {
    auto v = min;
    auto& t = static_cast<TDerived&>(*this);
    for (auto& e : t) {
      e = v;
      v += step;
    }
    return t;
  }

  /**
   * @brief Fill the container with evenly spaced value.
   * 
   * The first and last values of the container are _exactly_ `min` and `max`.
   * Intermediate values are computed as `container[i] = min + (max - min) / (size() - 1) * i`,
   * which means that the difference between two adjacent values
   * is not necessarily perfectly constant for floating point values,
   * as opposed to `range()`.
   * @see `range()`
   */
  TDerived& linspace(const T& min = Limits<T>::zero(), const T& max = Limits<T>::one())
  {
    const std::size_t size = std::distance(static_cast<TDerived&>(*this).begin(), static_cast<TDerived&>(*this).end());
    const auto step = (max - min) / (size - 1);
    auto it = static_cast<TDerived&>(*this).begin();
    for (std::size_t i = 0; i < size - 1; ++i, ++it) {
      *it = min + step * i;
    }
    *it = max;
    return static_cast<TDerived&>(*this);
  }

  /**
   * @brief Reverse the order of the elements.
   */
  TDerived& reverse()
  {
    auto& t = static_cast<TDerived&>(*this);
    std::reverse(t.begin(), t.end());
    return t;
  }

  /// @group_operations

  /**
   * @brief Check whether the container contains a given value.
   */
  bool contains(const T& value) const
  {
    return std::find(static_cast<const TDerived&>(*this).begin(), static_cast<const TDerived&>(*this).end(), value) !=
        static_cast<const TDerived&>(*this).end();
  }

  /**
   * @brief Check whether the container contains NaNs.
   */
  bool contains_nan() const
  {
    return std::any_of(
        static_cast<const TDerived&>(*this).begin(),
        static_cast<const TDerived&>(*this).end(),
        [&](const T& e) {
          return e != e;
        });
  }

  /**
   * @brief Check whether all elements are equal to a given value.
   * 
   * If the container is empty, return `false`.
   */
  bool contains_only(const T& value) const
  {
    bool out;
    return LINX_CRTP_CONST_DERIVED.domain().reduce(
        "contains_only()",
        KOKKOS_LAMBDA(auto... is) { return LINX_CRTP_CONST_DERIVED(is...) == value; },
        Kokkos::LAnd<bool>(out));
  }

  /**
   * @brief Get a reference to the (first) min element.
   * @see `distribution()`
   */
  const T& min() const
  {
    return *std::min_element(static_cast<const TDerived&>(*this).begin(), static_cast<const TDerived&>(*this).end());
  }

  /**
   * @brief Get a reference to the (first) max element.
   * @see `distribution()`
   */
  const T& max() const
  {
    return *std::max_element(static_cast<const TDerived&>(*this).begin(), static_cast<const TDerived&>(*this).end());
  }

  /**
   * @brief Get a pair of references to the (first) min and max elements.
   * @see `distribution()`
   */
  std::pair<const T&, const T&> minmax() const
  {
    const auto its =
        std::minmax_element(static_cast<const TDerived&>(*this).begin(), static_cast<const TDerived&>(*this).end());
    return {*its.first, *its.second};
  }

  /// @}
};

/**
 * @relatesalo RangeMixin
 * @brief Get a reference to the (first) min element.
 * @see `distribution()`
 */
template <typename TRange>
const typename TRange::value_type& min(const TRange& in)
{
  return *std::min_element(in.begin(), in.end());
}

/**
 * @relatesalo RangeMixin
 * @brief Get a reference to the (first) max element.
 * @see `distribution()`
 */
template <typename TRange>
const typename TRange::value_type& max(const TRange& in)
{
  return *std::max_element(in.begin(), in.end());
}

/**
 * @relatesalo RangeMixin
 * @brief Get a pair of references to the (first) min and max elements.
 * @see `distribution()`
 */
template <typename TRange>
std::pair<const typename TRange::value_type&, const typename TRange::value_type&> minmax(const TRange& in)
{
  const auto its = std::minmax_element(in.begin(), in.end());
  return {*its.first, *its.second};
}

/**
 * @relatesalo RangeMixin
 * @brief Compute the sum of a range.
 * @param offset An offset
 */
template <typename TRange>
double sum(const TRange& in, double offset = 0)
{
  return std::accumulate(in.begin(), in.end(), offset);
}

/**
 * @relatesalo RangeMixin
 * @brief Compute the product of a range.
 * @param factor A factor
 */
template <typename TRange>
double product(const TRange& in, double factor = 1)
{
  return std::accumulate(in.begin(), in.end(), factor, std::multiplies<double> {});
}

/**
 * @relatesalo RangeMixin
 * @brief Compute the mean of a range.
 * @see `distribution()`
 */
template <typename TRange>
double mean(const TRange& in)
{
  return sum(in) / in.size();
}

/**
 * @relatesalo RangeMixin
 * @brief Create a `DataDistribution` from the container.
 */
// template <typename TRange>
// DataDistribution<typename TRange::value_type> distribution(const TRange& in)
// {
//   return DataDistribution<typename TRange::value_type>(in);
// }

} // namespace Linx

#endif
