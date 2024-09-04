// @copyright 2022-2024, Antoine Basset (CNES)
// This file is part of Linx <github.com/kabasset/Linx>
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXBASE_MIXINS_RANGE_H
#define _LINXBASE_MIXINS_RANGE_H

#include "Linx/Base/Types.h"

#include <Kokkos_StdAlgorithms.hpp>
#include <algorithm>

namespace Linx {

/**
 * @ingroup mixins
 * @brief Base class to provide range operations.
 * @tparam T The value type
 * @tparam TDerived The child class which implements required methods
 */
template <typename T, typename TDerived>
struct RangeMixin {
  /**
   * @brief Copy values from a range.
   */
  template <std::input_iterator TIt>
  void assign(TIt begin, const TIt& end) const
  {
    const auto& container = LINX_CRTP_CONST_DERIVED.container();
    auto mirror = Kokkos::create_mirror_view(container);
    auto mirror_data = mirror.data();
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::HostSpace::execution_space>(0, container.size()),
        KOKKOS_LAMBDA(int i) {
          std::advance(begin, i);
          mirror_data[i] = *begin;
        });
    Kokkos::deep_copy(container, mirror);
  }

  /**
   * @brief Copy values from a host data pointer.
   */
  void assign(const T* data) const
  {
    const auto& container = LINX_CRTP_CONST_DERIVED.container();
    auto mirror = Kokkos::create_mirror_view(container);
    auto mirror_data = mirror.data();
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::HostSpace::execution_space>(0, container.size()),
        KOKKOS_LAMBDA(int i) { mirror_data[i] = data[i]; });
    Kokkos::deep_copy(container, mirror);
  }
  /**
   * @brief Fill the container with evenly spaced value.
   * @see `linspace()`
   */
  const TDerived& range(const T& min = Limits<T>::zero(), const T& step = Limits<T>::one()) const
  {
    range_impl(min, step);
    return LINX_CRTP_CONST_DERIVED;
  }

  /**
   * @brief Fill the container with evenly spaced value.
   * @see `range()`
   */
  const TDerived& linspace(const T& min = Limits<T>::zero(), const T& max = Limits<T>::one()) const
  {
    const auto step = (max - min) / (this->size() - 1);
    return range(min, step);
  }

  /**
   * @brief Reverse the order of the elements.
   */
  const TDerived& reverse() const // FIXME to DataMixin
  {
    std::reverse(LINX_CRTP_CONST_DERIVED.begin(), LINX_CRTP_CONST_DERIVED.end());
    return *this;
  }

  /// @cond
  /**
   * @brief Helper method which returns void.
   */
  void range_impl(const T& min, const T& step) const
  { // FIXME make private somehow?
    const auto size = LINX_CRTP_CONST_DERIVED.size();
    auto ptr = LINX_CRTP_CONST_DERIVED.data();
    Kokkos::parallel_for(
        "range()",
        size,
        KOKKOS_LAMBDA(int i) { ptr[i] = min + step * i; });
  }
  /// @endcond
};

} // namespace Linx

#endif
