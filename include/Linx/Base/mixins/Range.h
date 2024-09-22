// @copyright 2022-2024, Antoine Basset (CNES)
// This file is part of Linx <github.com/kabasset/Linx>
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXBASE_MIXINS_RANGE_H
#define _LINXBASE_MIXINS_RANGE_H

#include "Linx/Base/Packs.h"
#include "Linx/Base/Types.h"

#include <Kokkos_StdAlgorithms.hpp>
#include <algorithm>

namespace Linx {

namespace Impl {

template <typename TLayout>
struct IsContiguousLayout : std::false_type {};

template <>
struct IsContiguousLayout<Kokkos::LayoutLeft> : std::true_type {};

template <>
struct IsContiguousLayout<Kokkos::LayoutRight> : std::true_type {};

} // namespace Impl

template <typename TContainer>
constexpr bool is_contiguous()
{
  return Impl::IsContiguousLayout<typename TContainer::array_layout>::value;
}

template <bool IsContiguous, typename T, typename TDerived>
struct RangeMixin {};

/**
 * @ingroup mixins
 * @brief Base class to provide range operations.
 * @tparam T The value type
 * @tparam TDerived The child class which implements required methods
 */
template <typename T, typename TDerived>
struct RangeMixin<true, T, TDerived> {
  /**
   * @brief Copie values.
   */
  KOKKOS_INLINE_FUNCTION const TDerived& assign(std::convertible_to<T> auto... values) const
  {
    assign_impl(forward_as_tuple(values...), std::make_index_sequence<sizeof...(values)>());
    return LINX_CRTP_CONST_DERIVED;
  }

  /**
   * @brief Copy values from a list.
   */
  const TDerived& assign(std::initializer_list<T> values) const
  {
    return assign(values.begin());
  }

  /**
   * @brief Copy values from a range.
   */
  template <std::input_iterator TIt>
  const TDerived& assign(TIt begin) const
  {
    const auto& container = LINX_CRTP_CONST_DERIVED.container();
    auto mirror = Kokkos::create_mirror_view(container);
    auto mirror_data = mirror.data();
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::HostSpace::execution_space>(0, container.size()),
        KOKKOS_LAMBDA(int i) { mirror_data[i] = *std::ranges::next(begin, i); });
    Kokkos::deep_copy(container, mirror);
    return LINX_CRTP_CONST_DERIVED;
  }

  /**
   * @brief Copy values from a host data pointer.
   */
  const TDerived& assign(const T* data) const
  {
    const auto& container = LINX_CRTP_CONST_DERIVED.container();
    auto mirror = Kokkos::create_mirror_view(container);
    auto mirror_data = mirror.data();
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::HostSpace::execution_space>(0, container.size()),
        KOKKOS_LAMBDA(int i) { mirror_data[i] = data[i]; });
    Kokkos::deep_copy(container, mirror);
    return LINX_CRTP_CONST_DERIVED;
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
   * @brief Reference to the i-th element.
   */
  KOKKOS_INLINE_FUNCTION auto& operator[](std::integral auto i) const
  {
    return *std::ranges::next(std::ranges::begin(LINX_CRTP_CONST_DERIVED), i);
  }

  /**
   * @brief Reverse the order of the elements.
   */
  const TDerived& reverse() const // FIXME to DataMixin
  {
    const auto& derived = LINX_CRTP_CONST_DERIVED;
    std::reverse(std::ranges::begin(derived), std::ranges::end(derived));
    return derived;
  }

  /// @cond

  /**
   * @brief Helper function for unfolding pack.
   */
  template <std::size_t... Is>
  KOKKOS_INLINE_FUNCTION void assign_impl(const auto& values, std::index_sequence<Is...>) const
  {
    const auto& container = LINX_CRTP_CONST_DERIVED.container();
    ((container(Is) = get<Is>(values)), ...);
  }

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
