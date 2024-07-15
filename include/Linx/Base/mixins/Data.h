// @copyright 2022-2024, Antoine Basset (CNES)
// This file is part of Linx <github.com/kabasset/Linx>
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXBASE_MIXINS_DATA_H
#define _LINXBASE_MIXINS_DATA_H

#include "Linx/Base/mixins/Arithmetic.h"
#include "Linx/Base/mixins/Math.h"

#include <Kokkos_StdAlgorithms.hpp>

namespace Linx {

template <typename T, typename TArithmetic, typename TDerived>
struct DataMixin : ArithmeticMixin<TArithmetic, T, TDerived>, MathFunctionsMixin<T, TDerived> {
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
   * @brief Fill the container with distances between data address and element addresses.
   * 
   * Conceptually, this function performs:
   * 
   * \code
   * for (auto p : container.domain()) {
   *   container[p] = &container[p] - container.data();
   * }
   * \endcode
   */
  const TDerived& fill_with_offsets() const
  {
    const TDerived& derived = LINX_CRTP_CONST_DERIVED;
    const auto data = derived.data(); // FIXME is it derived(0...)?
    derived.domain().iterate(
        "fill_with_offsets()",
        KOKKOS_LAMBDA(auto... is) {
          auto ptr = &derived(is...);
          *ptr = ptr - data;
        });
    return derived;
  }

  /// @group_operations

  /**
   * @brief Test whether the container contains a given value.
   */
  bool contains(const T& value) const
  {
    bool out;
    return LINX_CRTP_CONST_DERIVED.domain().reduce(
        "contains()",
        KOKKOS_LAMBDA(auto... is) { return LINX_CRTP_CONST_DERIVED(is...) == value; },
        Kokkos::LOr<bool>(out));
  }

  /**
   * @brief Test whether the container contains NaNs.
   */
  bool contains_nan() const
  {
    bool out;
    return LINX_CRTP_CONST_DERIVED.domain().reduce(
        "contains_nan()",
        KOKKOS_LAMBDA(auto... is) {
          auto e = LINX_CRTP_CONST_DERIVED(is...);
          return e != e;
        },
        Kokkos::LOr<bool>(out));
  }

  /**
   * @brief Test whether all elements are equal to a given value.
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

  /// @}
};

} // namespace Linx

#endif