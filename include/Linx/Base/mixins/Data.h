// @copyright 2022-2024, Antoine Basset (CNES)
// This file is part of Linx <github.com/kabasset/Linx>
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXBASE_MIXINS_DATA_H
#define _LINXBASE_MIXINS_DATA_H

#include "Linx/Base/mixins/Arithmetic.h"
#include "Linx/Base/mixins/Math.h"

#include <Kokkos_StdAlgorithms.hpp>

namespace Linx {

/**
 * @brief Data container mixin.
 * 
 * @tparam T The value type
 * @tparam TArithmetic The arithmetic tag
 * @tparam TDerived The derived class
 */
template <typename T, typename TArithmetic, typename TDerived>
struct DataMixin :
    ArithmeticMixin<TArithmetic, T, TDerived>,
    MathFunctionsMixin<T, TDerived> { // FIXME deduce T = TDerived::value_type
  /// @{
  /// @group_modifiers

  /**
   * @brief Fill the container with a single value.
   */
  const TDerived& fill(const T& value) const
  {
    Kokkos::deep_copy(LINX_CRTP_CONST_DERIVED.container(), value);
    return LINX_CRTP_CONST_DERIVED;
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
    const auto& derived = LINX_CRTP_CONST_DERIVED;
    const auto data = derived.data();
    for_each(
        "fill_with_offsets()",
        derived.domain(),
        KOKKOS_LAMBDA(auto... is) {
          auto ptr = &derived(is...);
          *ptr = ptr - data;
        });
    return derived;
  }

  /**
   * @brief Assign the values from another container.
   */
  const TDerived& assign(const std::string& label, const auto& container) const
  {
    return generate(
        label,
        KOKKOS_LAMBDA(const auto& e) { return e; },
        container);
  }

  /**
   * @brief Apply a function to each element.
   * 
   * @param label A label for debugging
   * @param func The function
   * @param inputs Optional input containers
   * 
   * The first argument of the function is the element of the container itself.
   * If other images are passed as input, their elements are respectively passed to the function.
   * In this case, it is recommended to avoid side effects and to pass the inputs as readonly.
   * 
   * In other words:
   * 
   * \code
   * container.apply(label, func, a, b);
   * \endcode
   * 
   * conceptually performs:
   * 
   * \code
   * for (auto p : container.domain()) {
   *   container[p] = func(Linx::as_readonly(container)[p], Linx::as_readonly(a)[p], Linx::as_readonly(b)[p]);
   * }
   * \endcode
   * 
   * and is equivalent to:
   * 
   * \code
   * container.generate(label, func, container, a, b);
   * \endcode
   * 
   * @see `generate()`
   */
  const TDerived& apply(const std::string& label, auto&& func, const auto&... inputs) const
  {
    const auto& derived = as_readonly(LINX_CRTP_CONST_DERIVED);
    return LINX_CRTP_CONST_DERIVED
        .generate_with_side_effects(label, LINX_FORWARD(func), derived, as_readonly(inputs)...);
  }

  /**
   * @brief Assign each element according to a function.
   * 
   * @param label A label for debugging
   * @param func The function
   * @param inputs Optional input images
   * 
   * The arguments of the function are the elements of the input images, if any, i.e.:
   * 
   * \code
   * image.generate(label, func, a, b);
   * \endcode
   * 
   * conceptually performs:
   * 
   * \code
   * for (auto p : image.domain()) {
   *   image[p] = func(a[p], b[p]);
   * }
   * \endcode
   * 
   * @see `apply()`
   */
  const TDerived& generate(const std::string& label, auto&& func, const auto&... inputs) const
  {
    return LINX_CRTP_CONST_DERIVED.generate_with_side_effects(label, LINX_FORWARD(func), as_readonly(inputs)...);
  }

  /// @group_operations

  /**
   * @brief Test whether the container contains a given value.
   */
  bool contains(const T& value) const
  {
    bool out;
    const auto& derived = as_readonly(LINX_CRTP_CONST_DERIVED);
    kokkos_reduce(
        "contains()",
        derived.domain(),
        KOKKOS_LAMBDA(auto... is) { return derived(is...) == value; },
        Kokkos::LOr<bool>(out));
    return out;
  }

  /**
   * @brief Test whether the container contains NaNs.
   */
  bool contains_nan() const
  {
    bool out;
    const auto& derived = as_readonly(LINX_CRTP_CONST_DERIVED);
    kokkos_reduce(
        "contains_nan()",
        derived.domain(),
        KOKKOS_LAMBDA(auto... is) {
          auto e = derived(is...);
          return e != e;
        },
        Kokkos::LOr<bool>(out));
    return out;
  }

  /**
   * @brief Test whether all elements are equal to a given value.
   * 
   * If the container is empty, return `false`.
   */
  bool contains_only(const T& value) const
  {
    bool out;
    const auto& derived = as_readonly(LINX_CRTP_CONST_DERIVED);
    kokkos_reduce(
        "contains_only()",
        derived.domain(),
        KOKKOS_LAMBDA(auto... is) { return derived(is...) == value; },
        Kokkos::LAnd<bool>(out));
    return out;
  }

  /**
   * @brief Equality.
   */
  bool operator==(const auto& other) const
  {
    bool out;
    const auto& derived = as_readonly(LINX_CRTP_CONST_DERIVED);
    const auto& other_derived = as_readonly(other);
    kokkos_reduce(
        "==",
        derived.domain(),
        KOKKOS_LAMBDA(auto... is) { return derived(is...) == other_derived(is...); },
        Kokkos::LAnd<bool>(out));
    return out;
  }

  /**
   * @brief Inequality.
   */
  bool operator!=(const auto& other) const
  {
    return not(*this == other);
  }

  /// @}
};

} // namespace Linx

#endif