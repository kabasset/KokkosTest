// @copyright 2022-2024, Antoine Basset (CNES)
// This file is part of Linx <github.com/kabasset/Linx>
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXBASE_MIXINS_DATA_H
#define _LINXBASE_MIXINS_DATA_H

#include "Linx/Base/Functional.h"
#include "Linx/Base/Reduction.h"
#include "Linx/Base/mixins/Arithmetic.h"
#include "Linx/Base/mixins/Math.h"

#include <Kokkos_StdAlgorithms.hpp>
#include <string>
#include <utility> // integer_sequence, size_t

namespace Linx {

namespace Impl {

template <typename TContainer>
struct OffsetFiller {
  KOKKOS_INLINE_FUNCTION void operator()(auto... is) const
  {
    auto ptr = &m_container(is...);
    *ptr = ptr - m_data;
  }
  TContainer m_container;
  const typename TContainer::value_type* m_data;
};

template <typename TFunc, typename TOut, typename TIns, std::size_t... Is>
class Generator {
public:

  KOKKOS_INLINE_FUNCTION Generator(TFunc func, const TOut& out, const TIns& ins) : m_func(func), m_out(out), m_ins(ins)
  {}

  KOKKOS_INLINE_FUNCTION void operator()(auto... is) const
  {
    m_out(is...) = m_func(get<Is>(m_ins)(is...)...);
  }

private:

  TFunc m_func;
  TOut m_out;
  TIns m_ins;
};

} // namespace Impl

/**
 * @brief Data container mixin.
 * 
 * @tparam T The value type
 * @tparam TArithmetic The arithmetic tag
 * @tparam TDerived The derived class
 */
template <typename T, typename TArithmetic, typename TDerived>
struct DataMixin : public ArithmeticMixin<TArithmetic, T, TDerived>, public MathFunctionsMixin<T, TDerived> {
  /// @{
  /// @group_properties

  decltype(auto) label() const
  {
    return LINX_CRTP_CONST_DERIVED.container().label();
  }

  /**
   * @brief Container size. 
   */
  KOKKOS_INLINE_FUNCTION auto size() const
  {
    return LINX_CRTP_CONST_DERIVED.container().size();
  }

  /**
   * @brief Container size as a signed integer.
   */
  KOKKOS_INLINE_FUNCTION auto ssize() const
  {
    return static_cast<std::ptrdiff_t>(size());
  }

  /**
   * @brief Pointer to the raw data.
   */
  KOKKOS_INLINE_FUNCTION T* data() const
  {
    return LINX_CRTP_CONST_DERIVED.container().data();
  }

  KOKKOS_INLINE_FUNCTION const T* cdata() const
  {
    return const_cast<const T*>(data());
  }

  /**
   * @brief Test whether the container is empty.
   */
  KOKKOS_INLINE_FUNCTION bool empty() const
  {
    return size() == 0;
  }

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
    using Space = typename TDerived::execution_space;
    for_each<Space>(
        "fill_with_offsets()",
        derived.domain(),
        Impl::OffsetFiller<typename TDerived::Container>(derived.container(), derived.data()));
    return derived;
  }

  /**
   * @brief Copy the values from another container.
   */
  const TDerived& copy_from(const auto& container) const
  {
    return generate(compose_label("copy", container), Forward(), container);
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
    return generate_with_side_effects(label, LINX_FORWARD(func), as_readonly(inputs)...);
  }

  /**
   * @brief Assign each element according to a function.
   * 
   * @param label A label for debugging
   * @param func The function
   * @param others Optional containers the function acts on
   * 
   * The arguments of the function are the elements of the containers, if any, i.e.:
   * 
   * \code
   * container.generate_with_side_effects(label, func, a, b);
   * \endcode
   * 
   * conceptually performs:
   * 
   * \code
   * for (auto p : container.domain()) {
   *   container[p] = func(a[p], b[p]);
   * }
   * \endcode
   * 
   * The domain of the optional containers must include the container domain.
   * 
   * The function is allowed to have side effects, i.e., to modify its arguments.
   * In this case, the elements of the optional containers are effectively modified.
   * If the function has no side effect, it is preferrable to use `generate()` instead.
   * 
   * @see `DataMixin::apply()`
   * @see `DataMixin::generate()`
   */
  template <typename TFunc, typename... Ts>
  const TDerived& generate_with_side_effects(const std::string& label, TFunc&& func, const Ts&... others) const
  {
    generate_with_side_effects_impl(
        label,
        LINX_FORWARD(func),
        Tuple<Ts...>(others...),
        std::make_index_sequence<sizeof...(others)>());
    return LINX_CRTP_CONST_DERIVED;
  }

  template <typename TFunc, typename TIns, std::size_t... Is>
  void generate_with_side_effects_impl(
      const std::string& label,
      TFunc func,
      const TIns& others,
      std::index_sequence<Is...>) const // FIXME private
  {
    using Generator = Impl::Generator<TFunc, TDerived, TIns, Is...>;
    using Space = typename TDerived::execution_space;
    for_each<Space>(
        label,
        LINX_CRTP_CONST_DERIVED.domain(),
        Generator(LINX_FORWARD(func), LINX_CRTP_CONST_DERIVED, others));
  }

  /// @group_operations

  /**
   * @brief Test whether the container contains a given value.
   */
  bool contains(const T& value) const
  {
    const auto& derived = as_readonly(LINX_CRTP_CONST_DERIVED);
    return map_reduce("contains()", Equal(value), Or(), derived);
  }

  /**
   * @brief Test whether the container contains NaNs.
   */
  bool contains_nan() const
  {
    const auto& derived = as_readonly(LINX_CRTP_CONST_DERIVED);
    return map_reduce("contains_nan()", IsNan(), Or(), derived);
  }

  /**
   * @brief Test whether all elements are equal to a given value.
   * 
   * If the container is empty, return `false`.
   */
  bool contains_only(const T& value) const
  {
    const auto& derived = as_readonly(LINX_CRTP_CONST_DERIVED);
    return map_reduce("contains_only()", Equal(value), And(), derived);
  }

  /**
   * @brief Equality.
   */
  bool operator==(const auto& other) const
  {
    const auto& derived = as_readonly(LINX_CRTP_CONST_DERIVED);
    const auto& other_derived = as_readonly(other);
    return map_reduce("==", Equal(), And(), derived, other_derived);
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