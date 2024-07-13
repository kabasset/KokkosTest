// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXDATA_VECTOR_H
#define _LINXDATA_VECTOR_H

#include "Linx/Base/Containers.h"
#include "Linx/Base/Types.h"
#include "Linx/Base/mixins/Arithmetic.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <ranges>
#include <string>

namespace Linx {

/**
 * @brief Non-resizable 1D container with vector arithmetics and element-wise functions.
 * 
 * @tparam T The element value type
 * @tparam N The size, or -1 for runtime size
 */
template <typename T, int N>
class Vector : public ArithmeticMixin<VectorArithmetic, T, Vector<T, N>> {
public:

  // FIXME most aliases and methods to DataMixin

  static constexpr int Rank = N; ///< The size parameter
  using Container = typename DefaultContainer<T, N>::Vector; ///< The underlying container type

  using value_type = typename Container::value_type; ///< The raw element value type
  using element_type = std::decay_t<value_type>; ///< The decayed element value type
  using size_type = typename Container::size_type; ///< The index and size type
  using difference_type = std::ptrdiff_t; ///< The index difference type
  using reference = typename Container::reference_type; ///< The element reference type
  using pointer = typename Container::pointer_type; ///< The element pointer type
  using iterator = pointer; ///< The iterator type // FIXME from KE::begin()

  /**
   * @brief Constructor.
   * 
   * @param name The vector name
   * @param size The vector size
   * @param list The vector values
   * @param begin Iterator to the values beginning
   * @param end Iterator to the values end
   * 
   * @warning If the size is set at compile time, the size parameter or value count must match it.
   */
  explicit Vector(const std::string& name = "") : Vector(name, std::max(0, Rank)) {}

  /**
   * @copydoc Vector()
   */
  explicit Vector(const std::string name, std::integral auto size) : m_container(name, size) {}

  /**
   * @copydoc Vector()
   */
  Vector(std::initializer_list<T> values) : Vector("", values.begin(), values.end()) {}

  /**
   * @copydoc Vector()
   */
  explicit Vector(const std::string& name, std::initializer_list<T> values) : Vector(name, values.begin(), values.end())
  {}

  /**
   * @copydoc Vector()
   */
  explicit Vector(const std::string& name, std::ranges::range auto&& values) :
      Vector(name, std::ranges::begin(values), std::ranges::end(values))
  {}

  /**
   * @copydoc Vector()
   */
  explicit Vector(const std::string& name, std::input_iterator auto begin, std::input_iterator auto end) :
      Vector(name, std::distance(begin, end))
  {
    auto mirror = Kokkos::create_mirror_view(m_container);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::HostSpace::execution_space>(0, size()),
        KOKKOS_LAMBDA(int i) {
          std::advance(begin, i);
          mirror(i) = begin;
        });
    Kokkos::deep_copy(m_container, mirror);
  }

  /**
   * @copydoc Vector()
   */
  explicit Vector(const std::string& name, const T* begin, const T* end) : Vector(name, end - begin)
  {
    auto mirror = Kokkos::create_mirror_view(m_container);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::HostSpace::execution_space>(0, size()),
        KOKKOS_LAMBDA(int i) { mirror(i) = begin[i]; });
    Kokkos::deep_copy(m_container, mirror);
  }

  Vector& operator=(const value_type& value) // FIXME rm, replace with mixin's fill()
  {
    namespace KE = Kokkos::Experimental;
    KE::fill(Kokkos::DefaultExecutionSpace(), KE::begin(m_container), KE::end(m_container), value);
    return *this;
  }

  /**
   * @brief Container size. 
   */
  KOKKOS_INLINE_FUNCTION size_type size() const // FIXME to mixin
  {
    return m_container.size();
  }

  /**
   * @brief Container size as a signed integer.
   */
  KOKKOS_INLINE_FUNCTION difference_type ssize() const // FIXME to mixin
  {
    return static_cast<difference_type>(m_container.size());
  }

  /**
   * @brief Test whether the container is empty.
   */
  KOKKOS_INLINE_FUNCTION bool empty() const // FIXME to mixin
  {
    return m_container.size() == 0;
  }

  /**
   * @brief Access the i-th element.
   */
  KOKKOS_INLINE_FUNCTION reference operator[](std::integral auto i) const
  {
    return m_container(i);
  }

  /**
   * @brief Pointer to the raw data.
   */
  KOKKOS_INLINE_FUNCTION pointer data() const // FIXME to mixin
  {
    return m_container.data();
  }

  /**
   * @brief Iterator to the beginning.
   */
  KOKKOS_INLINE_FUNCTION iterator begin() const
  {
    return m_container.data(); // FIXME KE::begin(m_container)
  }

  /**
   * @brief Iterator to the end.
   */
  KOKKOS_INLINE_FUNCTION iterator end() const
  {
    return begin() + m_container.size(); // FIXME KE::end(m_container)
  }

  /**
   * @brief Apply a function to each element.
   * 
   * @param name A label for debugging
   * @param func The function
   * @param ins Optional input vectors
   * 
   * The first argument of the function is the element of the vector itself.
   * If other vectors are passed as input, their elements are respectively passed to the function.
   * 
   * In other words:
   * 
   * \code
   * v.apply(name, func, a, b);
   * \endcode
   * 
   * is equivalent to:
   * 
   * \code
   * for (int i = 0; i < v.size(); ++i) {
   *   v[i] = func(v[i], a[i], b[i]);
   * }
   * \endcode
   * 
   * and to:
   * 
   * \code
   * v.generate(name, func, v, a, b);
   * \endcode
   * 
   * @see `generate()`
   */
  template <typename TFunc, typename... TIns>
  void apply(const std::string& name, TFunc&& func, const TIns&... ins) const
  {
    generate(name, LINX_FORWARD(func), *this, ins...);
  }

  /**
   * @brief Assign each element according to a function.
   * 
   * @param name A label for debugging
   * @param func The function
   * @param ins Optional input vectors
   * 
   * The arguments of the function are the elements of the input vectors, if any, i.e.:
   * 
   * \code
   * v.generate(name, func, a, b);
   * \endcode
   * 
   * is equivalent to:
   * 
   * \code
   * for (int i = 0; i < v.size(); ++i) {
   *   v[i] = func(a[i], b[i]);
   * }
   * \endcode
   * 
   * @see `apply()`
   */
  template <typename TFunc, typename... TIns>
  void generate(const std::string& name, TFunc&& func, const TIns&... ins) const
  {
    iterate(
        name,
        KOKKOS_LAMBDA(auto i) { m_container(i) = func(ins[i]...); });
  }

  template <typename TFunc>
  void iterate(const std::string& name, TFunc&& func) const // FIXME rm
  {
    Kokkos::parallel_for(name, size(), LINX_FORWARD(func));
  }

private:

  /**
   * @brief The Kokkos container.
   */
  Container m_container;
};

} // namespace Linx

#endif
