// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXDATA_SEQUENCE_H
#define _LINXDATA_SEQUENCE_H

#include "Linx/Base/Containers.h"
#include "Linx/Base/Types.h"
#include "Linx/Base/mixins/Data.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <ranges>
#include <string>

namespace Linx {

/**
 * @brief Non-resizable 1D container with Euclid arithmetics and element-wise functions.
 * 
 * @tparam T The element value type
 * @tparam N The size, or -1 for runtime size
 */
template <typename T, int N, typename TContainer = typename DefaultContainer<T, N>::Sequence>
class Sequence : public DataMixin<T, EuclidArithmetic, Sequence<T, N>> {
public:

  // FIXME most aliases and methods to DataMixin

  static constexpr int Rank = N; ///< The size parameter
  using Container = TContainer; ///< The underlying container type

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
   * @param name The sequence name
   * @param size The sequence size
   * @param list The sequence values
   * @param begin Iterator to the values beginning
   * @param end Iterator to the values end
   * 
   * @warning If the size is set at compile time, the size parameter or value count must match it.
   */
  explicit Sequence(const std::string& name = "") : Sequence(name, std::max(0, Rank)) {}

  /**
   * @copydoc Sequence()
   */
  explicit Sequence(const std::string name, std::integral auto size) : m_container(name, size) {}

  /**
   * @copydoc Sequence()
   */
  Sequence(std::initializer_list<T> values) : Sequence("", values.begin(), values.end()) {}

  /**
   * @copydoc Sequence()
   */
  explicit Sequence(const std::string& name, std::initializer_list<T> values) :
      Sequence(name, values.begin(), values.end())
  {}

  /**
   * @copydoc Sequence()
   */
  explicit Sequence(const std::string& name, std::ranges::range auto&& values) :
      Sequence(name, std::ranges::begin(values), std::ranges::end(values))
  {}

  /**
   * @copydoc Sequence()
   */
  template <typename... TArgs>
  explicit Sequence(ForwardTag, TArgs&&... args) : m_container(LINX_FORWARD(args)...)
  {}

  /**
   * @copydoc Sequence()
   */
  explicit Sequence(const std::string& name, std::input_iterator auto begin, std::input_iterator auto end) :
      Sequence(name, std::distance(begin, end))
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
   * @copydoc Sequence()
   */
  explicit Sequence(const std::string& name, const T* begin, const T* end) : Sequence(name, end - begin)
  {
    auto mirror = Kokkos::create_mirror_view(m_container);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::HostSpace::execution_space>(0, size()),
        KOKKOS_LAMBDA(int i) { mirror(i) = begin[i]; });
    Kokkos::deep_copy(m_container, mirror);
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
   * @brief Underlying container.
   */
  KOKKOS_INLINE_FUNCTION const Container& container() const
  {
    return m_container;
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
   * @brief Assign each element according to a function.
   * 
   * @param name A label for debugging
   * @param func The function
   * @param others Optional sequences the function acts on
   * 
   * The arguments of the function are the elements of the sequences, if any, i.e.:
   * 
   * \code
   * sequence.generate_with_side_effects(label, func, a, b);
   * \endcode
   * 
   * conceptually performs:
   * 
   * \code
   * for (int i = 0; i < v.size(); ++i) {
   *   sequence[i] = func(a[i], b[i]);
   * }
   * \endcode
   * 
   * The size of the optional sequences must be at least as large as the sequence size.
   * 
   * The function is allowed to have side effects, i.e., to modify its arguments.
   * In this case, the elements of the optional sequences are effectively modified.
   * If the function has no side effect, it is preferrable to use `generate()` instead.
   * 
   * @see `DataMixin::apply()`
   * @see `DataMixin::generate()`
   */
  template <typename TFunc, typename... TIns>
  const Sequence& generate_with_side_effects(const std::string& label, TFunc&& func, const TIns&... others) const
  {
    Kokkos::parallel_for(
        label,
        size(),
        KOKKOS_LAMBDA(auto i) { m_container(i) = func(others[i]...); });
    return *this;
  }

private:

  /**
   * @brief The Kokkos container.
   */
  Container m_container;
};

/**
 * @brief Perform a shallow copy of an image, as a readonly image.
 * 
 * If the input image is aleady readonly, then this is a no-op.
 */
template <typename T, int N, typename TContainer>
KOKKOS_INLINE_FUNCTION decltype(auto) as_readonly(const Sequence<T, N, TContainer>& in)
{
  if constexpr (std::is_const_v<T>) {
    return in;
  } else {
    using Out = Sequence<const T, N, typename Rebind<TContainer>::AsReadonly>;
    return Out(Linx::ForwardTag {}, in.container());
  }
}

} // namespace Linx

#endif
