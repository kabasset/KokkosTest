// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXDATA_SEQUENCE_H
#define _LINXDATA_SEQUENCE_H

#include "Linx/Base/Containers.h"
#include "Linx/Base/Functional.h"
#include "Linx/Base/Types.h"
#include "Linx/Base/concepts/Array.h"
#include "Linx/Base/mixins/Data.h"
#include "Linx/Base/mixins/Range.h"
#include "Linx/Data/Slice.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <concepts>
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
class Sequence :
    public DataMixin<T, EuclidArithmetic, Sequence<T, N, TContainer>>,
    public RangeMixin<T, Sequence<T, N, TContainer>> {
public:

  // FIXME most aliases and methods to DataMixin

  static constexpr int Rank = N; ///< The size parameter
  using Container = TContainer; ///< The underlying container type
  using Index = std::int64_t;
  using Domain = Span<Index>;

  using value_type = typename Container::value_type; ///< The raw element value type
  using element_type = std::decay_t<value_type>; ///< The decayed element value type
  using size_type = typename Container::size_type; ///< The index and size type
  using difference_type = std::ptrdiff_t; ///< The index difference type
  using reference = typename Container::reference_type; ///< The element reference type
  using pointer = typename Container::pointer_type; ///< The element pointer type
  using iterator = decltype(Kokkos::Experimental::begin(Container())); ///< The iterator type
  using const_iterator = decltype(Kokkos::Experimental::cbegin(Container())); ///< The constant iterator type
  /**
   * @brief Constructor.
   * 
   * @param label The sequence label
   * @param size The sequence size
   * @param list The sequence values
   * @param begin Iterator to the values beginning
   * @param end Iterator to the values end
   * 
   * @warning If the size is set at compile time, the size parameter or value count must match it.
   */
  explicit Sequence(const std::string& label = "") : Sequence(label, std::max(0, Rank)) {}

  /**
   * @copydoc Sequence()
   */
  explicit Sequence(const std::string& label, std::integral auto size) : m_container(label, size) {}

  /**
   * @copydoc Sequence()
   */
  KOKKOS_INLINE_FUNCTION explicit Sequence(const Container& container) : m_container(container) {}

  /**
   * @copydoc Sequence()
   */
  KOKKOS_INLINE_FUNCTION explicit Sequence(Container&& container) : m_container(LINX_MOVE(container)) {}

  /**
   * @copydoc Sequence()
   */
  Sequence(std::initializer_list<T> values) : Sequence("", values.begin(), values.end()) {}

  /**
   * @copydoc Sequence()
   */
  explicit Sequence(const std::string& label, std::initializer_list<T> values) :
      Sequence(label, values.begin(), values.end())
  {}

  /**
   * @copydoc Sequence()
   */
  explicit Sequence(const std::string& label, std::ranges::range auto&& values) :
      Sequence(label, std::ranges::begin(values), std::ranges::end(values))
  {}

  /**
   * @copydoc Sequence()
   */
  template <typename... TArgs>
  KOKKOS_INLINE_FUNCTION explicit Sequence(Forward, TArgs&&... args) : m_container(LINX_FORWARD(args)...)
  {}

  /**
   * @copydoc Sequence()
   */
  Sequence(const std::string& label, std::input_iterator auto begin, std::input_iterator auto end) :
      Sequence(label, std::distance(begin, end))
  {
    this->assign(begin);
  }

  /**
   * @copydoc Sequence()
   */
  Sequence(const std::string& label, const T* begin, const T* end) : Sequence(label, end - begin)
  {
    this->assign(begin);
  }

  std::string label() const
  {
    return m_container.label();
  }

  KOKKOS_INLINE_FUNCTION Domain domain() const
  {
    return Slice<Index, SliceType::RightOpen>(0, m_container.size());
  }

  /**
   * @brief Container size, for compatibility with `DataContainer`.
   */
  KOKKOS_INLINE_FUNCTION size_type shape() const
  {
    return size();
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
   * @brief Access the i-th element, for compatibility with `DataContainer`.
   */
  KOKKOS_INLINE_FUNCTION reference operator()(std::integral auto i) const
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
    return Kokkos::Experimental::begin(m_container);
  }

  /**
   * @brief Iterator to the end.
   */
  KOKKOS_INLINE_FUNCTION iterator end() const
  {
    return Kokkos::Experimental::end(m_container);
  }

  /**
   * @brief Constant iterator to the beginning.
   */
  KOKKOS_INLINE_FUNCTION const_iterator cbegin() const
  {
    return Kokkos::Experimental::cbegin(m_container);
  }

  /**
   * @brief Constant iterator to the end.
   */
  KOKKOS_INLINE_FUNCTION const_iterator cend() const
  {
    return Kokkos::Experimental::cend(m_container);
  }

  /**
   * @brief Stream insertion.
   */
  friend std::ostream& operator<<(std::ostream& os, const Sequence& sequence)
  {
    auto hosted = on_host(sequence);
    os << "[" << hosted[0];
    for (std::size_t i = 1; i < hosted.size(); ++i) {
      os << ", " << hosted[i];
    }
    os << "]";
    return os;
  }

private:

  /**
   * @brief The Kokkos container.
   */
  Container m_container;
};

/**
 * @brief Perform a shallow copy of a sequence, as a readonly sequence.
 * 
 * If the input sequence is aleady readonly, then this is a no-op.
 */
template <typename T, int N, typename TContainer>
KOKKOS_INLINE_FUNCTION decltype(auto) as_readonly(const Sequence<T, N, TContainer>& in)
{
  if constexpr (std::is_const_v<T>) {
    return in;
  } else {
    using Out = Sequence<const T, N, typename Rebind<TContainer>::AsReadonly>;
    return Out(Linx::Forward {}, in.container());
  }
}

/**
 * @brief Perform a shallow copy of a sequence, as an atomic sequence.
 */
template <typename T, int N, typename TContainer>
KOKKOS_INLINE_FUNCTION decltype(auto) as_atomic(const Sequence<T, N, TContainer>& in)
{
  using Out = Sequence<T, N, typename Rebind<TContainer>::AsAtomic>;
  return Out(Linx::Forward {}, in.container());
}

/**
 * @brief Copy the data to host if on device.
 */
template <typename T, int N, typename TContainer>
auto on_host(const Sequence<T, N, TContainer>& seq)
{
  // FIXME early return if already on host
  auto container = Kokkos::create_mirror_view(seq.container());
  Kokkos::deep_copy(container, seq.container());
  using Container = typename std::decay_t<decltype(container)>;
  return Sequence<T, N, Container>(LINX_MOVE(container));
}

template <ArrayLike TIn, ArrayLike TOut>
void copy_to(const TIn& in, const TOut& out)
{
  auto domain = Slice(0, std::min<int>(std::size(in), std::size(out)));
  for_each(
      "copy_to()",
      domain,
      KOKKOS_LAMBDA(int i) { out[i] = in[i]; });
}

template <int M>
auto resize(const ArrayLike auto& in) // FIXME make_sequence? CTor?
{
  using T = std::decay_t<decltype(in[0])>;
  Sequence<T, M> out; // FIXME label
  copy_to(in, out);
  return out;
}

} // namespace Linx

#endif
