// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXDATA_VECTOR_H
#define _LINXDATA_VECTOR_H

#include "Linx/Base/TypeUtils.h"
#include "Linx/Data/Traits.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <string>

namespace Linx {

template <typename T, int N>
class Vector {
public:

  static constexpr int Rank = N;
  using Container = typename ContainerTraits<T, N>::Vector;

  using value_type = typename Container::value_type;
  using element_type = std::decay_t<value_type>;
  using size_type = typename Container::size_type;
  using difference_type = std::ptrdiff_t;
  using reference = typename Container::reference_type;
  // using const_reference = typename Container::const_reference_type;
  using pointer = typename Container::pointer_type;
  // using const_pointer = typename Container::const_pointer_type;
  using iterator = pointer;
  // using const_iterator = typename Container::const_iterator;

  explicit Vector(const std::string& name = "") : Vector(name, std::max(0, Rank)) {}

  template <typename TInt, typename std::enable_if_t<std::is_integral<TInt>::value>* = nullptr>
  explicit Vector(TInt size) : Vector("", size)
  {}

  template <typename TInt, typename std::enable_if_t<std::is_integral<TInt>::value>* = nullptr>
  explicit Vector(const std::string& name, TInt size) : m_container(name, size)
  {}

  Vector(std::initializer_list<T> list) : Vector("", list.begin(), list.end()) {}

  Vector(const std::string& name, std::initializer_list<T> list) : Vector(name, list.begin(), list.end()) {}

  template <typename TRange, typename std::enable_if_t<is_range<TRange>()>* = nullptr>
  explicit Vector(const std::string& name, TRange&& range) : Vector(name, range.begin(), range.end())
  {}

  template <typename TIt>
  explicit Vector(const std::string& name, TIt begin, TIt end) :
      Vector(name, end - begin) // FIXME enable proper iterators, not only poiters
  {
    auto mirror = Kokkos::create_mirror_view(m_container);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::HostSpace::execution_space>(0, size()),
        KOKKOS_LAMBDA(int i) { mirror(i) = begin[i]; });
    Kokkos::deep_copy(m_container, mirror);
  }

  Vector& operator=(const value_type& value)
  {
    namespace KE = Kokkos::Experimental;
    KE::fill(Kokkos::DefaultExecutionSpace(), KE::begin(m_container), KE::end(m_container), value);
    return *this;
  }

  KOKKOS_INLINE_FUNCTION size_type size() const
  {
    return m_container.size();
  }

  KOKKOS_INLINE_FUNCTION difference_type ssize() const
  {
    return static_cast<difference_type>(m_container.size());
  }

  KOKKOS_INLINE_FUNCTION bool empty() const
  {
    return m_container.size() == 0;
  }

  template <typename TInt>
  KOKKOS_INLINE_FUNCTION reference operator[](TInt i) const
  {
    return m_container(i);
  }

  KOKKOS_INLINE_FUNCTION pointer data() const
  {
    return m_container.data();
  }

  KOKKOS_INLINE_FUNCTION iterator begin() const
  {
    return m_container.data();
  }

  KOKKOS_INLINE_FUNCTION iterator end() const
  {
    return begin() + m_container.size();
  }

  template <typename TFunc, typename... TIns>
  void apply(const std::string& name, TFunc&& func, const TIns&... ins) const
  {
    generate(name, func, *this, ins...);
  }

  template <typename TFunc, typename... TIns>
  void generate(const std::string& name, TFunc&& func, const TIns&... ins) const
  {
    iterate(
        name,
        KOKKOS_LAMBDA(auto i) { m_container(i) = func(ins[i]...); });
  }

  template <typename TFunc>
  void iterate(const std::string& name, TFunc&& func) const
  {
    Kokkos::parallel_for(name, size(), LINX_FORWARD(func));
  }

private:

  Container m_container;
};

} // namespace Linx

#endif
