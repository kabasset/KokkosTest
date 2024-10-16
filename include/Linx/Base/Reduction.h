// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXBASE_REDUCTION_H
#define _LINXBASE_REDUCTION_H

#include "Linx/Base/Containers.h"
#include "Linx/Base/Exceptions.h"
#include "Linx/Base/Functional.h"
#include "Linx/Base/Packs.h"
#include "Linx/Base/Types.h"
#include "Linx/Base/mixins/Data.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <string>
#include <utility> // integer_sequence, size_t

namespace Linx {

namespace Impl {

/**
 * @brief Functor which return a value from coordinates, typically using one or several images.
 */
template <typename T, typename TFunc, typename TIns, std::size_t... Is>
class Projection {
public:

  using value_type = std::remove_cv_t<T>;

  KOKKOS_INLINE_FUNCTION Projection(const TFunc& func, const TIns& ins) : m_func(func), m_ins(ins) {}

  KOKKOS_INLINE_FUNCTION value_type operator()(auto... is) const
  {
    return m_func(get<Is>(m_ins)(is...)...);
  }

private:

  TFunc m_func;
  TIns m_ins;
};

/**
 * @brief Kokkos reducer built from a binary operation functor.
 */
template <typename T, typename TFunc, typename TSpace>
class Reducer {
public:

  using reducer = Reducer; // Required for concept
  using value_type = std::remove_cv_t<T>;
  using result_view_type = Kokkos::View<value_type, TSpace>;

  KOKKOS_INLINE_FUNCTION Reducer(value_type& value, const TFunc& func, const T& identity) :
      m_view(&value), m_func(func), m_identity(identity)
  {}

  KOKKOS_INLINE_FUNCTION Reducer(const result_view_type& view, const TFunc& func, const T& identity) :
      m_view(view), m_func(func), m_identity(identity)
  {}

  KOKKOS_INLINE_FUNCTION void join(value_type& dst, const value_type& src) const
  {
    dst = m_func(dst, src);
  }

  KOKKOS_INLINE_FUNCTION void init(value_type& value) const
  {
    value = m_identity;
  }

  KOKKOS_INLINE_FUNCTION value_type& reference() const
  {
    return *m_view.data();
  }

  KOKKOS_INLINE_FUNCTION result_view_type view() const
  {
    return m_view;
  }

private:

  result_view_type m_view;
  TFunc m_func;
  value_type m_identity;
};

/**
 * @brief Functor which combines a projection and a reducer.
 */
template <typename T, typename TProj, typename TRed, std::size_t... Is>
class ProjectionReducer {
public:

  static constexpr std::size_t Rank = sizeof...(Is);
  using value_type = std::remove_cv_t<T>;

  /**
   * @brief Constructor.
   */
  KOKKOS_INLINE_FUNCTION ProjectionReducer(const TProj& projection, const TRed& reducer) :
      m_projection(projection), m_reducer(reducer)
  {}

  /**
   * @brief `reducer.join(tmp, projection(is...))`
   * @param args `is..., tmp`
   */
  template <typename... Ts>
  KOKKOS_INLINE_FUNCTION void operator()(Ts&&... args) const
  {
    auto tuple = forward_as_tuple(args...);
    static_assert(sizeof...(args) == Rank + 1);
    m_reducer.join(get<Rank>(tuple), m_projection(get<Is>(tuple)...));
  }

private:

  TProj m_projection;
  TRed m_reducer;
};

/**
 * @brief Helper function to iterate over the pack parameters.
 */
template <typename TSpace, typename TRegion, typename TProj, typename TRed, std::size_t... Is>
void kokkos_reduce_impl(
    const std::string& label,
    const TRegion& region,
    const TProj& projection,
    const TRed& reducer,
    std::index_sequence<Is...>)
{
  if constexpr (TRegion::Rank == 0) {
    return;
  } else {
    using T = typename TRed::value_type;
    using ProjectionReducer = Impl::ProjectionReducer<T, TProj, TRed, Is...>;
    Kokkos::parallel_reduce(
        label,
        kokkos_execution_policy<TSpace>(region),
        ProjectionReducer(projection, reducer),
        reducer);
  }
}

} // namespace Impl

/**
 * @brief Apply a reduction to a region.
 * 
 * @param label Some label for debugging
 * @param region The region
 * @param projection The projection function
 * @param reducer The reduction function
 * 
 * The projection function takes as input a list of indices and outputs some value.
 * Images are projections.
 * 
 * The reducer satisfies Kokkos' `ReducerConcept`.
 * The `join()` method of the reducer is used for both intra- and inter-thread reduction.
 */
template <
    typename TSpace = Kokkos::DefaultExecutionSpace,
    typename TRegion,
    typename TProj,
    typename TRed> // FIXME restrict to Regions
void kokkos_reduce(const std::string& label, const TRegion& region, const TProj& projection, const TRed& reducer)
{
  // FIXME call parallel_for only
#define LINX_CASE_RANK(n) \
  case n: \
    if constexpr (is_nadic<int, n, TProj>()) { \
      return Impl::kokkos_reduce_impl<TSpace>( \
          label, \
          pad<n>(region), \
          projection, \
          reducer, \
          std::make_index_sequence<n>()); \
    } else { \
      return; \
    }

  if constexpr (TRegion::Rank == -1) {
    switch (region.rank()) {
      case 0:
        return;
        LINX_CASE_RANK(1)
        LINX_CASE_RANK(2)
        LINX_CASE_RANK(3)
        LINX_CASE_RANK(4)
        LINX_CASE_RANK(5)
        LINX_CASE_RANK(6)
      default:
        throw Linx::OutOfBounds<'[', ']'>("Dynamic rank", region.rank(), {0, 6});
    }
  } else {
    Impl::kokkos_reduce_impl<TSpace>(label, region, projection, reducer, std::make_index_sequence<TRegion::Rank>());
  }

#undef LINX_CASE_RANK
}

/**
 * @brief Compute a reduction.
 * 
 * @param label A label for debugging
 * @param monoid The reduction monoid
 * @param in The input data container
 * 
 * The monoid is an associative binary operator functor, for which `identity_element()` is defined,
 * i.e. the following is available: `identity_element<T>(monoid)`, where `T` is the element type of `in`.
 */
template <typename TMonoid, typename TIn>
auto reduce(const std::string& label, const TMonoid& monoid, const TIn& in)
{
  using T = typename TIn::element_type;
  using Reducer = Impl::Reducer<T, TMonoid, Kokkos::HostSpace>;
  T value = identity_element<T>(monoid);
  kokkos_reduce<typename TIn::execution_space>(
      label,
      in.domain(),
      as_readonly(in),
      Reducer(value, monoid, identity_element<T>(monoid)));
  Kokkos::fence();
  return value;
}

/**
 * @brief Compute a reduction with mapping.
 * 
 * @param label A label for debugging
 * @param map The mapping functor
 * @param monoid The reduction monoid
 * @param ins Input data containers
 * 
 * For each position of the input domain, the elements of each input data container are passed to the mapping function
 * before the reduction monoid is applied, i.e., `map_reduce("", map, monoid, a, b, c)` produces:
 * 
 * \code
 * map(a[p0], b[p0], c[p0]) + map(a[p1], b[p1], c[p1]) + ... + map(a[pN], b[pN], c[pN])
 * \endcode
 * 
 * where `p0, p1, ... , pN` are the positions in the image domain and `+` denotes the monoid operator.
 * 
 * Typically, the dot product of two containers `a` and `b` can be implemented as:
 * 
 * \code
 * map_reduce("dot", Multiply(), Add(), a, b);
 * \endcode
 * 
 * @see `reduce()`
 */
template <typename TMap, typename TMonoid, typename... TIns>
auto map_reduce(const std::string& label, const TMap& map, const TMonoid& monoid, const TIns&... ins)
{
  return map_reduce_with_side_effects(label, map, monoid, as_readonly(ins)...);
}

namespace Impl {

/**
 * @brief Helper function to iterate over the pack.
 */
template <typename TMap, typename TMonoid, typename TIns, std::size_t... Is>
auto map_reduce_with_side_effects_impl(
    const std::string& label,
    const TMap& map,
    const TMonoid& monoid,
    const TIns& ins,
    std::index_sequence<Is...>)
{
  const auto& in0 = get<0>(ins);
  using Value = std::decay_t<decltype(in0)>::element_type;
  using T = decltype(identity_element<Value>(monoid));
  using Space = std::decay_t<decltype(in0)>::execution_space; // FIXME test accessibility of all Is
  using Projection = Impl::Projection<T, TMap, TIns, Is...>;
  using Reducer = Impl::Reducer<T, TMonoid, Kokkos::HostSpace>;
  T value = identity_element<T>(monoid);
  kokkos_reduce<Space>(label, in0.domain(), Projection(map, ins), Reducer(value, monoid, identity_element<T>(monoid)));
  Kokkos::fence();
  return value;
}

} // namespace Impl

/**
 * @copydoc map_reduce()
 */
template <typename TMap, typename TMonoid, typename... TIns>
auto map_reduce_with_side_effects(const std::string& label, const TMap& map, const TMonoid& monoid, const TIns&... ins)
{
  return Impl::map_reduce_with_side_effects_impl(
      label,
      map,
      monoid,
      Tuple<TIns...>(ins...),
      std::make_index_sequence<sizeof...(TIns)>());
}

template <typename TIn>
typename TIn::element_type min(const TIn& in)
{
  return reduce("min", Min(), in);
}

template <typename TIn>
typename TIn::element_type max(const TIn& in)
{
  return reduce("max", Max(), in);
}

/**
 * @brief Compute the sum of all elements of a data container.
 */
template <typename TIn>
typename TIn::element_type sum(const TIn& in) // FIXME limit to DataMixins
{
  return reduce("sum", Add(), in);
}

/**
 * @brief Compute the product of all elements of a data container.
 */
template <typename TIn>
typename TIn::element_type product(const TIn& in) // FIXME limit to DataMixins
{
  return reduce("product", Multiply(), in);
}

/**
 * @brief Compute the dot product of two data containers.
 */
template <typename TLhs, typename TRhs>
typename TLhs::element_type dot(const TLhs& lhs, const TRhs& rhs)
{
  return map_reduce("dot", Multiply(), Add(), lhs, rhs);
}

/**
 * @brief Compute the Lp-norm of a vector raised to the power p.
 * @tparam P The power
 */
template <int P, typename TIn>
typename TIn::element_type norm(const TIn& in)
{
  return map_reduce("norm", Abspow<P>(), Add(), in);
}

/**
 * @brief Compute the absolute Lp-distance between two vectors raised to the power p.
 * @tparam P The power
 */
template <int P, typename TLhs, typename TRhs>
typename TLhs::element_type distance(const TLhs& lhs, const TRhs& rhs)
{
  return map_reduce("distance", Abspow<P>(), Add(), lhs, rhs);
}

} // namespace Linx

#endif
