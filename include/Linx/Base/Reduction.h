// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXBASE_REDUCTION_H
#define _LINXBASE_REDUCTION_H

#include "Linx/Base/Containers.h"
#include "Linx/Base/Functional.h"
#include "Linx/Base/Packs.h"
#include "Linx/Base/Types.h"
#include "Linx/Base/mixins/Data.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <string>
#include <utility> // integer_sequence, size_t

namespace Linx {

/// @cond
namespace Internal {

template <typename T, typename TFunc, typename TIns, std::size_t... Is>
class Projection {
public:

  using value_type = std::remove_cv_t<T>;

  KOKKOS_INLINE_FUNCTION Projection(TFunc func, const TIns& ins) : m_func(func), m_ins(ins) {}

  KOKKOS_INLINE_FUNCTION value_type operator()(auto... is) const
  {
    return m_func(std::get<Is>(m_ins)(is...)...);
  }

private:

  TFunc m_func;
  const TIns& m_ins;
};

template <typename T, typename TFunc, typename TSpace>
class Reducer {
public:

  using reducer = Reducer; // Required for concept
  using value_type = std::remove_cv_t<T>;
  using result_view_type = Kokkos::View<value_type, TSpace>;

  KOKKOS_INLINE_FUNCTION Reducer(value_type& value, TFunc func, T neutral) :
      m_view(&value), m_func(func), m_neutral(neutral)
  {}

  KOKKOS_INLINE_FUNCTION Reducer(const result_view_type& view, TFunc func, T neutral) :
      m_view(view), m_func(func), m_neutral(neutral)
  {}

  KOKKOS_INLINE_FUNCTION void join(value_type& dst, const value_type& src) const
  {
    dst = m_func(dst, src);
  }

  KOKKOS_INLINE_FUNCTION void init(value_type& value) const
  {
    value = m_neutral;
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
  value_type m_neutral;
};

template <typename T, typename TProj, typename TFunc, std::size_t... Is>
class ProjectionReducer {
public:

  static constexpr std::size_t Rank = sizeof...(Is);
  using value_type = std::remove_cv_t<T>;

  /**
   * @brief Constructor.
   */
  KOKKOS_INLINE_FUNCTION ProjectionReducer(TProj projection, const TFunc& reducer) :
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
  TFunc m_reducer;
};

template <typename TRegion, typename TProj, typename TFunc, std::size_t... Is>
void kokkos_reduce_impl(
    const std::string& label,
    const TRegion& region,
    TProj projection,
    TFunc reducer,
    std::index_sequence<Is...>)
{
  using T = typename TFunc::value_type;
  using ProjectionReducer = Internal::ProjectionReducer<T, TProj, TFunc, Is...>;
  Kokkos::parallel_reduce(label, kokkos_execution_policy(region), ProjectionReducer(projection, reducer), reducer);
}

} // namespace Internal
/// @endcond

/**
 * @brief Apply a reduction to a region.
 * 
 * @param label Some label for debugging
 * @param region The region
 * @param projection The projection function
 * @param reducer The reduction function
 * 
 * The projection function takes as input a list of indices and outputs some value.
 * 
 * The reducer satisfies Kokkos' `ReducerConcept`.
 * The `join()` method of the reducer is used for both intra- and inter-thread reduction.
 */
template <typename TRegion, typename TProj, typename TFunc> // FIXME restrict to Regions
void kokkos_reduce(const std::string& label, const TRegion& region, TProj projection, TFunc reducer)
{
  Internal::kokkos_reduce_impl(label, region, projection, reducer, std::make_index_sequence<TRegion::Rank>());
}

/**
 * @brief Compute a reduction.
 * 
 * @param label A label for debugging
 * @param func The reduction function
 * @param neutral The reduction neutral element
 * @param in The input data container
 */
template <typename TFunc, typename T, typename TIn>
T reduce(const std::string& label, TFunc&& func, T neutral, const TIn& in)
{
  using Reducer = Internal::Reducer<T, TFunc, typename TIn::Container::memory_space>;
  T value;
  kokkos_reduce(label, in.domain(), as_readonly(in), Reducer(value, LINX_FORWARD(func), LINX_FORWARD(neutral)));
  return value;
}

/**
 * @brief Compute a reduction with mapping.
 * 
 * @param label A label for debugging
 * @param func The reduction function
 * @param neutral The reduction neutral element
 * @param projection The mapping function
 * @param ins Input data containers
 */
template <typename TFunc, typename T, typename TProj, typename... TIns>
T map_reduce(const std::string& label, TFunc&& func, T neutral, TProj&& projection, const TIns&... ins)
{
  return map_reduce_with_side_effects(
      label,
      LINX_FORWARD(func),
      neutral,
      LINX_FORWARD(projection),
      as_readonly(ins)...);
}

/// @cond
namespace Internal {

template <typename TFunc, typename T, typename TProj, typename TIns, std::size_t... Is>
T map_reduce_with_side_effects_impl(
    const std::string& label,
    TFunc&& func,
    T neutral,
    TProj&& projection,
    const TIns& ins,
    std::index_sequence<Is...>)
{
  using Projection = Internal::Projection<T, TProj, TIns, Is...>;
  using First = std::decay_t<std::tuple_element_t<0, TIns>>;
  using Reducer = Internal::Reducer<T, TFunc, typename First::Container::memory_space>;
  T value;
  kokkos_reduce(
      label,
      std::get<0>(ins).domain(),
      Projection(LINX_FORWARD(projection), ins),
      Reducer(value, LINX_FORWARD(func), LINX_FORWARD(neutral)));
  return value;
}

} // namespace Internal
/// @endcond

/**
 * @copydoc map_reduce()
 */
template <typename TFunc, typename T, typename TProj, typename... TIns>
T map_reduce_with_side_effects(
    const std::string& label,
    TFunc&& func,
    T neutral,
    TProj&& projection,
    const TIns&... ins)
{
  return Internal::map_reduce_with_side_effects_impl(
      label,
      LINX_FORWARD(func),
      neutral,
      LINX_FORWARD(projection),
      std::forward_as_tuple(ins...),
      std::make_index_sequence<sizeof...(TIns)>());
}

template <typename TIn>
typename TIn::element_type min(const TIn& in)
{
  using T = typename TIn::element_type;
  T out;
  kokkos_reduce(compose_label("min", in), in.domain(), as_readonly(in), Kokkos::Min<T>(out));
  Kokkos::fence();
  return out;
}

template <typename TIn>
typename TIn::element_type max(const TIn& in)
{
  using T = typename TIn::element_type;
  T out;
  kokkos_reduce(compose_label("max", in), in.domain(), as_readonly(in), Kokkos::Max<T>(out));
  Kokkos::fence();
  return out;
}

/**
 * @brief Compute the sum of all elements of a data container.
 */
template <typename TIn>
typename TIn::element_type sum(const TIn& in) // FIXME limit to DataMixins
{
  using T = typename TIn::element_type; // FIXME to DataMixin
  return reduce("sum", Plus(), T {}, in);
}

/**
 * @brief Compute the dot product of two data containers.
 */
template <typename TLhs, typename TRhs>
typename TLhs::element_type dot(const TLhs& lhs, const TRhs& rhs)
{
  using T = typename TLhs::element_type; // FIXME to DataMixin
  return map_reduce("dot", Plus(), T {}, Multiplies(), lhs, rhs);
}

/**
 * @brief Compute the Lp-norm of a vector raised to the power p.
 * @tparam P The power
 */
template <int P, typename TIn>
typename TIn::element_type norm(const TIn& in)
{
  using T = typename TIn::element_type;
  return map_reduce("norm", Plus(), T {}, Abspow<P, T>(), in);
}

/**
 * @brief Compute the absolute Lp-distance between two vectors raised to the power p.
 * @tparam P The power
 */
template <int P, typename TLhs, typename TRhs>
typename TLhs::element_type distance(const TLhs& lhs, const TRhs& rhs)
{
  using T = typename TLhs::element_type; // FIXME type of r - l
  return map_reduce("distance", Plus(), T {}, Abspow<P, T>(), lhs, rhs);
}

} // namespace Linx

#endif
