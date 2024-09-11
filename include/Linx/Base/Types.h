// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXBASE_TYPES_H
#define _LINXBASE_TYPES_H

#include <Kokkos_Complex.hpp>
#include <complex>
#include <concepts>
#include <limits>
#include <sstream>
#include <tuple>

namespace Linx {

/**
 * @brief List of supported integral types.
 */
#define LINX_SUPPORTED_INTS \
  bool, unsigned char, char, signed char, unsigned short, signed short, unsigned int, signed int, unsigned long, \
      signed long, unsigned long long, signed long long

/**
 * @brief List of supported floating point types.
 */
#define LINX_SUPPORTED_FLOATS float, double, long double

/**
 * @brief List of supported complex types.
 */
#define LINX_SUPPORTED_COMPLEXES \
  std::complex<float>, std::complex<double>, std::complex<long double>, Kokkos::complex<float>, \
      Kokkos::complex<double>, Kokkos::complex<long double>

/**
 * @brief List of supported types.
 */
#define LINX_SUPPORTED_TYPES LINX_SUPPORTED_INTS, LINX_SUPPORTED_FLOATS, LINX_SUPPORTED_COMPLEXES

/**
 * @brief List of supported types as a tuple.
 */
using SupportedTypes = std::tuple<LINX_SUPPORTED_TYPES>;

/**
 * @brief `BOOST_AUTO_TEST_CASE_TEMPLATE` for each supported type.
 */
#define LINX_TEST_CASE_TEMPLATE(name) BOOST_AUTO_TEST_CASE_TEMPLATE(name, T, SupportedTypes)

/**
 * @brief Define a default virtual destructor.
 */
#define LINX_VIRTUAL_DTOR(classname) \
  /** @brief Destructor. */ \
  virtual ~classname() = default;

/**
 * @brief Define default copy constructor and assignment operator.
 */
#define LINX_DEFAULT_COPYABLE(classname) \
  /** @brief Copy constructor. */ \
  KOKKOS_FUNCTION classname(const classname&) = default; \
  /** @brief Copy assignment operator. */ \
  KOKKOS_FUNCTION classname& operator=(const classname&) = default;

/**
 * @brief Define deleted copy constructor and assignment operator.
 */
#define LINX_NON_COPYABLE(classname) \
  /** @brief Deleted copy constructor. */ \
  KOKKOS_FUNCTION classname(const classname&) = delete; \
  /** @brief Deleted copy assignment operator. */ \
  KOKKOS_FUNCTION classname& operator=(const classname&) = delete;

/**
 * @brief Define default move constructor and assignment operator.
 */
#define LINX_DEFAULT_MOVABLE(classname) \
  /** @brief Move constructor. */ \
  KOKKOS_FUNCTION classname(classname&&) = default; \
  /** @brief Move assignment operator. */ \
  KOKKOS_FUNCTION classname& operator=(classname&&) = default;

/**
 * @brief Define deleted move constructor and assignment operator.
 */
#define LINX_NON_MOVABLE(classname) \
  /** @brief Deleted move constructor. */ \
  KOKKOS_FUNCTION classname(classname&&) = delete; \
  /** @brief Deleted move assignment operator. */ \
  KOKKOS_FUNCTION classname& operator=(classname&&) = delete;

/**
 * @brief Non-function `std::move`.
 */
#define LINX_MOVE(...) static_cast<std::remove_reference_t<decltype(__VA_ARGS__)>&&>(__VA_ARGS__)

/**
 * @brief Non-function `std::forward`.
 */
#define LINX_FORWARD(...) static_cast<decltype(__VA_ARGS__)&&>(__VA_ARGS__)

/**
 * @brief Static cast to the derived type.
 */
#define LINX_CRTP_DERIVED static_cast<TDerived&>(*this)

/**
 * @brief Static cast to the constant derived type.
 */
#define LINX_CRTP_CONST_DERIVED static_cast<const TDerived&>(*this)

/**
 * @brief Get the value type of a container.
 * 
 * If the container is constant, then the type is, too.
 */
template <typename TContainer>
using Value = std::
    conditional_t<std::is_const_v<TContainer>, const typename TContainer::value_type, typename TContainer::value_type>;

/// @cond
namespace Internal {

template <template <typename...> class TTemplate, typename TClass>
struct IsSpecialization : std::false_type {};

template <template <typename...> class TTemplate, typename... TArgs>
struct IsSpecialization<TTemplate, TTemplate<TArgs...>> : std::true_type {};

} // namespace Internal
/// @endcond

/**
 * @brief Test whether a class is a specialization of some class template.
 */
template <template <typename...> class TTemplate, typename TClass>
constexpr bool is_specialization = Internal::IsSpecialization<TTemplate, TClass>::value;

/**
 * @brief Type traits.
 */
template <typename T>
struct TypeTraits {
  /**
   * @brief The floating point type which corresponds to `T`.
   * 
   * A floating point type wide enough to handle any value of type `T`.
   * Can be complex.
   */
  using Floating = std::conditional_t<std::is_floating_point<T>::value, T, double>;

  /**
   * @brief The scalar type which corresponds to `T`.
   * 
   * The type itself in general, or the value type of complex types.
   */
  using Scalar = T;

  /**
   * @brief Make some `T` from a scalar.
   * 
   * Return the value itself if `T` is already scalar,
   * or a complex with same real and imaginary parts if `T` is complex.
   */
  KOKKOS_INLINE_FUNCTION static T from_scalar(Scalar in)
  {
    return in;
  }

  /**
   * @brief Make some `T` from a scalar function.
   * 
   * If `T` is complex, apply the function twice to get the real and imaginary parts.
   */
  template <typename TFunc, typename TArg>
  KOKKOS_INLINE_FUNCTION static T apply_scalar(TFunc&& func, TArg&& arg)
  {
    return LINX_FORWARD(func)(LINX_FORWARD(arg));
  }
};

/// @cond
template <typename T>
struct TypeTraits<std::complex<T>> {
  using Floating = std::complex<T>;

  using Scalar = T;

  KOKKOS_INLINE_FUNCTION static std::complex<T> from_scalar(T in)
  {
    return {in, in};
  }

  template <typename TFunc, typename TArg>
  KOKKOS_INLINE_FUNCTION static std::complex<T> apply_scalar(TFunc&& func, TArg&& arg)
  {
    return {LINX_FORWARD(func)(LINX_FORWARD(arg)), LINX_FORWARD(func)(LINX_FORWARD(arg))};
  }
};
/// @endcond

/**
 * @brief Compute the floor of an input floating point value, as an integer value.
 */
template <std::integral TInt, typename TFloat>
KOKKOS_INLINE_FUNCTION TInt floor(TFloat in)
{
  TInt out = in;
  return out - (in < 0);
}

/**
 * @brief Numeric limits and related key values of a value type.
 */
template <typename T>
struct Limits {
  /**
   * @brief The type for real numbers, the component type for complex numbers.
   */
  using Scalar = typename TypeTraits<T>::Scalar;

  /**
   * @brief 0 in general, or `false` for Booleans.
   */
  KOKKOS_INLINE_FUNCTION static T zero()
  {
    return TypeTraits<T>::from_scalar(0);
  }

  /**
   * @brief 1 in general, or `true` for Booleans, or 1 + i for complexes.
   */
  KOKKOS_INLINE_FUNCTION static T one()
  {
    return TypeTraits<T>::from_scalar(1);
  }

  /**
   * @brief The lowest possible value.
   */
  KOKKOS_INLINE_FUNCTION static T min()
  {
    return TypeTraits<T>::from_scalar(std::numeric_limits<Scalar>::lowest());
  }

  /**
   * @brief The highest possible value.
   */
  KOKKOS_INLINE_FUNCTION static T max()
  {
    return TypeTraits<T>::from_scalar(std::numeric_limits<Scalar>::max());
  }

  /**
   * @brief The infinity value if defined, or `max()` otherwise.
   */
  KOKKOS_INLINE_FUNCTION static T inf()
  {
    constexpr auto infinity = std::numeric_limits<Scalar>::infinity();
    return infinity ? TypeTraits<T>::from_scalar(infinity) : max();
  }

  /**
   * @brief The difference between two consecutive values.
   */
  KOKKOS_INLINE_FUNCTION static T epsilon()
  {
    return TypeTraits<T>::from_scalar(std::numeric_limits<Scalar>::epsilon());
  }

  /**
   * @brief The min plus one epsilon.
   */
  KOKKOS_INLINE_FUNCTION static T almost_min()
  {
    return min() + epsilon();
  }

  /**
   * @brief The max minus one epsilon.
   */
  KOKKOS_INLINE_FUNCTION static T almost_max()
  {
    return max() - epsilon();
  }

  /**
   * @brief The min over two.
   */
  KOKKOS_INLINE_FUNCTION static T half_min()
  {
    return min() / 2;
  }

  /**
   * @brief The max over two in general, rounded up for integers, or `true` for Booleans.
   */
  KOKKOS_INLINE_FUNCTION static T half_max()
  {
    return max() / 2 + std::is_integral<T>::value;
  }
};

/// @cond
template <typename T>
struct IsComplex : std::false_type {};

template <typename T>
struct IsComplex<std::complex<T>> : std::true_type {};

template <typename T>
struct IsComplex<Kokkos::complex<T>> : std::true_type {};
/// @endcond

/**
 * @brief Test whether a type is complex.
 */
template <typename T>
constexpr bool is_complex()
{
  return IsComplex<T>::value;
}

/// @cond
namespace Internal {

template <template <typename...> class C, typename... Ts>
std::true_type is_base_template_of_impl(const C<Ts...>*);

template <template <typename...> class C>
std::false_type is_base_template_of_impl(...);

} // namespace Internal
/// @endcond

/**
 * @brief Test whether a class derives from a base class template.
 */
template <template <typename...> class TBase, typename TDerived>
constexpr bool is_base_template_of()
{
  return decltype(Internal::is_base_template_of_impl<TBase>(std::declval<TDerived*>()))::value;
}

template <typename T>
concept Labeled = requires(const T obj) // FIXME to Base/concepts
{
  obj.label();
};

std::string label(const Labeled auto& in)
{
  return in.label();
}

std::string label(const auto& in)
{
  std::stringstream ss;
  ss << in;
  return ss.str();
}

/**
 * @brief Create a label from a function name and input containers.
 * @return `<func>(<in.label()>)`
 */
std::string compose_label(const std::string& func, const auto& in0, const auto&... ins)
{
  std::stringstream ss;
  ss << func << "(" << label(in0);
  ((ss << ", " << label(ins)), ...);
  ss << ")";
  return ss.str();
}

/**
 * @copybrief compose_label()
 * 
 * Nullary overload.
 */
inline std::string compose_label(const std::string& func)
{
  return func + "()";
}

/**
 * @brief Simplistic wrapper for strong naming and disambiguation.
 */
template <typename T>
struct Wrapper {
  KOKKOS_INLINE_FUNCTION explicit Wrapper(T v) : value {v} {}
  T value;
};

} // namespace Linx

#endif
