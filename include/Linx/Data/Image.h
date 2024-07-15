// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXDATA_IMAGE_H
#define _LINXDATA_IMAGE_H

#include "Linx/Base/Containers.h"
#include "Linx/Base/Types.h"
#include "Linx/Base/mixins/Data.h"
#include "Linx/Data/Box.h"
#include "Linx/Data/Vector.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <string>

namespace Linx {

/**
 * @brief ND image.
 * 
 * @tparam T Element type
 * @tparam N Rank, i.e. number of axes
 * @tparam TContainer Underlying element container
 * 
 * Copy constructor and copy assignment operator perform shallow copy:
 * 
 * \code
 * auto a = Image<int, 2>("a", 4, 3).fill(1);
 * auto b = a;
 * assert a(0, 0) == 1;
 * b.fill(2);
 * assert a(0, 0) == 2;
 * \endcode
 * 
 * Deep copy is available as `copy()` or `operator+`:
 * 
 * \code
 * Image<float, 2> a("A");
 * auto b = a; // Shallow copy
 * auto c = a.copy("C"); // Deep copy labeled "C"
 * auto d = +a; // Deep copy labeled "copy of A"
 * \endcode
 * 
 * By default, images may be allocated on device, e.g. GPU.
 * They can be copied to the host with `to_host()`, which is a no-op if the image is already on the host.
 */
template <typename T, int N, typename TContainer = typename DefaultContainer<T, N>::Image>
class Image : public DataMixin<T, EuclidArithmetic, Image<T, N, TContainer>> {
public:

  static constexpr int Rank = N;
  using Container = TContainer;

  using value_type = typename Container::value_type;
  using element_type = std::decay_t<value_type>;
  using size_type = typename Container::size_type;
  using difference_type = std::ptrdiff_t;
  using reference = typename Container::reference_type;
  using pointer = typename Container::pointer_type;

  /**
   * @brief Constructor.
   * 
   * @param label The image label
   * @param shape The image shape along each axis
   * @param container The image container
   */
  Image(const std::string& label, std::integral auto... shape) : m_container(label, shape...) {}

  /**
   * @copydoc Image()
   */
  Image(const std::string& label, const Vector<std::integral auto, Rank>& shape) :
      Image(label, shape, std::make_index_sequence<Rank>())
  {} // FIXME support N = -1

  /**
   * @copydoc Image()
   */
  Image(const Container& container) : m_container(container) {}

  /**
   * @copydoc Image()
   */
  Image(Container&& container) : m_container(container) {}

  /**
   * @brief Image label. 
   */
  KOKKOS_INLINE_FUNCTION auto label() const
  {
    return m_container.label();
  }

  /**
   * @brief Number of elements. 
   */
  KOKKOS_INLINE_FUNCTION size_type size() const
  {
    return m_container.size();
  }

  /**
   * @brief Image extent along a given axis.
   */
  KOKKOS_INLINE_FUNCTION int extent(int i) const
  {
    return m_container.extent_int(i);
  }

  /**
   * @brief Image shape along all axes. 
   */
  KOKKOS_INLINE_FUNCTION Vector<int, N> shape() const
  {
    Vector<int, N> out;
    for (int i = 0; i < N; ++i) {
      out[i] = m_container.extent_int(i);
    }
    return out;
  }

  /**
   * @brief Image domain. 
   */
  KOKKOS_INLINE_FUNCTION Box<int, N> domain() const
  {
    Vector<int, N> f;
    Vector<int, N> e;
    for (int i = 0; i < N; ++i) {
      f[i] = 0;
      e[i] = m_container.extent_int(i);
    }
    return {LINX_MOVE(f), LINX_MOVE(e)};
  }

  /**
   * @brief Underlying pixel container.
   */
  const auto& container() const
  {
    return m_container;
  }

  /**
   * @brief Underlying pointer to data.
   */
  KOKKOS_INLINE_FUNCTION auto data() const
  {
    return m_container.data();
  }

  /**
   * @brief Reference to the element at given position.
   */
  KOKKOS_INLINE_FUNCTION decltype(auto) operator[](const Vector<std::integral auto, N>& position) const
  {
    return at(position, std::make_index_sequence<N>());
  }

  /**
   * @brief Reference to the element at given indices.
   */
  KOKKOS_INLINE_FUNCTION decltype(auto) operator()(std::integral auto... indices) const
  {
    return m_container(indices...);
  }

  /**
   * @brief Apply a function to each element.
   * 
   * @param label A label for debugging
   * @param func The function
   * @param ins Optional input images
   * 
   * The first argument of the function is the element of the image itself.
   * If other images are passed as input, their elements are respectively passed to the function.
   * 
   * In other words:
   * 
   * \code
   * image.apply(label, func, a, b);
   * \endcode
   * 
   * performs:
   * 
   * \code
   * for (auto p : image.domain()) {
   *   image[p] = func(image[p], a[p], b[p]);
   * }
   * \endcode
   * 
   * and is equivalent to:
   * 
   * \code
   * image.generate(label, func, image, a, b);
   * \endcode
   * 
   * @see `generate()`
   */
  template <typename TFunc, typename... Ts>
  const Image& apply(const std::string& label, TFunc&& func, const Ts&... ins) const
  {
    return generate(label, LINX_FORWARD(func), m_container, ins...);
  }

  /**
   * @brief Assign each element according to a function.
   * 
   * @param label A label for debugging
   * @param func The function
   * @param ins Optional input images
   * 
   * The arguments of the function are the elements of the input images, if any, i.e.:
   * 
   * \code
   * image.generate(label, func, a, b);
   * \endcode
   * 
   * performs:
   * 
   * \code
   * for (auto p : image.domain()) {
   *   image[p] = func(a[p], b[p]);
   * }
   * \endcode
   * 
   * @see `apply()`
   */
  template <typename TFunc, typename... Ts>
  const Image& generate(const std::string& label, TFunc&& func, const Ts&... ins) const
  {
    domain().iterate(
        label,
        KOKKOS_LAMBDA(auto... is) { m_container(is...) = func(ins(is...)...); });
    return *this;
  }

  /**
   * @brief Compute a reduction.
   * 
   * @param label A label for debugging
   * @param reducer A Kokkos reducer
   * @param projection A projection function
   * @param ins Optional input images
   */
  template <typename TRed>
  auto reduce(const std::string& label, TRed&& reducer) const
  {
    return domain().reduce(
        label,
        KOKKOS_LAMBDA(auto... is) { return m_container(is...); },
        LINX_FORWARD(reducer));
  }

  /**
   * @copydoc reduce()
   */
  template <typename TRed, typename TProj, typename... Ts>
  auto reduce(const std::string& label, TRed&& reducer, TProj&& projection, const Ts&... ins) const
  {
    return domain().reduce(
        label,
        KOKKOS_LAMBDA(auto... is) { return projection(m_container(is...), ins(is...)...); },
        LINX_FORWARD(reducer));
  }

private:

  /**
   * @brief Helper constructor to unroll shape.
   */
  template <typename TShape, std::size_t... Is>
  Image(const std::string& label, const TShape& shape, std::index_sequence<Is...>) : Image(label, shape[Is]...)
  {}

  /**
   * @brief Helper accessor to unroll position.
   */
  template <typename TPosition, std::size_t... Is>
  inline T& at(const TPosition& position, std::index_sequence<Is...>) const
  {
    return operator()(position[Is]...);
  }

  /**
   * @brief The underlying container.
   */
  Container m_container;
};

/**
 * @brief Contiguous image with row-major ordering.
 */
template <typename T, int N>
class Raster : public Image<T, N, typename DefaultContainer<T, N, Kokkos::LayoutLeft>::Image> {
public:

  using Super = Image<T, N, typename DefaultContainer<T, N, Kokkos::LayoutLeft>::Image>;
  using iterator = Super::pointer;

  // Inherit constructors
  using Super::Image;

  // No need for a virtual destructor, because there are no member variables

  /**
   * @brief Iterator to the beginning.
   */
  iterator begin() const
  {
    return this->data();
  }

  /**
   * @brief Iterator to the end.
   */
  iterator end() const
  {
    return begin() + this->size();
  }

  /**
   * @brief Fill the container with evenly spaced value.
   * 
   * The difference between two adjacent values is _exactly_ `step`,
   * i.e. `container[i + 1] = container[i] + step`.
   * This means that rounding errors may sum up,
   * as opposed to `linspace()`.
   * 
   * @see `linspace()`
   */
  const Raster& range(const T& min = Limits<T>::zero(), const T& step = Limits<T>::one()) const
  {
    auto v = min;
    for (auto& e : *this) {
      e = v;
      v += step;
    }
    return *this;
  }

  /**
   * @brief Fill the container with evenly spaced value.
   * 
   * The first and last values of the container are _exactly_ `min` and `max`.
   * Intermediate values are computed as `container[i] = min + (max - min) / (size() - 1) * i`,
   * which means that the difference between two adjacent values
   * is not necessarily perfectly constant for floating point values,
   * as opposed to `range()`.
   * 
   * @see `range()`
   */
  const Raster& linspace(const T& min = Limits<T>::zero(), const T& max = Limits<T>::one()) const
  {
    const std::size_t size = this->size();
    const auto step = (max - min) / (size - 1);
    auto it = begin();
    for (std::size_t i = 0; i < size - 1; ++i, ++it) {
      *it = min + step * i;
    }
    *it = max;
    return *this;
  }
};

} // namespace Linx

#endif
