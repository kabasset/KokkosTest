// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXDATA_IMAGE_H
#define _LINXDATA_IMAGE_H

#include "Linx/Base/Containers.h"
#include "Linx/Base/Functional.h"
#include "Linx/Base/Types.h"
#include "Linx/Base/mixins/Data.h"
#include "Linx/Data/Box.h"
#include "Linx/Data/Sequence.h"
#include "Linx/Data/Slice.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <concepts>
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
 * assert(a(0, 0) == 1);
 * b.fill(2);
 * assert(a(0, 0) == 2);
 * \endcode
 * 
 * Deep copy is available as `copy_as()` or `operator+`:
 * 
 * \code
 * Image<float, 2> a("A");
 * auto b = a; // Shallow copy
 * auto c = a.copy_as("C"); // Deep copy labeled "C"
 * auto d = +a; // Deep copy labeled "copy(A)"
 * \endcode
 * 
 * By default, images may be allocated on device, e.g. GPU.
 * They can be copied to the host with `to_host()`, which is a no-op if the image is already on the host.
 */
template <typename T, int N, typename TContainer = typename DefaultContainer<T, N>::Image>
class Image : public DataMixin<T, EuclidArithmetic, Image<T, N, TContainer>> {
public:

  static constexpr int Rank = N; ///< The dimension parameter
  using Container = TContainer; ///< The underlying container type
  using Index = std::int64_t; ///< The default index type // FIXME get from Kokkos according to Properties
  using Shape = Position<Index, N>; ///< The shape type
  using Domain = Box<Index, N>; ///< The domain type
  using Super = DataMixin<T, EuclidArithmetic, Image<T, N, TContainer>>; ///< The parent class
  
  using memory_space = typename Container::memory_space;
  using execution_space = typename Container::execution_space;

  using value_type = typename Container::value_type; ///< The raw value type
  using element_type = std::decay_t<value_type>; ///< The decayed value type
  using size_type = typename Container::size_type; ///< The index and size type
  using difference_type = std::ptrdiff_t; ///< The index difference type
  using reference = typename Container::reference_type; ///< The element reference type
  using pointer = typename Container::pointer_type; ///< The element pointer type

  /**
   * @brief Constructor.
   * 
   * @param label The image label
   * @param shape The image shape along each axis
   * @param container A compatible container
   * @param args Arguments to be forwarded to the container constructor
   */
  explicit Image(const std::string& label, std::integral auto... shape) : m_container(label, shape...) {}

  /**
   * @copydoc Image()
   */
  template <std::integral TInt, typename UContainer>
  explicit Image(const std::string& label, const Sequence<TInt, Rank, UContainer>& shape) :
      Image(label, shape, std::make_index_sequence<Rank>()) // FIXME use ArrayLike?
  {} // FIXME support N = -1

  /**
   * @copydoc Image()
   */
  KOKKOS_INLINE_FUNCTION explicit Image(const Container& container) : m_container(container) {}

  /**
   * @copydoc Image()
   */
  KOKKOS_INLINE_FUNCTION explicit Image(Container&& container) : m_container(container) {}

  /**
   * @copydoc Image()
   */
  template <typename... TArgs>
  KOKKOS_INLINE_FUNCTION explicit Image(Forward, TArgs&&... args) : m_container(LINX_FORWARD(args)...)
  {}

  /**
   * @brief Image label. 
   */
  decltype(auto) label() const
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
  KOKKOS_INLINE_FUNCTION int extent(std::integral auto i) const
  {
    return m_container.extent_int(i);
  }

  /**
   * @brief Image shape along all axes. 
   */
  Shape shape() const
  {
    Shape out;
    for (int i = 0; i < N; ++i) {
      out[i] = m_container.extent_int(i);
    }
    return out;
  }

  /**
   * @brief Image domain. 
   */
  Domain domain() const
  {
    return domain(m_container);
  }

  /**
   * @brief Underlying container.
   */
  KOKKOS_INLINE_FUNCTION const Container& container() const
  {
    return m_container;
  }

  /**
   * @brief Underlying pointer to data.
   */
  KOKKOS_INLINE_FUNCTION pointer data() const
  {
    return m_container.data();
  }

  /**
   * @brief Reference to the element at given position.
   */
  template <std::integral TInt>
  KOKKOS_INLINE_FUNCTION reference operator[](const Sequence<TInt, N>& position) const
  {
    return at(position, std::make_index_sequence<N>());
  }

  /**
   * @brief Reference to the element at given indices.
   */
  KOKKOS_INLINE_FUNCTION reference operator()(std::integral auto... indices) const
  {
    return m_container(indices...);
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
  KOKKOS_INLINE_FUNCTION reference at(const TPosition& position, std::index_sequence<Is...>) const
  {
    return operator()(position[Is]...);
  }

  /**
   * @brief Helper function for 0-based containers.
   */
  template <typename... TArgs>
  static Domain domain(const Kokkos::View<TArgs...>& container)
  {
    Shape start("Image domain start");
    Shape stop("Image domain stop");
    for (int i = 0; i < Rank; ++i) {
      start[i] = 0;
      stop[i] = container.extent_int(i);
    }
    return {LINX_MOVE(start), LINX_MOVE(stop)};
  }

  /**
   * @brief Helper function for offset container.
   */
  template <typename... TArgs>
  static Domain domain(const Kokkos::Experimental::OffsetView<TArgs...>& container)
  {
    typename Domain::Container stop;
    for (int i = 0; i < Rank; ++i) {
      stop[i] = container.end(i);
    }
    return Domain {container.begins(), LINX_MOVE(stop)};
  }

  /**
   * @brief The underlying container.
   */
  Container m_container;
};

template <typename T>
struct IsImage : std::false_type {};

template <typename T, int N, typename... TArgs>
struct IsImage<Image<T, N, TArgs...>> : std::true_type {};

template <typename T>
concept AnyImage = IsImage<T>::value; // is_specialization won't work with non-type template parameters

/**
 * @brief Perform a shallow copy of an image, as a readonly image.
 * 
 * If the input image is aleady readonly, then this is a no-op.
 */
template <typename T, int N, typename TContainer>
KOKKOS_INLINE_FUNCTION decltype(auto) as_readonly(const Image<T, N, TContainer>& in)
{
  if constexpr (std::is_const_v<T>) {
    return in;
  } else {
    using Out = Image<const T, N, typename Rebind<TContainer>::AsReadonly>;
    return Out(Linx::Forward {}, in.container());
  }
}

/**
 * @brief Perform a shallow copy of an image, as an atomic image.
 */
template <typename T, int N, typename TContainer>
KOKKOS_INLINE_FUNCTION decltype(auto) as_atomic(const Image<T, N, TContainer>& in)
{
  using Out = Image<T, N, typename Rebind<TContainer>::AsAtomic>;
  return Out(Linx::Forward {}, in.container());
}

/**
 * @brief Copy the data to host if on device.
 */
template <typename T, int N, typename TContainer>
auto on_host(const Image<T, N, TContainer>& image)
{
  // FIXME early return if already on host
  auto container = Kokkos::create_mirror_view(image.container());
  Kokkos::deep_copy(container, image.container());
  using Container = typename std::decay_t<decltype(container)>;
  return Image<T, N, Container>(Forward(), LINX_MOVE(container));
}

} // namespace Linx

#endif
