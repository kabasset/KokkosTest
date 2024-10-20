// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXBASE_ARRAYPOOL_H
#define _LINXBASE_ARRAYPOOL_H

#include <Kokkos_Random.hpp> // Random_UniqueIndex::get_state_idx
#include <cstddef> // size_t
#include <type_traits> // remove_cvref

namespace Linx {

/**
 * @brief A pool of arrays for pre-allocating thread-wise memory.
 */
template <typename T, typename TSpace = Kokkos::DefaultExecutionSpace>
class ArrayPool {
  friend class Array;

private:

  using device_type = typename TSpace::device_type; ///< The device type

public:

  /**
   * @brief A non-owning array, which is returned by the pool.
   */
  class Array {
  public:

    using value_type = T; ///< The raw value type
    using element_type = std::remove_cvref_t<T>; ///< The decayed value type
    using pointer = value_type*; ///< The pointer type
    using reference = value_type&; ///< The reference type

    /**
     * @brief Constructor (acquires memory).
     */
    KOKKOS_INLINE_FUNCTION Array(const ArrayPool& pool) :
        m_pool(pool), m_index(m_pool.get_state()), m_data(&m_pool.m_memory(m_index, 0)),
        m_size(m_pool.m_memory.extent(1))
    {}

    /**
     * @brief Destructor (releases memory).
     */
    KOKKOS_INLINE_FUNCTION ~Array()
    {
      m_pool.free_state(m_index);
    }

    /**
     * @brief Array size.
     */
    KOKKOS_INLINE_FUNCTION std::size_t size() const
    {
      return m_size;
    }

    /**
     * @brief Pointer to the data. 
     */
    KOKKOS_INLINE_FUNCTION pointer data()
    {
      return m_data;
    }

    /**
     * @brief Access the element at given index.
     */
    KOKKOS_INLINE_FUNCTION reference operator[](Index i)
    {
      return m_data[i];
    }

  private:

    const ArrayPool& m_pool; ///< Parent pool
    Index m_index; ///< In-pool index
    T* m_data; ///< Data pointer
    std::size_t m_size; ///< Array size
  };

  /**
   * @brief Constructor.
   * @param size The size of each array
   */
  ArrayPool(std::size_t size) : m_locks("locks", TSpace().concurrency(), 1), m_memory("memory", m_locks.size(), size) {}

  /**
   * @brief Get one of the arrays. 
   */
  KOKKOS_INLINE_FUNCTION Array array() const
  {
    return Array(*this);
  }

private:

  /**
   * @brief Acquire an array and get its index.
   */
  KOKKOS_INLINE_FUNCTION Index get_state() const
  {
    return Kokkos::Impl::Random_UniqueIndex<device_type>::get_state_idx(m_locks);
  }

  /**
   * @brief Release an array.
   */
  KOKKOS_INLINE_FUNCTION void free_state(Index i) const
  {
    m_locks(i, 0) = 0;
  }

private:

  Kokkos::View<int**, device_type> m_locks; ///< The lock record
  Kokkos::View<T**, device_type> m_memory; ///< The actual memory
};

} // namespace Linx

#endif
