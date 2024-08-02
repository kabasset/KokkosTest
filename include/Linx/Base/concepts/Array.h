// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXBASE_CONCEPTS_ARRAY_H
#define _LINXBASE_CONCEPTS_ARRAY_H

namespace Linx {

/**
 * @brief Concept for array-like classes.
 * 
 * Array-like classes have a size and integral subscript operator.
 */
template <typename T>
concept ArrayLike = requires(const T array)
{
  std::size(array);
  array[0];
};

template <typename T>
concept DataContainer = requires(const T data)
{
  typename T::size_type;
  typename T::value_type;
  typename T::pointer;
  typename T::Container;
  std::size(data);
  data.shape();
  data.domain();
  data.label(); // FIXME convertible to str
  data.data(); // FIXME pointer
  data.container(); // FIXME const T::Container&, compatible with deep_copy
  data(int(0)); // FIXME according to Rank?
  data.generate_with_side_effects(std::string(), []() {
    return typename T::value_type {};
  });
};

} // namespace Linx

#endif
