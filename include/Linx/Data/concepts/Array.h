// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXDATA_CONCEPTS_ARRAY_H
#define _LINXDATA_CONCEPTS_ARRAY_H

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

} // namespace Linx

#endif
