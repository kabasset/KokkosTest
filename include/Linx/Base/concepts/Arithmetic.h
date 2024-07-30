// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-PackageSourceInfo: https://github.com/kabasset/KokkosTest
// SPDX-License-Identifier: Apache-2.0

#ifndef _LINXBASE_CONCEPTS_ARITHMETIC_H
#define _LINXBASE_CONCEPTS_ARITHMETIC_H

namespace Linx {

/**
 * @brief Concept for additivity, i.e. addable and subtractable types.
 */
template <typename T, typename U>
concept Additive = requires(T lhs, U rhs)
{
  ++lhs;
  --lhs;
  lhs++;
  lhs--;
  lhs += rhs;
  lhs -= rhs;
  lhs + rhs;
  lhs - rhs;
};

} // namespace Linx

#endif
