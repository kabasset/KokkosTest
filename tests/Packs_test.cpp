// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE PacksTest

#include "Linx/Base/Packs.h"
#include "Linx/Run/ProgramContext.h"

#include <boost/test/unit_test.hpp>

LINX_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

BOOST_AUTO_TEST_CASE(apply_last_first_test)
{
  auto out = Linx::apply_last_first(
      [](int head, auto...) {
        return head;
      },
      1,
      2,
      3,
      4);
  BOOST_TEST(out == 4);

  auto tail_size = Linx::apply_last_first(
      [](int, auto... tail) {
        auto vec = std::vector<int> {tail...};
        return vec.size();
      },
      1,
      2,
      3,
      4);
  BOOST_TEST(tail_size == 3);
}

BOOST_AUTO_TEST_SUITE_END()
