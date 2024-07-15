// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE "ArrayState"

#include "Array.h"
#include "KokkosContext.h"

#include <boost/test/unit_test.hpp>

BOOST_TEST_GLOBAL_FIXTURE(KokkosContext);

BOOST_AUTO_TEST_SUITE(ArrayState);

BOOST_AUTO_TEST_CASE(domain_test)
{
  const int width = 4;
  const int height = 3;
  const int depth = 2;
  using Array = Linx::Array<int, 3>;
  Array a("a", {width, height, depth});
  Array b("b", width, height, depth);
  // BOOST_TEST(a.shape() == b.shape());
  // BOOST_TEST(a.domain() == b.domain());
  for (int i = 0; i < 3; ++i) {
    BOOST_TEST(a.shape()[i] == b.shape()[i]);
    BOOST_TEST(a.domain().front[i] == b.domain().front[i]);
    BOOST_TEST(a.domain().back[i] == b.domain().back[i]);
  }
}

BOOST_AUTO_TEST_SUITE_END();
