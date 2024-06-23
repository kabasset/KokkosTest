// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE "ArrayOfBools"

#include "Array.h"
#include "KokkosContext.h"

#include <boost/test/unit_test.hpp>

BOOST_TEST_GLOBAL_FIXTURE(KokkosContext);

BOOST_AUTO_TEST_SUITE(ArrayOfBools);

BOOST_AUTO_TEST_CASE(contiguity_test)
{
  const int width = 5; // Odd
  const int height = 3; // Odd
  const int n = width * height;
  using Array = Linx::Array<bool, 2>;
  Array a("a", {width, height});
  bool ca[n];
  a.iterate(KOKKOS_LAMBDA(int i, int j) { a(i, j) = (i + j) % 2; });
  std::copy_n(a.view().data(), n, ca);
  const auto span = a.view().span();
  BOOST_TEST(span == sizeof(ca));
  for (int i = 0; i < n; ++i) {
    BOOST_TEST(ca[i] == i % 2);
  }
}

BOOST_AUTO_TEST_SUITE_END();
