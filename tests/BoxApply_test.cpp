// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE BoxApplyTest

#include "Linx/Data/Image.h"
#include "Linx/Run/ProgramContext.h"

#include <boost/test/unit_test.hpp>

using Linx::ProgramContext;
BOOST_TEST_GLOBAL_FIXTURE(ProgramContext);
BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE);

BOOST_AUTO_TEST_CASE(reduce_test)
{
  std::vector<int> f {0, 0};
  std::vector<int> b {3, 4};
  Linx::Box<int, 2> box(f, b);

  auto sum = box.template reduce<int>("sum", [](int& out, int i, int j) {
    out += 1;
  });
  BOOST_TEST(sum == box.size());
}

// BOOST_AUTO_TEST_CASE(multi_reduce_test)
// {
//   struct BoxShape {
//     int width;
//     int height;
//     int size;
//   };

//   std::vector<int> f {0, 0};
//   std::vector<int> b {3, 4};
//   Linx::Box<int, 2> box(f, b);

//   auto shape = box.template reduce<BoxShape>(
//       "sum",
//       KOKKOS_LAMBDA(BoxShape & out, int i, int j) {
//         out.width = std::max(i + 1, out.width);
//         out.height = std::max(j + 1, out.height);
//         out.size = out.size + 1; // FIXME how to aggregate threads?
//       });
//   BOOST_TEST(shape.width == box.extent(0));
//   BOOST_TEST(shape.height == box.extent(1));
//   BOOST_TEST(shape.size == box.size());
// }

BOOST_AUTO_TEST_SUITE_END();
