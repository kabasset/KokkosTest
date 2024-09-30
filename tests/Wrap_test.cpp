// SPDX-FileCopyrightText: Copyright (C) 2024, Antoine Basset
// SPDX-License-Identifier: Apache-2.0

#define BOOST_TEST_MODULE WrapTest

#include "Linx/Base/Types.h"
#include "Linx/Run/ProgramContext.h"

#include <boost/test/unit_test.hpp>

template <typename T>
using Strong = Linx::Wrap<T, struct StrongTag>;

template <typename T>
using StrongRef = Linx::Wrap<const T&, struct StrongRefTag>;

LINX_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

BOOST_AUTO_TEST_CASE(data_wrapper_test)
{
  int data[] = {1, 2, 3};
  auto wrapper = Linx::Wrap(data);
  BOOST_TEST(wrapper.value == data);
  BOOST_TEST((std::is_same_v<decltype(wrapper)::value_type, int*>));
}

BOOST_AUTO_TEST_CASE(cdata_wrapper_test)
{
  const int cdata[] = {1, 2, 3};
  auto wrapper = Linx::Wrap(cdata);
  BOOST_TEST(wrapper.value == cdata);
  BOOST_TEST((std::is_same_v<decltype(wrapper)::value_type, const int*>));
}

BOOST_AUTO_TEST_CASE(strong_test)
{
  int value = 4;
  auto strong = Strong(value);
  BOOST_TEST(strong.value == value);
  BOOST_TEST((std::is_same_v<decltype(strong)::value_type, int>));
}

BOOST_AUTO_TEST_CASE(strongref_test)
{
  const int value = 4;
  auto strong = StrongRef(value);
  BOOST_TEST(strong.value == value);
  BOOST_TEST((std::is_same_v<decltype(strong)::value_type, const int&>));
}

BOOST_AUTO_TEST_SUITE_END()
