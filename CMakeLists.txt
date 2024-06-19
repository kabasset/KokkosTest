cmake_minimum_required(VERSION 2.8)
project(KokkosTest)

find_package(Boost COMPONENTS unit_test_framework REQUIRED)
find_package(Kokkos REQUIRED)

enable_testing()
add_executable(Kokkos_test Kokkos_test.cpp)
target_link_libraries(Kokkos_test Kokkos::kokkos ${Boost_LIBRARIES})

add_test(Kokkos_test Kokkos_test)

