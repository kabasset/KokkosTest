cmake_minimum_required(VERSION 2.8)
project(KokkosTest)

find_package(Boost COMPONENTS unit_test_framework REQUIRED)
find_package(Kokkos REQUIRED)

enable_testing()

add_executable(RasterApply_test RasterApply_test.cpp)
target_link_libraries(RasterApply_test Kokkos::kokkos ${Boost_LIBRARIES})
add_test(RasterApply_test RasterApply_test)

add_executable(KokkosApply_test KokkosApply_test.cpp)
target_link_libraries(KokkosApply_test Kokkos::kokkos ${Boost_LIBRARIES})
add_test(KokkosApply_test KokkosApply_test)

add_executable(KokkosReduce_test KokkosReduce_test.cpp)
target_link_libraries(KokkosReduce_test Kokkos::kokkos ${Boost_LIBRARIES})
add_test(KokkosReduce_test KokkosReduce_test)
