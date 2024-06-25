cmake_minimum_required(VERSION 2.8)
project(KokkosTest)

find_package(Boost COMPONENTS unit_test_framework REQUIRED)
find_package(Kokkos REQUIRED)

add_library(Linx INTERFACE)
target_include_directories(Linx INTERFACE .)
target_link_libraries(Linx INTERFACE Kokkos::kokkos)

enable_testing()

add_executable(ArrayOfBools_test ArrayOfBools_test.cpp)
target_link_libraries(ArrayOfBools_test Kokkos::kokkos ${Boost_LIBRARIES})
add_test(ArrayOfBools_test ArrayOfBools_test)

add_executable(ArrayState_test ArrayState_test.cpp)
target_link_libraries(ArrayState_test Kokkos::kokkos ${Boost_LIBRARIES})
add_test(ArrayState_test ArrayState_test)

add_executable(ArrayApply_test ArrayApply_test.cpp)
target_link_libraries(ArrayApply_test Kokkos::kokkos ${Boost_LIBRARIES})
add_test(ArrayApply_test ArrayApply_test)

add_executable(KokkosApply_test KokkosApply_test.cpp)
target_link_libraries(KokkosApply_test Kokkos::kokkos ${Boost_LIBRARIES})
add_test(KokkosApply_test KokkosApply_test)

add_executable(KokkosReduce_test KokkosReduce_test.cpp)
target_link_libraries(KokkosReduce_test Kokkos::kokkos ${Boost_LIBRARIES})
add_test(KokkosReduce_test KokkosReduce_test)

add_executable(Vector_test Vector_test.cpp)
target_link_libraries(Vector_test Linx ${Boost_LIBRARIES})
add_test(Vector_test Vector_test)
