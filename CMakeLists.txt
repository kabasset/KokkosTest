# Setup

cmake_minimum_required(VERSION 3.16)
project(KokkosTest)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Dependencies

find_package(Boost COMPONENTS unit_test_framework program_options REQUIRED)
find_package(Kokkos REQUIRED)

# Library

add_library(Linx INTERFACE)
target_include_directories(Linx INTERFACE include)
target_link_libraries(Linx INTERFACE Kokkos::kokkos ${Boost_LIBRARIES})

# Executables

add_executable(KokkosBenchmarkConvolution src/KokkosBenchmarkConvolution.cpp)
target_link_libraries(KokkosBenchmarkConvolution Linx)

add_executable(KokkosBenchmarkExp src/KokkosBenchmarkExp.cpp)
target_link_libraries(KokkosBenchmarkExp Linx)

add_executable(KokkosBenchmarkIteration src/KokkosBenchmarkIteration.cpp)
target_link_libraries(KokkosBenchmarkIteration Linx)

# Tests

enable_testing()

add_executable(KokkosSmoke_test tests/KokkosSmoke_test.cpp)
target_link_libraries(KokkosSmoke_test Kokkos::kokkos ${Boost_LIBRARIES})
add_test(KokkosSmoke_test KokkosSmoke_test)

add_executable(BoxApply_test tests/BoxApply_test.cpp)
target_link_libraries(BoxApply_test Linx ${Boost_LIBRARIES})
add_test(BoxApply_test BoxApply_test)

add_executable(ImageApply_test tests/ImageApply_test.cpp)
target_link_libraries(ImageApply_test Linx ${Boost_LIBRARIES})
add_test(ImageApply_test ImageApply_test)

add_executable(ImageCorrelate_test tests/ImageCorrelate_test.cpp)
target_link_libraries(ImageCorrelate_test Linx ${Boost_LIBRARIES})
add_test(ImageCorrelate_test ImageCorrelate_test)

add_executable(ImagePixelwise_test tests/ImagePixelwise_test.cpp)
target_link_libraries(ImagePixelwise_test Linx ${Boost_LIBRARIES})
add_test(ImagePixelwise_test ImagePixelwise_test)

add_executable(Packs_test tests/Packs_test.cpp)
target_link_libraries(Packs_test Linx ${Boost_LIBRARIES})
add_test(Packs_test Packs_test)

add_executable(RasterOrdering_test tests/RasterOrdering_test.cpp)
target_link_libraries(RasterOrdering_test Linx ${Boost_LIBRARIES})
add_test(RasterOrdering_test RasterOrdering_test)

add_executable(Vector_test tests/Vector_test.cpp)
target_link_libraries(Vector_test Linx ${Boost_LIBRARIES})
add_test(Vector_test Vector_test)

# Deprecated

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
