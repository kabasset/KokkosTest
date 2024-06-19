# KokkosTest

## Build

Build Kokkos:

```sh
cd <kokkos_clone_dir>
git clone https://github.com/kokkos/kokkos.git

mkdir <kokkos_build_dir>
cd <kokkos_build_dir>
cmake <kokkos_clone_dir>/kokkos -DCMAKE_CXX_COMPILER=g++ -DCMAKE_INSTALL_PREFIX=<kokkos_install_dir>
make install
```

```sh
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=<kokkos_install_dir>
make
make test
```
