cmake_minimum_required(VERSION 3.18)
project(proxgqp LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(PROXGQP_VENDOR_DEPENDENCIES
  "Fetch and build Eigen and SuiteSparse instead of finding them" OFF)

find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

if(PROXGQP_VENDOR_DEPENDENCIES)
  include(FetchContent)
  set(FETCHCONTENT_QUIET OFF)

  FetchContent_Declare(eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG ${PROXGQP_EIGEN_TAG}
    GIT_SHALLOW TRUE)
  set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(eigen)

  execute_process(
    COMMAND ${Python_EXECUTABLE} -c
      "import scipy_openblas64 as b; print(b.get_lib_dir()); print(b.get_include_dir()); print(b.get_library())"
    OUTPUT_VARIABLE PROXGQP_OPENBLAS_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PROXGQP_OPENBLAS_STATUS)
  if(NOT PROXGQP_OPENBLAS_STATUS EQUAL 0)
    message(FATAL_ERROR
      "PROXGQP_VENDOR_DEPENDENCIES needs scipy-openblas64 importable at build time")
  endif()
  string(REPLACE "\n" ";" PROXGQP_OPENBLAS_FIELDS "${PROXGQP_OPENBLAS_OUTPUT}")
  list(GET PROXGQP_OPENBLAS_FIELDS 0 PROXGQP_OPENBLAS_LIBDIR)
  list(GET PROXGQP_OPENBLAS_FIELDS 1 PROXGQP_OPENBLAS_INCLUDE)
  list(GET PROXGQP_OPENBLAS_FIELDS 2 PROXGQP_OPENBLAS_NAME)
  find_library(PROXGQP_OPENBLAS_LIBRARY ${PROXGQP_OPENBLAS_NAME}
    PATHS ${PROXGQP_OPENBLAS_LIBDIR} NO_DEFAULT_PATH REQUIRED)
  set(BLAS_LIBRARY ${PROXGQP_OPENBLAS_LIBRARY})
  set(LAPACK_LIBRARY ${PROXGQP_OPENBLAS_LIBRARY})
  set(BLAS_LIBRARIES ${PROXGQP_OPENBLAS_LIBRARY} CACHE STRING "" FORCE)
  set(LAPACK_LIBRARIES ${PROXGQP_OPENBLAS_LIBRARY} CACHE STRING "" FORCE)

  set(SUITESPARSE_ENABLE_PROJECTS "suitesparse_config;amd;camd;colamd;ccolamd;cholmod"
      CACHE STRING "" FORCE)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(BUILD_STATIC_LIBS ON CACHE BOOL "" FORCE)
  set(SUITESPARSE_USE_OPENMP OFF CACHE BOOL "" FORCE)
  set(SUITESPARSE_USE_CUDA OFF CACHE BOOL "" FORCE)
  set(CHOLMOD_GPL ON CACHE BOOL "" FORCE)
  set(CHOLMOD_PARTITION OFF CACHE BOOL "" FORCE)
  set(SUITESPARSE_USE_STRICT OFF CACHE BOOL "" FORCE)
  set(SUITESPARSE_USE_FORTRAN OFF CACHE BOOL "" FORCE)
  set(SUITESPARSE_HAS_FORTRAN OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(suitesparse
    GIT_REPOSITORY https://github.com/DrTimothyAldenDavis/SuiteSparse.git
    GIT_TAG ${PROXGQP_SUITESPARSE_TAG}
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(suitesparse)

  set(CHOLMOD_LIBRARY CHOLMOD_static)
  set(SUITESPARSE_INCLUDE_DIR ${PROXGQP_OPENBLAS_INCLUDE})
else()
  find_package(Eigen3 REQUIRED NO_MODULE)
  find_path(SUITESPARSE_INCLUDE_DIR cholmod.h PATH_SUFFIXES suitesparse REQUIRED)
  find_library(CHOLMOD_LIBRARY cholmod REQUIRED)
  find_library(LAPACK_LIBRARY lapack REQUIRED)
  find_library(BLAS_LIBRARY blas REQUIRED)
endif()

file(GLOB PROXGQP_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/solve.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gblock.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/backend/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cone/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/road/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/smoothing/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/initialisation/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/globalisation/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/schedule/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/termination/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ruiz/*.cpp)

nanobind_add_module(_core NB_STATIC
  ${PROXGQP_SOURCES}
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/qdldl/qdldl.c
  ${CMAKE_CURRENT_SOURCE_DIR}/bindings/bindings.cpp)

target_include_directories(_core PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/qdldl
  ${SUITESPARSE_INCLUDE_DIR})

target_link_libraries(_core PRIVATE Eigen3::Eigen
  ${CHOLMOD_LIBRARY} ${LAPACK_LIBRARY} ${BLAS_LIBRARY})

set_target_properties(_core PROPERTIES
  BUILD_RPATH "${CMAKE_PREFIX_PATH}/lib"
  INSTALL_RPATH "${CMAKE_PREFIX_PATH}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/proxgqp")

install(TARGETS _core LIBRARY DESTINATION proxgqp)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/proxgqp
  DESTINATION . FILES_MATCHING PATTERN "*.py")

enable_testing()
add_library(proxgqp_core STATIC
  ${PROXGQP_SOURCES}
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/qdldl/qdldl.c)
target_include_directories(proxgqp_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/qdldl
  ${SUITESPARSE_INCLUDE_DIR})
target_link_libraries(proxgqp_core PUBLIC Eigen3::Eigen
  ${CHOLMOD_LIBRARY} ${LAPACK_LIBRARY} ${BLAS_LIBRARY})
set_target_properties(proxgqp_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  BUILD_RPATH "${CMAKE_PREFIX_PATH}/lib"
  INSTALL_RPATH "${CMAKE_PREFIX_PATH}/lib")

file(GLOB PROXGQP_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/test/test_*.cpp)
foreach(source ${PROXGQP_TESTS})
  get_filename_component(name ${source} NAME_WE)
  add_executable(${name} ${source})
  target_link_libraries(${name} PRIVATE proxgqp_core)
  add_test(NAME ${name} COMMAND ${name})
endforeach()
