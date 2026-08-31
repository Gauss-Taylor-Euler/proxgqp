cmake_minimum_required(VERSION 3.18)
project(proxgqp LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_path(SUITESPARSE_INCLUDE_DIR cholmod.h PATH_SUFFIXES suitesparse REQUIRED)
find_library(CHOLMOD_LIBRARY cholmod REQUIRED)
find_library(LAPACK_LIBRARY lapack REQUIRED)
find_library(BLAS_LIBRARY blas REQUIRED)

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
