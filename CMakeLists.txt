# ======================================================================== #
# Copyright 2023-2023 Ingo Wald                                            #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

cmake_policy(SET CMP0048 NEW)
project(cuBQL VERSION 0.0.1 LANGUAGES C CXX CUDA)

cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.18)
  cmake_policy(SET CMP0104 NEW)
endif()

if (NOT (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR}))
  set(CUBQL_IS_SUBPROJECT ON)
else()
  set(CUBQL_IS_SUBPROJECT OFF)
endif()

set(CUBQL_CUDA_ARCHITECTURES "auto" CACHE STRING "which arch to use for cubby" )

if (CUBQL_CUDA_ARCHITECTURES STREQUAL "auto")
  # leave CMAKE_CUDA_ARCHITECTURES to whatever the may may or may not have set
else()
  set(CMAKE_CUDA_ARCHITECTURES ${CUBQL_CUDA_ARCHITECTURES})
endif()

if (NOT (DEFINED CMAKE_CUDA_ARCHITECTURES))
  # new cmake policy: make sure that CMAKE_CUDA_ARCHITECTURES is set
  # to _something_, even if it's only "OFF". iw - seems to not apply on ubuntu 20.04!?
  if(${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.18)
    set(CMAKE_CUDA_ARCHITECTURES OFF)
  endif()
endif()


# ------------------------------------------------------------------
# general cmake project configs
# ------------------------------------------------------------------
if (CUBQL_IS_SUBPROJECT)
  mark_as_advanced(CUBQL_CUDA_ARCHITECTURES)
else()
  if(NOT SET_UP_CONFIGURATIONS_DONE)
    set(SET_UP_CONFIGURATIONS_DONE 1)

    # No reason to set CMAKE_CONFIGURATION_TYPES if it's not a multiconfig generator
    # Also no reason mess with CMAKE_BUILD_TYPE if it's a multiconfig generator.
    if(CMAKE_CONFIGURATION_TYPES) # multiconfig generator?
      set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE) 
    else()
      if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
      endif()
      set_property(CACHE CMAKE_BUILD_TYPE PROPERTY HELPSTRING "Choose the type of build")
      # set the valid options for cmake-gui drop-down list
      set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug;Release")
    endif()
  endif()

  SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
endif()

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.17)
  # nothing to do, setting CMAKE_CUDA_ARCHITECTURES is hte way to go
else()
  foreach (arch IN ITEMS ${CMAKE_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${arch},code=sm_${arch}")
  endforeach()
endif()

# a interface-only library that only sets include paths etc; when
# using this the user has to manulaly set CUBQL_BUILDER_INSTANTIATION
# in one of his/her source files
add_library(cuBQL_interface INTERFACE)
target_sources(cuBQL_interface INTERFACE
  # main public "interface" to this library
  cuBQL/bvh.h
  cuBQL/queries/fcp.h
  # general math struff to make public stuff work
  cuBQL/math/common.h
  cuBQL/math/math.h
  cuBQL/math/vec.h
  cuBQL/math/box.h
  # internal stuff
  cuBQL/impl/builder_common.h
  cuBQL/impl/sm_builder.h
  cuBQL/impl/sah_builder.h
  cuBQL/impl/gpu_builder.h
  cuBQL/impl/wide_gpu_builder.h
  )
target_include_directories(cuBQL_interface INTERFACE
  ${PROJECT_SOURCE_DIR}/
  )

# builds an actual static library that already contains
# template-instantiations of the builder(s), helper functions, etc
add_library(cuBQL_impl STATIC
  cuBQL/impl/instantiate_builders.cu
  )
target_link_libraries(cuBQL_impl PUBLIC
  cuBQL_interface
  )

if (NOT CUBQL_IS_SUBPROJECT)
  add_subdirectory(testing)
  add_subdirectory(samples)
endif()
