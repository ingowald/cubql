# ======================================================================== #
# Copyright 2023-2024 Ingo Wald                                            #
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

cmake_minimum_required(VERSION 3.16)



cmake_policy(SET CMP0048 NEW)
project(cuBQL VERSION 1.0.0 LANGUAGES C CXX)


include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
  message("#cuBQL: CUDA _FOUND_! building both cuda and host libs")

  # iw - do NOT use an 'option' - they can't do strings
#  set(CUBQL_CUDA_ARCHITECTURES "auto" CACHE STRING
#    "Which CUDA architecture to build for")
#  if (CUBQL_CUDA_ARCHITECTURES STREQUAL "auto")
#    if (CMAKE_CUDA_ARCHITECTURES)
#      set(CUBQL_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
#    else()
#      # since cmake 3.23
#      set (CUBQL_CUDA_ARCHITECTURES "all-major")
#    endif()
#    message("#cuBQL: cuda-archs set to 'auto': selected ${CUBQL_CUDA_ARCHITECTURES}")
#  else()
#    message("#cuBQL: cuda-archs explicitly set to ${CUBQL_CUDA_ARCHITECTURES}")
#  endif()
  enable_language(CUDA)
  set(CUBQL_HAVE_CUDA ON)
else()
  message("could not find CUDA - going to build only host libs")
  set(CUBQL_HAVE_CUDA OFF)
endif()


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.18)
  cmake_policy(SET CMP0104 NEW)
endif()

if (NOT (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR}))
  set(CUBQL_IS_SUBPROJECT ON)
else()
  set(CUBQL_IS_SUBPROJECT OFF)
endif()




if (CUBQL_IS_SUBPROJECT)
  if (NOT CMAKE_CUDA_ARCHITECTURES)
    message(FATAL_ERROR
      "#cuBQL: you've included cuBQL as a subproject, but "
      "CMAKE_CUDA_ARCHITECTURES "
      "is not set. cuBQL will automatically set some useful arch flags"
      "propagating target properties to libs that use that target we"
      "cannot automatically set the proper arch flags for ,"
      "your own targets, which will likely lead to some CUDA  "
      "errors like 'CUDA_ARCHITECTURES not set for ...'. "
      "Unless you already know what you're doing, most likely what"
      "you want is to add a like like\n"
      "   set(CMAKE_CUDA_ARCHITECTURES \"all-major\")\n"
      "or"
      "   set(CMAKE_CUDA_ARCHITECTURES \"native\")\n"
      "into your CMakeLists.txt."
      )
  endif()
else()
  if(${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.24)
    # "native" and "all-major" were introduced in cmake 3.24, older
    # versions dont' have that :-/
    set(CUBQL_CUDA_ARCHITECTURES "native" CACHE STRING
      "Which CUDA architecture to build for")
    set(CMAKE_CUDA_ARCHITECTURES ${CUBQL_CUDA_ARCHITECTURES})
  else()
    set(CMAKE_CUDA_ARCHITECTURES OFF)
  endif()
endif()


# ------------------------------------------------------------------
# general cmake project configs
# ------------------------------------------------------------------
if (CUBQL_IS_SUBPROJECT)
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

add_subdirectory(cuBQL)

option(CUBQL_ENABLE_TESTING "Enable Testing?" OFF)
if (NOT CUBQL_IS_SUBPROJECT)
  add_subdirectory(samples)
  if (CUBQL_ENABLE_TESTING)
    add_subdirectory(testing)
  endif()
endif()
