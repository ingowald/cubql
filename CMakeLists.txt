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
set(CMAKE_BUILD_TYPE_INIT "Release")
project(cuBQL VERSION 1.0.0 LANGUAGES C CXX)


if (NOT CMAKE_CUDA_COMPILER)
  include(CheckLanguage)
  check_language(CUDA)
endif()

if (CMAKE_CUDA_COMPILER)
  message("#cuBQL: CUDA _FOUND_! building both cuda and host libs")
  enable_language(CUDA)
  set(CUBQL_HAVE_CUDA ON)
else()
  message(AUTHOR_WARNING
    " ===========================================================\n"
    " #cuBQL: could not find CUDA - going to build only host libs\n"
    " ===========================================================\n"
    )
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




# ------------------------------------------------------------------
# check if we're a subproject, and if so, only include the library
# itself
# ------------------------------------------------------------------
if (CUBQL_IS_SUBPROJECT)
  # ------------------------------------------------------------------
  # we're included from a parent project; it's this parent project's
  # job to project-specific configs like configure output paths and
  # build type, and to set CUQBL_CUDA_ARCHITECTURES to whatever
  # arch(s) the project wants us to build for.
  #
  # Check if CUBQL_CUDA_ARCHITECTURES is set, and throw an error if
  # not
  # ------------------------------------------------------------------
  if ((NOT CMAKE_CUDA_ARCHITECTURES)
      OR
      ((${CMAKE_VERSION} VERSION_LESS 3.24) AND ("${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "52"))
      )
    message(FATAL_ERROR
      " =====================================================================\n"
      " #cuBQL: you've included cuBQL as a subproject (as it should!),\n"
      " but CMAKE_CUDA_ARCHITECTURES was not set. As it is\n"
      " almost certainly going to cause some problems in your project\n"
      " if we were we were to build cuBQL for different archs than your own\n"
      " project, we will cowardly refuse do so. Before including cuBQL\n"
      " you should explicitly set the cmake variable\n"
      " CMAKE_CUDA_ARCHITECTURES to whatever the main project will want\n"
      " to use, too.\n"
      
      " \n"
      
      " If in doubt as to what to use, at least for cmake >= 3.24 you can also use\n"
      "   set(CMAKE_CUDA_ARCHITECTURES \"all-major\")\n"
      "   add_subdirectory(<path-to-cubql) ...)\n"
      " or\n"
      "   set(CMAKE_CUDA_ARCHITECTURES \"native\")\n"
      "   add_subdirectory(<path-to-cubql) ...)\n"
      
      "====================================================================="
      )
  endif()
else()
  if (CMAKE_CUDA_ARCHITECTURES)
    # set on the cmdline, for a local build
    set(CUBQL_CUDA_ARCHITECTURES_INIT "${CMAKE_CUDA_ARCHITECTURES}")
  elseif (${CMAKE_VERSION} VERSION_LESS 3.24)
    message(AUTHOR_WARNING
      "====================================================================="
      
      "\n"

      "cuBQL: you're building with cmake version < 3.24, which doesn't
      yet support the 'native' flag for CMAKE_CUDA_ARCHITECTURES.
      We're defaulting to '70;80'; if that doesn't capture the kind of
      GPU you are using, please either specify whatever arch(s) you
      want to build for, or use a newer cmake for which we'll default
      to 'native'"
      
      "\n"

      "====================================================================="
      )
    set(CUBQL_CUDA_ARCHITECTURES_INIT "70;80")
  else()
    set(CUBQL_CUDA_ARCHITECTURES_INIT "native")
  endif()
  set(CUBQL_CUDA_ARCHITECTURES "${CUBQL_CUDA_ARCHITECTURES_INIT}" CACHE STRING
    "Which CUDA architecture to build for")
  set(CMAKE_CUDA_ARCHITECTURES "${CUBQL_CUDA_ARCHITECTURES}")
  #  message("CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}")
  
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
  
  SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  # we're building locally - set 'global' cuda arch for samples and test cases
endif()

message("#cuBQL: compiling for CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
add_subdirectory(cuBQL)

option(CUBQL_ENABLE_TESTING "Enable Testing?" OFF)
if (NOT CUBQL_IS_SUBPROJECT)
  add_subdirectory(samples)
  if (CUBQL_ENABLE_TESTING)
    add_subdirectory(testing)
  endif()
endif()
