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

# a interface-only library that only sets include paths etc; when
# using this the user has to manulaly set CUBQL_BUILDER_INSTANTIATION
# in one of his/her source files
add_library(cuBQL INTERFACE)

target_sources(cuBQL INTERFACE
  # main public "interface" to this library
  bvh.h
  # general math struff to make public stuff work
  math/common.h
  math/math.h
  math/vec.h
  math/box.h
  # general (lambda-templated) traversal routines
  traversal/shrinkingRadiusQuery.h
  traversal/fixedBoxQuery.h
  # host side builder(s)
  builder/cpu/spatialMedian.h
  # cuda side builder(s)
  builder/cuda/builder_common.h
  builder/cuda/sm_builder.h
  builder/cuda/sah_builder.h
  builder/cuda/gpu_builder.h
  builder/cuda/radix.h
  builder/cuda/rebinMortonBuilder.h
  builder/cuda/wide_gpu_builder.h
  )
target_include_directories(cuBQL INTERFACE
  ${PROJECT_SOURCE_DIR}
  )
  
# helper for creating type- and device-specific implementations of
# cubql; i.e., one for host_float4, one for cuda_int3, etc. Since
# everything other than the builders are entirely header only these
# type-specific targets will, in fact, only contain instantiations of
# the specific builders for the given type and device
function(add_specific_instantiation device suffix T D)
  add_library(cuBQL_${device}_${T}${D} SHARED EXCLUDE_FROM_ALL
    impl/instantiate_builders.${suffix}
    )
  if (${device} STREQUAL "cuda")
    target_compile_definitions(cuBQL_${device}_${T}${D}
      PUBLIC
      -DCUBQL_HAVE_CUDA=1
      )
#    set_target_properties(cuBQL_${device}_${T}${D}
#      PROPERTIES
#      CUDA_ARCHITECTURES "${CUBQL_CUDA_ARCHITECTURES}"
#      )
  endif()
  set_target_properties(cuBQL_${device}_${T}${D}
    PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    )
  target_link_libraries(cuBQL_${device}_${T}${D}
    PUBLIC
    cuBQL
    )
  target_compile_definitions(cuBQL_${device}_${T}${D}
    PRIVATE
    -DCUBQL_INSTANTIATE_T=${T}
    -DCUBQL_INSTANTIATE_D=${D}
    )

  add_library(cuBQL_${device}_${T}${D}_static STATIC EXCLUDE_FROM_ALL
    impl/instantiate_builders.${suffix}
    )
  if (${device} STREQUAL "cuda")
    target_compile_definitions(cuBQL_${device}_${T}${D}_static
      PUBLIC
      -DCUBQL_HAVE_CUDA=1
      )
    set_target_properties(cuBQL_${device}_${T}${D}_static
      PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      )
#    set_target_properties(cuBQL_${device}_${T}${D}_static
#      PROPERTIES
#      CUDA_ARCHITECTURES "${CUBQL_CUDA_ARCHITECTURES}"
#      )
  endif()
  set_target_properties(cuBQL_${device}_${T}${D}_static
    PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    )
  target_link_libraries(cuBQL_${device}_${T}${D}_static
    PUBLIC
    cuBQL
    )
  target_compile_definitions(cuBQL_${device}_${T}${D}_static
    PRIVATE
    -DCUBQL_INSTANTIATE_T=${T}
    -DCUBQL_INSTANTIATE_D=${D}
    )
endfunction()

# ------------------------------------------------------------------
# generate all type/dim specific targets, ie, one target for each
# {int,float,double,long}x{2,3,4}. Each such target contains the
# pre-compiled builder(s) for that specific type/dimension, for the
# case where the user does NOT want to use the header-only
# mechanism. To avoid "polluting" the user's project with lots of
# different targets that he or she may or may not need we add all
# those as 'EXCLUDE_FROM_ALL', so only those that get actually used
# will actually get built
# ------------------------------------------------------------------
foreach(T IN ITEMS float int double longlong)
  foreach(D IN ITEMS 2 3 4)
    add_specific_instantiation(cpu cpp ${T} ${D})
    if (CUBQL_HAVE_CUDA)
      add_specific_instantiation(cuda cu ${T} ${D})
    endif()
  endforeach()
endforeach()

# the collection of all different type-specific queries currently
# supplied by cubql. all these should be header-only and should all
# automatically on both cpu and cuda, so this is a INTERFACE library
add_library(cuBQL_queries INTERFACE
  #
  queries/points/knn.h
  #
  queries/points/findClosest.h
  queries/points/knn.h
  )
target_link_libraries(cuBQL_queries
  INTERFACE
  cuBQL
  )



