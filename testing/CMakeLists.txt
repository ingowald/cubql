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

# value to use (during testing) for an arbitrarily 'large' N for which
# there are no obvious equivlanet cuda types
set(CUBQL_TEST_N "8" CACHE STRING "num dims for testing") 

if (NOT CUBQL_IS_SUBMODULE)
  add_subdirectory(unit-tests)
    
  add_library(cuBQL_testing
    helper/Generator.h
    helper/Generator.cu
    helper/CUDAArray.h
    helper/triangles.h
    helper/triangles.cu
    helper/instantiate_builders.cu
    )
  target_link_libraries(cuBQL_testing PUBLIC cuBQL_interface)
  target_compile_definitions(cuBQL_testing PUBLIC -DCUBQL_TEST_N=${CUBQL_TEST_N})

  add_executable(cuBQL_testGenerators helper/testGenerators.cu)
  target_link_libraries(cuBQL_testGenerators cuBQL_testing)

  # helper tool to plot 2D point generators
  add_executable(cuBQL_plotPoints helper/plotPoints.cu)
  target_link_libraries(cuBQL_plotPoints cuBQL_testing)
  # helper tool to plot 2D box generators
  add_executable(cuBQL_plotBoxes helper/plotBoxes.cu)
  target_link_libraries(cuBQL_plotBoxes cuBQL_testing)
  
  # ==================================================================
  # different helpers for generating *test data* for various kernels
  # ==================================================================

  # generate uniformly distributed points in [lower,upper] range
  # (default is [0,1]^3)
#  add_executable(cuBQL_makePoints_uniform makePoints_uniform.cu)
#  target_link_libraries(cuBQL_makePoints_uniform PUBLIC cuBQL_testing)

  # generate a clustered distribution of N points, where we use
  # sqrt(N) clusters whose centers are uniformly distributed in
  # [lower,upper] ([0,1]^3 by default), and then a gaussian
  # distribution around each of those centers.
#  add_executable(cuBQL_makePoints_clustered makePoints_clustered.cu)
#  target_link_libraries(cuBQL_makePoints_clustered PUBLIC cuBQL_testing)

# add_executable(cuBQL_convert_br convert_br.cu)
# target_link_libraries(cuBQL_convert_br PUBLIC cuBQL_testing)

  # generate points on a triangle mesh, using uniform density
#  add_executable(cuBQL_makePoints_sampleOBJ makePoints_sampleOBJ.cu)
#  target_link_libraries(cuBQL_makePoints_sampleOBJ PUBLIC cuBQL_testing)

  # generate list of triangles by reading an obj file
#  add_executable(cuBQL_makeTris_fromOBJ makeTris_fromOBJ.cu)
#  target_link_libraries(cuBQL_makeTris_fromOBJ PUBLIC cuBQL_testing)
#
#  add_executable(cuBQL_makeBoxes_fromPoints makeBoxes_fromPoints.cu)
#  target_link_libraries(cuBQL_makeBoxes_fromPoints PUBLIC cuBQL_testing)

#  add_executable(cuBQL_makeBoxes_fromTris makeBoxes_fromTris.cu)
#  target_link_libraries(cuBQL_makeBoxes_fromTris PUBLIC cuBQL_testing)
  


  # ==================================================================
  # actual kernels, benchmarks, etc that _exercise_ this library
  # ==================================================================
  option(DO_STATS "track and print traversal statsin fctAndKnn test case?" OFF)
  
  add_executable(cuBQL_fcpAndKnnPoints fcpAndKnn.cu)
  target_link_libraries(cuBQL_fcpAndKnnPoints PUBLIC cuBQL_testing)
  if (DO_STATS)
    target_compile_definitions(cuBQL_fcpAndKnnPoints PUBLIC -DDO_STATS=1)
  endif()
  
  add_executable(cuBQL_fcpAndKnnBoxes fcpAndKnn.cu)
  target_link_libraries(cuBQL_fcpAndKnnBoxes PUBLIC cuBQL_testing)
  target_compile_definitions(cuBQL_fcpAndKnnBoxes PUBLIC -DUSE_BOXES=1)

  add_executable(cuBQL_genBoxes genBoxes.cu)
  target_link_libraries(cuBQL_genBoxes PUBLIC cuBQL_testing)

  
endif()
