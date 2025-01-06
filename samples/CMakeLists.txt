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

# common stuff - data generators and IO for loading/generating test data
add_subdirectory(common)

# ==================================================================
# sample that performs gpu-side closest-point queries between sets of
# float3 points
# ==================================================================
add_subdirectory(s01_closestPoint_points_cpu)
add_subdirectory(s01_closestPoint_points_gpu)

# ==================================================================
# sample that performs gpu-side closest-point queries between sets of
# float3 points
# ==================================================================
if (CUBQL_HAVE_CUDA)
  add_executable(cuBQL_sample02_distanceToTriangleMesh
    s02_distanceToTriangleMesh/distanceToTriangleMesh.cu
    )
  target_link_libraries(cuBQL_sample02_distanceToTriangleMesh
    # the cuda-side builders for float3 data
    cuBQL_cuda_float3
    # common samples stuff
    cuBQL_samples_common
    )
endif()


