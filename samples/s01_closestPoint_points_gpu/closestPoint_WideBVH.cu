// ======================================================================== //
// Copyright 2023-2024 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

/*! \file closestPointGPU.cu Implements a small demo-app that
  generates a set of data points, another set of query points, and
  then uses cuBQL to perform closest-point qeuries (ie, it finds,
  for each query point, the respectively closest data point */

// cuBQL itself, and the BVH type(s) it defines
#include "cuBQL/bvh.h"
#include "cuBQL/builder/cuda.h"
// some specialized query kernels for find-closest, on 'points' data
#include "cuBQL/queries/points/findClosest.h"
// helper class to generate various data distributions
#include "samples/common/Generator.h"

using namespace cuBQL;

enum { BVH_WIDTH = 8 };

typedef cuBQL::WideBVH<float,3,BVH_WIDTH> wide_bvh3f;

__global__
void computeBoxes(box3f *d_boxes, const vec3f *d_data, int numData)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (tid >= numData) return;

  d_boxes[tid] = box3f().including(d_data[tid]);
}

__global__
void runQueries(wide_bvh3f bvh,
                const vec3f *d_data,
                const vec3f *d_queries,
                int numQueries)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (tid >= numQueries) return;
  
  vec3f queryPoint = d_queries[tid];
  int closestID = cuBQL::points::findClosest(/* the cubql bvh we've built */
                                            bvh,
                                            /* data that this bvh was built over*/
                                            d_data,
                                            queryPoint);
  vec3f closestPoint = d_data[closestID];
  printf("[%i] closest point to (%f %f %f) is point #%i, at (%f %f %f)\n",
         tid,
         queryPoint.x,
         queryPoint.y,
         queryPoint.z,
         closestID,
         closestPoint.x,
         closestPoint.y,
         closestPoint.z);
}


int main(int, char **)
{
  int numDataPoints = 10000;
  int numQueryPoints = 20;
  /*! generate 10,000 uniformly distributed data points */
  std::vector<vec3f> dataPoints
    = cuBQL::samples::convert<float>
    (cuBQL::samples::UniformPointGenerator<3>()
     .generate(numDataPoints,290374));
  std::cout << "#cubql: generated " << dataPoints.size()
            << " data points" << std::endl;
  std::vector<vec3f> queryPoints
    = cuBQL::samples::convert<float>
    (cuBQL::samples::UniformPointGenerator<3>()
     .generate(numQueryPoints,/*seed*/1234567));
  std::cout << "#cubql: generated " << queryPoints.size()
            << " query points" << std::endl;

  vec3f *d_queryPoints = 0;
  vec3f *d_dataPoints = 0;
  box3f *d_primBounds = 0;
  CUBQL_CUDA_CALL(Malloc((void **)&d_queryPoints,queryPoints.size()*sizeof(vec3f)));
  CUBQL_CUDA_CALL(Memcpy(d_queryPoints,queryPoints.data(),
                         queryPoints.size()*sizeof(queryPoints[0]),
                         cudaMemcpyDefault));
  CUBQL_CUDA_CALL(Malloc((void **)&d_dataPoints,dataPoints.size()*sizeof(vec3f)));
  CUBQL_CUDA_CALL(Memcpy(d_dataPoints,dataPoints.data(),
                         dataPoints.size()*sizeof(dataPoints[0]),
                         cudaMemcpyDefault));
  CUBQL_CUDA_CALL(Malloc((void **)&d_primBounds,dataPoints.size()*sizeof(box3f)));
  computeBoxes<<<divRoundUp(numDataPoints,128),128>>>
    (d_primBounds,d_dataPoints,numDataPoints);

  // generate cuBQL bvh
  wide_bvh3f bvh;
  cuBQL::gpuBuilder(bvh,d_primBounds,numDataPoints,BuildConfig());
  runQueries<<<divRoundUp(numQueryPoints,128),128>>>
    (bvh,d_dataPoints,d_queryPoints,numQueryPoints);
  
  CUBQL_CUDA_SYNC_CHECK();
  return 0;
}
 
