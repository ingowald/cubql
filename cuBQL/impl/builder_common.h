// ======================================================================== //
// Copyright 2023-2023 Ingo Wald                                            //
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

/* this file contains the entire builder; this should never be included directly */
#pragma once

#include "cuBQL/bvh.h"
#include <cub/cub.cuh>

namespace cuBQL {
  namespace gpuBuilder_impl {

    template<typename T, typename count_t>
    inline void _ALLOC(T *&ptr, count_t count, cudaStream_t s)
    { CUBQL_CUDA_CALL(MallocAsync((void**)&ptr,count*sizeof(T),s)); }
    
    template<typename T>
    inline void _FREE(T *&ptr, cudaStream_t s)
    { CUBQL_CUDA_CALL(FreeAsync((void*)ptr,s)); ptr = 0; }
    
    typedef enum : int8_t { OPEN_BRANCH, OPEN_NODE, DONE_NODE } NodeState;
    
    template<typename box_t>
    struct AtomicBox {
      inline __device__ void set_empty();
      inline __device__ float get_center(int dim) const;
      inline __device__ box_t make_box() const;

      inline __device__ float get_lower(int dim) const { return decode(lower[dim]); }
      inline __device__ float get_upper(int dim) const { return decode(upper[dim]); }

      int32_t lower[box_t::numDims];
      int32_t upper[box_t::numDims];

      inline static __device__ int32_t encode(float f);
      inline static __device__ float   decode(int32_t bits);
    };
    
    template<typename box_t>
    inline __device__ float AtomicBox<box_t>::get_center(int dim) const
    {
      return 0.5f*(decode(lower[dim])+decode(upper[dim]));
    }

    template<typename box_t>
    inline __device__ box_t AtomicBox<box_t>::make_box() const
    {
      box_t box;
#pragma unroll
      for (int d=0;d<box_t::numDims;d++) {
        (&box.lower.x)[d] = decode(lower[d]);
        (&box.upper.x)[d] = decode(upper[d]);
      }
      return box;
    }
    
    template<typename box_t>
    inline __device__ int32_t AtomicBox<box_t>::encode(float f)
    {
      const int32_t sign = 0x80000000;
      int32_t bits = __float_as_int(f);
      if (bits & sign) bits ^= 0x7fffffff;
      return bits;
    }
      
    template<typename box_t>
    inline __device__ float AtomicBox<box_t>::decode(int32_t bits)
    {
      const int32_t sign = 0x80000000;
      if (bits & sign) bits ^= 0x7fffffff;
      return __int_as_float(bits);
    }
    
    template<typename box_t>
    inline __device__ void AtomicBox<box_t>::set_empty()
    {
#pragma unroll
      for (int d=0;d<box_t::numDims;d++) {
        lower[d] = encode(+FLT_MAX);
        upper[d] = encode(-FLT_MAX);
      }
    }
    template<typename box_t, typename prim_box_t>
    inline __device__ void atomic_grow(AtomicBox<box_t> &abox, const prim_box_t &other)
    {
#pragma unroll
      for (int d=0;d<box_t::numDims;d++) {
        const int32_t enc_lower = AtomicBox<box_t>::encode(other.get_lower(d));
        const int32_t enc_upper = AtomicBox<box_t>::encode(other.get_upper(d));
        if (enc_lower < abox.lower[d]) atomicMin(&abox.lower[d],enc_lower);
        if (enc_upper > abox.upper[d]) atomicMax(&abox.upper[d],enc_upper);
      }
    }

    template<typename box_t>
    inline __device__ void atomic_grow(AtomicBox<box_t> &abox, const float3 &other)
    {
#pragma unroll
      for (int d=0;d<box_t::numDims;d++) {
        const int32_t enc = AtomicBox<box_t>::encode(get(other,d));
        if (enc < abox.lower[d]) atomicMin(&abox.lower[d],enc);
        if (enc > abox.upper[d]) atomicMax(&abox.upper[d],enc);
      }
    }
    
    struct BuildState {
      uint32_t  numNodes;
    };

  } // ::cuBQL::gpuBuilder_impl
} // ::cuBQL

