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

#pragma once

#include "cuBQL/math/math.h"
#include <vector>
#include <cuda_runtime.h>

namespace cuBQL {
  namespace test_rig {

    /*! allocator for test_rig::CUDAArray that will make this class
      allocate device memory */
    struct DeviceMem {
      inline static void alloc(void **pointer, size_t numBytes)
      {
        CUBQL_CUDA_CALL(Malloc(pointer,numBytes));
      }
    };
  
    /*! allocator for test_rig::CUDAArray that will make this class
      allocate managed memory */
    struct ManagedMem {
      inline static void alloc(void **pointer, size_t numBytes)
      {
        CUBQL_CUDA_CALL(MallocManaged(pointer,numBytes));
      }
    };

    /*! helper class or device (and/or managed) memory that operates
      similat to a std::vector, with upload/download hepers etc */
    template<typename T, typename Allocator=test_rig::DeviceMem>
    struct CUDAArray {

      inline CUDAArray() {}
      inline CUDAArray(size_t initSize) { resize(initSize); };
      inline ~CUDAArray() { free(); }
    
      /*! number of bytes allocated on the device */
      inline size_t numBytes() const { return N * sizeof(T); }

      /*! returns a (device side!-)reference to the first element. This
        primarily exists to allow things like "sizeof(*myCUDAArray)"; the
        address should not be dereferenced on the host */
      inline T operator*() const { return *d_data; }

      /*! returns if the device-memory is non-null */
      inline operator bool() const { return d_data; }
    
      /*! re-sizes device memory to specified number of elements,
        invalidating the current content. if device mem is already of
        exactly this size this is a no-op, otherwise old memory will
        be freed and new one allocated */
      void resize(int64_t N);

      void free() { resize(0); }
    
      /*! allocates a host-vector of the same size, downloads device
        data, and returns that vector */
      std::vector<T> download() const;

      /*! resizes (if required) the device memory to the same size of
        the vector, then uploads this host data to device */
      inline void upload(const std::vector<T> &vt);

      /*! upload host vector to given offset (counted in elements, not
        in bytes). Note that uplike resize(vector<>) this variant will
        _not_ do any resize, so this requires the device vector to be
        pre-allocated to sufficient size */
      inline void upload(const std::vector<T> &vt, size_t ofsInElements);

      /*! clears to allocated memory region to 0 */
      inline void bzero();
      T *get() const { return d_data; }

      /*! make this look like a std::vector ... */
      inline T *data() const { return d_data; }
    
      /*! make this look like a std::vector ... */
      inline size_t size() const { return N; }
    
      /*! allocated device pointer. this can change upon resizing. */
      T     *d_data = 0;
      size_t N      = 0;
    };

    /*! clears to allocated memory region to 0 */
    template<typename T, typename Allocator>
    inline void CUDAArray<T,Allocator>::bzero()
    {
      CUBQL_CUDA_CALL(Memset(d_data,0,N*sizeof(T)));
    }

    /*! upload host vector to given offset (counted in elements, not
      in bytes). Note that uplike resize(vector<>) this variant will
      _not_ do any resize, so this requires the device vector to be
      pre-allocated to sufficient size */
    template<typename T, typename Allocator>
    inline void CUDAArray<T,Allocator>::upload(const std::vector<T> &vt, size_t ofsInElements)
    {
      assert((ofsInElements + vt.size()) <= N);
      CUBQL_CUDA_CALL(Memcpy(d_data+ofsInElements,vt.data(),vt.size()*sizeof(T),cudaMemcpyDefault));
      CUBQL_CUDA_SYNC_CHECK();
    }

    /*! resizes (if required) the device memory to the same size of
      the vector, then uploads this host data to device */
    template<typename T, typename Allocator>
    inline void CUDAArray<T,Allocator>::upload(const std::vector<T> &vt)
    {
      resize(vt.size());
      size_t sz = vt.size() * sizeof(T);
      CUBQL_CUDA_CALL(Memcpy(d_data,vt.data(),vt.size()*sizeof(T),cudaMemcpyDefault));
      CUBQL_CUDA_SYNC_CHECK();
    }

    /*! re-sizes device memory to specified number of elements,
      invalidating the current content. if device mem is already of
      exactly this size this is a no-op, otherwise old memory will
      be freed and new one allocated */
    template<typename T, typename Allocator>
    void CUDAArray<T,Allocator>::resize(int64_t N)
    {
      if (N < 0)
        throw std::runtime_error("invalid array size!?");

      if (this->N == N) return;
      this->N = N;
      if (d_data) CUBQL_CUDA_CALL(Free(d_data));
      d_data = 0;
      if (N) Allocator::alloc((void**)&d_data,N*sizeof(T));
      assert(N == 0 || d_data != nullptr);
    }

    /*! allocates a host-vector of the same size, downloads device
      data, and returns that vector */
    template<typename T, typename Allocator>
    std::vector<T> CUDAArray<T,Allocator>::download() const
    {
      std::vector<T> host(N);
      CUBQL_CUDA_CALL(Memcpy(host.data(),d_data,N*sizeof(T),cudaMemcpyDefault));
      CUBQL_CUDA_SYNC_CHECK();
      return host;
    }

  } // ::cuBQL::test_rig
} // ::cuBQL

