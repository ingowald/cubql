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

#include "cuBQL/impl/sm_builder.h"

namespace cuBQL {
  namespace mortonBuilder_impl {
    using gpuBuilder_impl::atomic_grow;
    using gpuBuilder_impl::_ALLOC;
    using gpuBuilder_impl::_FREE;
    
    template<typename T, int D> struct Quantizer;

    template<int D> struct numMortonBits;
    template<> struct numMortonBits<2> { enum { value = 31 }; };
    template<> struct numMortonBits<3> { enum { value = 20 }; };
    template<> struct numMortonBits<4> { enum { value = 15 }; };
    
    template<int D>
    struct Quantizer<float,D> {
      using vec_t = cuBQL::vec_t<float,D>;
      using box_t = cuBQL::box_t<float,D>;
      
      inline __device__ cuBQL::vec_t<uint32_t,D> quantize(vec_t P) const
      {
        using vec_ui = cuBQL::vec_t<uint32_t,D>;

        vec_ui cell = vec_ui((P-quantizeBias)*quantizeScale);
        cell = min(cell,vec_ui((1u<<numMortonBits<D>::value)-1));
        return cell;
      }
        
      inline __device__ void init(cuBQL::box_t<float,D> centBounds)
      {
        quantizeBias
          = centBounds.lower;
        quantizeScale
          = vec_t(1<<numMortonBits<D>::value)
          * rcp(max(vec_t(reduce_max(centBounds.size())),vec_t(1e-20f)));
      }
        
      /*! coefficients of `scale*(x-bias)` in the 21-bit fixed-point
          quantization operation that does
          `(x-centBoundsLower)/(centBoundsSize)*(1<<10)`. Ie, bias is
          centBoundsLower, and scale is `(1<<10)/(centBoundsSize)` */
      vec_t CUBQL_ALIGN(16) quantizeBias, CUBQL_ALIGN(16) quantizeScale;
    };

    template<int D>
    struct Quantizer<double,D> {
      using vec_t = cuBQL::vec_t<double,D>;
      using box_t = cuBQL::box_t<double,D>;
      
      inline __device__ cuBQL::vec_t<uint32_t,D> quantize(vec_t P) const
      {
        using vec_ui = cuBQL::vec_t<uint32_t,D>;

        vec_ui cell = vec_ui((P-quantizeBias)*quantizeScale);
        cell = min(cell,vec_ui((1u<<numMortonBits<D>::value)-1));
        return cell;
      }
        
      inline __device__ void init(cuBQL::box_t<double,D> centBounds)
      {
        quantizeBias
          = centBounds.lower;
        quantizeScale
          = vec_t(1<<numMortonBits<D>::value)
          * rcp(max(vec_t(reduce_max(centBounds.size())),vec_t(1e-20f)));
      }
        
      /*! coefficients of `scale*(x-bias)` in the 21-bit fixed-point
          quantization operation that does
          `(x-centBoundsLower)/(centBoundsSize)*(1<<10)`. Ie, bias is
          centBoundsLower, and scale is `(1<<10)/(centBoundsSize)` */
      vec_t CUBQL_ALIGN(16) quantizeBias, CUBQL_ALIGN(16) quantizeScale;
    };

    template<int D>
    struct Quantizer<int,D> {
      using vec_t = cuBQL::vec_t<int,D>;
      using box_t = cuBQL::box_t<int,D>;
      
      inline __device__ void init(cuBQL::box_t<int,D> centBounds)
      {
        quantizeBias = centBounds.lower;
        int maxValue = reduce_max(centBounds.size());
        shlBits = __clz(maxValue);
      }

      inline __device__ cuBQL::vec_t<uint32_t,D> quantize(vec_t P) const
      {
        cuBQL::vec_t<uint32_t,D> cell = cuBQL::vec_t<uint32_t,D>(P-quantizeBias);
        // move all relevant bits to top
        cell = cell << shlBits;
        return cell >> cuBQL::vec_t<uint32_t,D>(32-numMortonBits<D>::value);
      }
        
      /*! coefficients of `scale*(x-bias)` in the 21-bit fixed-point
          quantization operation that does
          `(x-centBoundsLower)/(centBoundsSize)*(1<<10)`. Ie, bias is
          centBoundsLower, and scale is `(1<<10)/(centBoundsSize)` */
      vec_t quantizeBias;
      int   shlBits;
    };
    


    
    /*! maintains high-level summary of the build process */
    template<typename T, int D>
    struct CUBQL_ALIGN(16) BuildState {
      using vec_t = cuBQL::vec_t<T,D>;//float,3>;
      using box_t = cuBQL::box_t<T,D>;//float,3>;
      using bvh_t = cuBQL::BinaryBVH<T,D>;//float,3>;
      using atomic_box_t = gpuBuilder_impl::AtomicBox<box_t>;
      
      /*! number of nodes alloced so far */
      int numNodesAlloced;

      /*! number of *valid* prims that get put into the BVH; this will
          be computed by sarting with the input number of prims, and
          removing those that have invalid/empty bounds */
      int numValidPrims;
      
      /*! bounds of prim centers, relative to which we will computing
        morton codes */
      atomic_box_t a_centBounds;
      box_t        centBounds;
      /*! coefficients of `scale*(x-bias)` in the 21-bit fixed-point
          quantization operation that does
          `(x-centBoundsLower)/(centBoundsSize)*(1<<21)`. Ie, bias is
          centBoundsLower, and scale is `(1<<21)/(centBoundsSize)` */
      // vec_t CUBQL_ALIGN(16) quantizeBias, CUBQL_ALIGN(16) quantizeScale;
      Quantizer<T,D> quantizer;
    };

    template<typename T, int D>
    __global__
    void clearBuildState(BuildState<T,D> *buildState,
                         int          numPrims)
    {
      if (threadIdx.x != 0) return;
      
      buildState->a_centBounds.clear();
      // let's _start_ with the assumption that all are valid, and
      // subtract those later on that are not.
      buildState->numValidPrims   = numPrims;
      buildState->numNodesAlloced = 0;
    }
    
    template<typename T, int D>
    __global__
    void fillBuildState(BuildState<T,D>  *buildState,
                        const typename BuildState<T,D>::box_t *prims,
                        int          numPrims)
    {
      using atomic_box_t = typename BuildState<T,D>::atomic_box_t;
      using box_t        = typename BuildState<T,D>::box_t;
      
      __shared__ atomic_box_t l_centBounds;
      if (threadIdx.x == 0)
        l_centBounds.clear();
      
      // ------------------------------------------------------------------
      __syncthreads();
      // ------------------------------------------------------------------
      int tid = threadIdx.x + blockIdx.x*blockDim.x;
      if (tid < numPrims) {
        box_t prim = prims[tid];
        if (!prim.empty()) 
          atomic_grow(l_centBounds,prim.center());
      }
      
      // ------------------------------------------------------------------
      __syncthreads();
      // ------------------------------------------------------------------
      if (threadIdx.x == 0)
        atomic_grow(buildState->a_centBounds,l_centBounds);
    }

    template<typename T, int D>
    __global__
    void finishBuildState(BuildState<T,D>  *buildState)
    {
      using ctx_t = BuildState<T,D>;
      using atomic_box_t = typename ctx_t::atomic_box_t;
      using box_t        = typename ctx_t::box_t;
      
      if (threadIdx.x != 0) return;
      
      box_t centBounds = buildState->a_centBounds.make_box();
      buildState->centBounds = centBounds;
      /* from above: coefficients of `scale*(x-bias)` in the 21-bit
        fixed-point quantization operation that does
        `(x-centBoundsLower)/(centBoundsSize)*(1<<21)`. Ie, bias is
        centBoundsLower, and scale is `(1<<21)/(centBoundsSize)` */
      // buildState->quantizeBias
      //   = centBounds.lower;
      // buildState->quantizeScale
      //   = vec3f(1<<21)*rcp(max(centBounds.size(),vec3f(1e-20f)));
      buildState->quantizer.init(centBounds);
    }


    inline __device__
    uint64_t shiftBits(uint64_t x, uint64_t maskOfBitstoMove, int howMuchToShift)
    { return ((x & maskOfBitstoMove)<<howMuchToShift) | (x & ~maskOfBitstoMove); }

    
    /* morton code computation: how the bits shift for 32 input bits, in 1:1 pattern:

       desired final step:
       _P_O._N_M:_L_K._J_I:_H_G._F_E:_D_C._B_A:_p_o._n_m:_l_k._j_i:_h_g._f_e:_d_c._b_a:

       stage -1
       __PO.__NM:__LK.__JI:__HG.__FE:__DC.__BA:__po.__nm:__lk.__ji:__hg.__fe:__dc.__ba:
       mask:
       0010.0010:0010.0010:0010.0010:0010.0010:0010.0010:0010.0010:0010.0010:0010.0010
       move by 1

       stage -2
       ____.PONM:____.LKJI:____.HGFE:____.DCBA:____.ponm:____.lkji:____.hgfe:____.dcba:
       mask:
       0000.1100:0000.1100:0000.1100:0000.1100:0000.1100:0000.1100:0000.1100:0000.1100
       move by 2

       stage -3
       ____.____:PONM.LKJI:____.____:HGFE.DCBA:____.____:ponm.lkji:____.____:hgfe.dcba:
       mask:
       0000.0000:1111.0000:0000.0000:1111.0000:0000.0000:1111.0000:0000.0000:1111.0000
       move by 4

       stage -4
       ____.____:____.____:PONM.LKJI:HGFE.DCBA:____.____:____.____:ponm.lkji:hgfe.dcba:
       mask:
       0000.0000:0000.0000:1111.1111:0000.0000:0000.0000:0000.0000:1111.1111.0000:0000
       move by 8

       stage -5
       ____.____:____.____:____.____:____.____:PONM.LKJI:HGFE.DCBA:ponm.lkji:hgfe.dcba:
       move:
       0000.0000:0000.0000:0000.0000:0000.0000:1111.1111:1111.1111:0000.0000:0000.0000
       move by 16
    */
    inline __device__
    uint64_t insert_one_wide_gaps(uint64_t x)
    {
      x = shiftBits(x,
                    0b0000000000000000000000000000000011111111111111110000000000000000ull,16);
      x = shiftBits(x,
                    0b0000000000000000111111110000000000000000000000001111111100000000ull,8);
      x = shiftBits(x,
                    0b0000000011110000000000001111000000000000111100000000000011110000ull,4);
      x = shiftBits(x,
                    0b0000110000001100000011000000110000001100000011000000110000001100ull,2);
      x = shiftBits(x,
                    0b0010001000100010001000100010001000100010001000100010001000100010ull,1);
      return x;
    }
    
    /* morton code computation: how the bits shift for 21 input bits, in 2:1 pattern:

       desired final step:
       ___u.__t_:_s__.r__q:__p_._o__:n__m.__l_:_k__.j__i:__h_._g__:f__e.__d_:_c__.b__a:

       stage -1
       ___u.____:ts__.__rq:____.po__:__nm.____:lk__.__ji:____.hg__:__fe.____:dc__.__ba:
       mask:
       0000.0000:1000.0010:0000.1000:0010.0000:1000.0010:0000.1000:0010.0000:1000.0010
       move by 2
       hex    00:       82:       08:       20:       82:       08:       20:       82

       stage -2
       ___u.____:____.tsrq:____.____:ponm.____:____.lkji:____.____:hgfe.____:____.dcba:
       mask:
       0000.0000:0000.1100:0000.0000:1100.0000:0000.1100:0000.0000:1100.0000:0000.1100
       move by 4
       hex    00:       0c:       00:       c0:       0c:       00:       c0:       0c

       stage -3
       ____.____:___u.tsrq:____.____:____.____:ponm.lkji:____.____:____.____:hgfe.dcba:
       mask:
       0000.0000:1111.0000:0000.0000:0000.0000:1111.0000:0000.0000:0000.0000:1111.0000
       move by 8
       hex    00:       f0:       00:       00:       f0:       00:       00:       f0

       stage -4
       ____.____:___u.tsrq:____.____:____.____:____.____:____.____:ponm.lkji:hgfe.dcba:
       mask:
       0000.0000:0000.0000:0000.0000:0000.0000:0000.0000:0000.0000:1111.1111.0000:0000
       move by 16
       hex     00:      00:       00:       00:       00:       00:       ff:       00

       stage -5
       ____.____:____.____:____.____:____.____:____.____:___u.tsrq:ponm.lkji:hgfe.dcba:
       move:
       0000.0000:0000.0000:0000.0000:0000.0000:0000.0000:0001.1111:0000.0000:0000.0000
       move by 32
       hex    00:       00:       00:       00:       00:       1f:       00:       00
    */
    inline __device__
    uint64_t insert_two_wide_gaps(uint64_t x)
    {
      //hex    00:       00:       00:       00:       00:       10:       00:       00
      x = shiftBits(x,0x00000000001f0000ull,32); 
      //hex     00:      00:       00:       00:       00:       00:       ff:       00
      x = shiftBits(x,0x000000000000ff00ull,16); 
      //hex    00:       f0:       00:       00:       f0:       00:       00:       f0
      x = shiftBits(x,0x00f00000f00000f0ull,8); 
      //hex    00:       0c:       00:       c0:       0c:       00:       c0:       0c
      x = shiftBits(x,0x000c00c00c00c00cull,4); 
      //hex    00:       82:       08:       20:       82:       08:       20:       82
      x = shiftBits(x,0x0082082082082082ull,2);
      return x;
    }
    
    
    inline __device__
    uint64_t interleaveBits(vec2ui mortonCell)
    {
      return
        (insert_one_wide_gaps(mortonCell.x) << 0)
        |
        (insert_one_wide_gaps(mortonCell.y) << 1);
      // return bitInterleave11(mortonCell.x,mortonCell.y);
    }

    inline __device__
    uint64_t interleaveBits(vec3ui mortonCell)
    {
      return
        (insert_two_wide_gaps(mortonCell.z) << 2) |
        (insert_two_wide_gaps(mortonCell.y) << 1) |
        (insert_two_wide_gaps(mortonCell.x) << 0);
    }

    inline __device__
    uint64_t interleaveBits(vec4ui mortonCell)
    {
      uint64_t xy = interleaveBits(vec2ui{mortonCell.x,mortonCell.y});
      uint64_t zw = interleaveBits(vec2ui{mortonCell.z,mortonCell.w});
      return
        (insert_one_wide_gaps(xy) << 0) |
        (insert_one_wide_gaps(zw) << 1);
    }
    
    template<typename T, int D>
    inline __device__
    // uint32_t computeMortonCode(vec3f P, vec3f quantizeBias, vec3f quantizeScale)
    uint64_t computeMortonCode(typename BuildState<T,D>::vec_t P,
                               const Quantizer<T,D> quantizer)
    {
      return interleaveBits(quantizer.quantize(P));
    }    
    
    template<typename T, int D>
    __global__
    void computeUnsortedKeysAndPrimIDs(uint64_t    *mortonCodes,
                                       uint32_t    *primIDs,
                                       BuildState<T,D>  *buildState,
                                       const typename BuildState<T,D>::box_t *prims,
                                       int numPrims)
    {
      using atomic_box_t = typename BuildState<T,D>::atomic_box_t;
      using box_t        = typename BuildState<T,D>::box_t;
      
      int tid = threadIdx.x + blockIdx.x*blockDim.x;
      if (tid >= numPrims) return;

      int primID = tid;
      box_t prim = prims[primID];
      while (prim.empty()) {
        primID = atomicAdd(&buildState->numValidPrims,-1)-1;
        if (tid >= primID) return;
        prim = prims[primID];
      }

      primIDs[tid] = primID;
      mortonCodes[tid]
        = computeMortonCode(prim.center(),buildState->quantizer);
                            // buildState->quantizeBias,
                            // buildState->quantizeScale);
    }

    struct TempNode {
      union {
        /*! nodes that have been opened by their parents, but have not
          yet been finished. such nodes descibe a list of
          primitives; the range of keys covered in this subtree -
          which can/will be used to determine where to split - is
          encoded in first and last key in that range */
        struct {
          uint32_t begin;
          uint32_t end;
        } open;
        /*! nodes that are finished and done */
        struct {
          uint32_t offset;
          uint32_t count;
        } finished;
        // force alignment to 8-byte values, so compiler can
        // read/write more efficiently
        uint64_t bits;
      };
    };


    inline __device__
    bool findSplit(int &split,
                   const uint64_t *__restrict__ keys,
                   int begin, int end)
    {
      uint64_t firstKey = keys[begin];
      uint64_t lastKey  = keys[end-1];
      
      if (firstKey == lastKey)
        // same keys entire range - no split in there ....
        return false;
      
      int numMatchingBits = __clzll(firstKey ^ lastKey);
      // the first key in the plane we're searching has
      // 'numMatchingBits+1' top bits of lastkey, and 0es otherwise
      const uint64_t searchKey = lastKey & (0xffffffffffffffffull<<(63-numMatchingBits));

      while (end > begin) {
        int mid = (begin+end)/2;
        if (keys[mid] < searchKey) {
          begin = mid+1;
        } else {
          end = mid;
        }
      }
      split = begin;
      return true;
    }

    template<typename T, int D>
    __global__
    void initNodes(BuildState<T,D> *buildState,
                   TempNode   *nodes,
                   int numValidPrims)
    {
      if (threadIdx.x != 0) return;
      
      buildState->numNodesAlloced = 2;
      TempNode n0, n1;
      n0.open.begin = 0;
      n0.open.end   = numValidPrims;
      n1.bits = 0;
      nodes[0] = n0;
      nodes[1] = n1;
    }

    template<typename T, int D>
    __global__
    void createNodes(BuildState<T,D> *buildState,
                     int leafThreshold,
                     TempNode *nodes,
                     int begin, int end,
                     const uint64_t *keys)
    {
      using atomic_box_t = typename BuildState<T,D>::atomic_box_t;
      using box_t        = typename BuildState<T,D>::box_t;
      
      __shared__ int l_allocOffset;
      
      if (threadIdx.x == 0)
        l_allocOffset = 0;
      // ==================================================================
      __syncthreads();
      // ==================================================================
      
      int tid = threadIdx.x+blockIdx.x*blockDim.x;
      int nodeID = begin + tid;
      bool validNode = (nodeID < end);
      int split   = -1;
      int childID = -1;
      TempNode node;
      
      if (validNode) {
        node = nodes[nodeID];
        int size = node.open.end - node.open.begin;
        if (size <= leafThreshold) {
          // we WANT to make a leaf
          node.finished.offset = node.open.begin;
          node.finished.count  = size;
        } else if (!findSplit(split,keys,node.open.begin,node.open.end)) {
          // we HAVE TO make a leaf because we couldn't split
          node.finished.offset = node.open.begin;
          node.finished.count  = size;
        } else {
          // we COULD split - yay!
          childID = atomicAdd(&l_allocOffset,2);
        }
      }
      
      // ==================================================================
      __syncthreads();
      // ==================================================================
      if (threadIdx.x == 0)
        l_allocOffset = atomicAdd(&buildState->numNodesAlloced,l_allocOffset);
      // ==================================================================
      __syncthreads();
      // ==================================================================
      if (childID >= 0) {
        childID += l_allocOffset;
        TempNode c0, c1;
        c0.open.begin = node.open.begin;
        c0.open.end   = split;
        c1.open.begin = split;
        c1.open.end   = node.open.end;
        // we COULD actually write those as a int4 if we really wanted
        // to ...
        nodes[childID+0]     = c0;
        nodes[childID+1]     = c1;
        node.finished.offset = childID;
        node.finished.count  = 0;
      }
      if (validNode)
        nodes[nodeID] = node;
    }

    template<typename T, int D>
    __global__
    void writeFinalNodes(typename bvh_t<T,D>::Node *finalNodes,
                         const TempNode *__restrict__ tempNodes,
                         int numNodes)
    {
      int tid = threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= numNodes) return;
      typename BinaryBVH<T,D>::Node node;
      TempNode tempNode = tempNodes[tid];
      node.offset = tempNode.finished.offset;
      node.count = tempNode.finished.count;

      if (tid == 1)
        node.offsetAndCountBits = 0;
      
      finalNodes[tid].offsetAndCountBits = node.offsetAndCountBits;
    }
    
    template<typename T, int D>
    void build(bvh_t<T,D>        &bvh,
               const typename BuildState<T,D>::box_t       *boxes,
               int                numPrims,
               BuildConfig        buildConfig,
               cudaStream_t       s,
               GpuMemoryResource &memResource)
    {
      const int makeLeafThreshold
        = (buildConfig.makeLeafThreshold > 0)
        ? min(buildConfig.makeLeafThreshold,buildConfig.maxAllowedLeafSize)
        : 1;

      // ==================================================================
      // first MAJOR step: compute buildstate's centBounds value,
      // which we need for computing morton codes.
      // ==================================================================
      /* step 1.1, init build state; in particular, clear the shared
        centbounds we need to atomically grow centroid bounds in next
        step */
      BuildState<T,D> *d_buildState = 0;
      _ALLOC(d_buildState,1,s,memResource);
      clearBuildState<<<32,1,0,s>>>
        (d_buildState,numPrims);
      /* step 1.2, compute the centbounds we need for morton codes; we
         do this by atomically growing this shared centBounds with
         each (non-invalid) input prim */
      fillBuildState<<<divRoundUp(numPrims,1024),1024,0,s>>>
        (d_buildState,boxes,numPrims);
      /* step 1.3, convert vom atomic_box to regular box, which is
         cheaper to digest for the following kernels */
      finishBuildState<<<32,1,0,s>>>
        (d_buildState);

      static BuildState<T,D> *h_buildState = 0;
      if (!h_buildState)
        CUBQL_CUDA_CALL(MallocHost((void**)&h_buildState,
                                   sizeof(*h_buildState)));

      
      cudaEvent_t stateDownloadedEvent;
      CUBQL_CUDA_CALL(EventCreate(&stateDownloadedEvent));
      
      CUBQL_CUDA_CALL(MemcpyAsync(h_buildState,d_buildState,
                                  sizeof(*h_buildState),
                                  cudaMemcpyDeviceToHost,s));
      CUBQL_CUDA_CALL(EventRecord(stateDownloadedEvent,s));
      CUBQL_CUDA_CALL(EventSynchronize(stateDownloadedEvent));

      const int numValidPrims = h_buildState->numValidPrims;

      // ==================================================================
      // second MAJOR step: compute morton codes and primIDs array,
      // and do key/value sort to get those pairs sorted by ascending
      // morton code
      // ==================================================================
      /* 2.1, allocate mem for _unsorted_ prim IDs and morton codes,
       then compute initial primID array (will already exclude prims
       that are invalid) and (unsorted) morton code array */
      uint64_t *d_primKeys_unsorted;
      uint32_t *d_primIDs_unsorted;
      _ALLOC(d_primKeys_unsorted,numPrims,s,memResource);
      _ALLOC(d_primIDs_unsorted,numPrims,s,memResource);
      computeUnsortedKeysAndPrimIDs
        <<<divRoundUp(numValidPrims,1024),1024,0,s>>>
        (d_primKeys_unsorted,d_primIDs_unsorted,
         d_buildState,boxes,numPrims);

      /* 2.2: ask cub radix sorter for how much temp mem it needs, and
         allocate */
      size_t cub_tempMemSize;
      uint64_t *d_primKeys_sorted = 0;
      uint32_t *d_primIDs_inMortonOrder = 0;
      // with tempMem ptr null this won't do anything but return reqd
      // temp size*/
      cub::DeviceRadixSort::SortPairs
        (nullptr,cub_tempMemSize,
         /*keys in:*/   d_primKeys_unsorted,
         /*keys out:*/  d_primKeys_sorted,
         /*values in:*/ d_primIDs_unsorted,
         /*values out:*/d_primIDs_inMortonOrder,
         numValidPrims,0,64,s);
      
      // 2.3: allocate temp mem and output arrays
      void     *d_tempMem = 0;
      memResource.malloc(&d_tempMem,cub_tempMemSize,s);
      _ALLOC(d_primKeys_sorted,numValidPrims,s,memResource);
      _ALLOC(d_primIDs_inMortonOrder,numValidPrims,s,memResource);

      // 2.4: sort
      cub::DeviceRadixSort::SortPairs
        (d_tempMem,cub_tempMemSize,
         /*keys in:*/   d_primKeys_unsorted,
         /*keys out:*/  d_primKeys_sorted,
         /*values in:*/ d_primIDs_unsorted,
         /*values out:*/d_primIDs_inMortonOrder,
         numValidPrims,0,64,s);

      // 2.5 - cleanup after sort: no longer need tempmem, or unsorted inputs
      _FREE(d_primKeys_unsorted,s,memResource);
      _FREE(d_primIDs_unsorted,s,memResource);
      _FREE(d_tempMem,s,memResource);

      // ==================================================================
      // third MAJOR step: create temp-nodes from keys
      // ==================================================================
      /* 3.1: allocate nodes array (do this only onw so we can re-use
         just freed memory); and initialize node 0 to span entire
         range of prims */
      uint32_t upperBoundOnNumNodesToBeCreated = 2*numValidPrims;
      TempNode *nodes = 0;
      _ALLOC(nodes,upperBoundOnNumNodesToBeCreated,s,memResource);
      initNodes<<<32,1,0,s>>>(d_buildState,nodes,numValidPrims);

      /* 3.2 extract nodes until no more (temp-)nodes get created */
      int numNodesAlloced = 1; /*!< device actually things it's two,
                                  but we intentionally use 1 here to
                                  make first round start with right
                                  could of _valid_ nodes*/
      
      int numNodesDone    = 0;
      while (numNodesDone < numNodesAlloced) {
        int numNodesStillToDo = numNodesAlloced - numNodesDone;
        createNodes<<<divRoundUp(numNodesStillToDo,1024),1024,0,s>>>
          (d_buildState,makeLeafThreshold,
           nodes,numNodesDone,numNodesAlloced,
           d_primKeys_sorted);
        CUBQL_CUDA_CALL(MemcpyAsync(h_buildState,d_buildState,sizeof(*h_buildState),
                                    cudaMemcpyDeviceToHost,s));
        CUBQL_CUDA_CALL(EventRecord(stateDownloadedEvent,s));
        CUBQL_CUDA_CALL(EventSynchronize(stateDownloadedEvent));
        
        numNodesDone = numNodesAlloced;
        numNodesAlloced = h_buildState->numNodesAlloced;
      }
      
      // ==================================================================
      // step four: create actual ndoes - we now know how many, and
      // what they point to; let's just fillin topology and let refit
      // fill in the boxes later on
      // ==================================================================
      /* 4.1 - free keys, we no longer need them. */
      _FREE(d_primKeys_sorted,s,memResource);
      /* 4.2 - save morton-ordered prims in bvh - that's where the
         final nodes will be pointing into, so they are our primID
         array. */
      bvh.primIDs = d_primIDs_inMortonOrder;
      bvh.numPrims = numValidPrims;

      /* 4.3 alloc 'final' nodes; we now know exactly how many we
         have */
      bvh.numNodes = numNodesAlloced;
      _ALLOC(bvh.nodes,numNodesAlloced,s,memResource);
      writeFinalNodes<T,D>
        <<<divRoundUp(numNodesAlloced,1024),1024,0,s>>>
        (bvh.nodes,nodes,numNodesAlloced);
      
      /* 4.4 cleanup - free temp nodes, free build state, and release event */
      CUBQL_CUDA_CALL(EventDestroy(stateDownloadedEvent));
      _FREE(nodes,s,memResource);
      _FREE(d_buildState,s,memResource);

      // ==================================================================
      // done. all we need to do now is refit the bboxes
      // ==================================================================
      gpuBuilder_impl::refit(bvh,boxes,s,memResource);
    }
  }

  template<typename T, int D>
  void mortonBuilder(BinaryBVH<T,D>   &bvh,
                     const box_t<T,D> *boxes,
                     int                   numPrims,
                     BuildConfig           buildConfig,
                     cudaStream_t          s,
                     GpuMemoryResource    &memResource)
  { mortonBuilder_impl::build(bvh,boxes,numPrims,buildConfig,s,memResource); }
}

