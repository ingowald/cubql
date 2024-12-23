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
 
#pragma once

#include "cuBQL/bvh.h"
#include <cuBQL/math/vec.h>
#include <cuBQL/math/box.h>
#include <cuBQL/math/affine.h>
#include <cuBQL/math/conservativeDistances.h>

namespace cuBQL {
  namespace shrinkingRadiusQuery {
    // ******************************************************************
    // INTERFACE
    // (which functions this header file provides)
    // ******************************************************************
    
    // ******************************************************************
    /*! performs a 'shrinking radius (primitive-)query', which iterate
      through all bvh leaves that overlap a given query ball that is
      centered around a fixed point in space, and whose radius may by
      successively reduced (shrunk) durin that query. Every time the
      query reaches a new candidate leaf it calls the provided
      callback function, which, after processing the given leaf, can
      then return a new maximum query radius which, if smaller than
      the radius of the query ball at that point in time, will from
      that point on be used as new query radius. Note that the query
      radius can only be *shrunk* during traversal; if the
      user-provided callback returns a radius larger than what the
      query ball has already been shrunk to the existing smaller value
      will be used.

      Expected signature of the lambda is

      float perPrimLambda(uint32_t primID);

      which is expected to return a new maximum search radius (or a
      value larger than the current search radius, in which case this
      value gets ignored)
    */
    template<typename T, int D, typename Lambda>
    inline __cubql_both
    void forEachPrim(/*! lambda that gets called for each candidate
                       primitmive index that may contain any
                       primitmives. if this lamdba does find a new,
                       better result than whatever the query had
                       before this lambda MUST return the SQUARE of
                       the new culling radius */
                     const Lambda &lambdaToExecuteForEachCandidate,
                     /* the bvh we're querying into */
                     bvh_t<T,D> bvh,
                     /*! the center of out query ball */
                     vec_t<T,D> queryPoint,
                     /*! the SQUARE of the maximum query radius to
                       which we want to restrict the search; can be
                       CUBQL_INF for unrestricted searches */
                     float sqrMaxSearchRadius=CUBQL_INF);
    
    // ******************************************************************
    /*! performs a 'shrinking radius (primitive-)query', which iterate
      through all bvh leaves that overlap a given query ball that is
      centered around a fixed point in space, and whose radius may by
      successively reduced (shrunk) durin that query. Every time the
      query reaches a new candidate leaf it calls the provided
      callback function, which, after processing the given leaf, can
      then return a new maximum query radius which, if smaller than
      the radius of the query ball at that point in time, will from
      that point on be used as new query radius. Note that the query
      radius can only be *shrunk* during traversal; if the
      user-provided callback returns a radius larger than what the
      query ball has already been shrunk to the existing smaller value
      will be used.

      Expected signature of the lambda is

      float perPrimLambda(uint32_t primID);

      which is expected to return a new maximum search radius (or a
      value larger than the current search radius, in which case this
      value gets ignored)
    */
    template<typename T, int D, int W, typename Lambda>
    inline __cubql_both
    void forEachPrim(/*! lambda that gets called for each candidate
                       primitmive index that may contain any
                       primitmives. if this lamdba does find a new,
                       better result than whatever the query had
                       before this lambda MUST return the SQUARE of
                       the new culling radius */
                     const Lambda &lambdaToExecuteForEachCandidate,
                     /* the bvh we're querying into */
                     WideBVH<T,D,W> bvh,
                     /*! the center of out query ball */
                     vec_t<T,D> queryPoint,
                     /*! the SQUARE of the maximum query radius to
                       which we want to restrict the search; can be
                       CUBQL_INF for unrestricted searches */
                     float sqrMaxSearchRadius=CUBQL_INF);

    /*! same as regular `forEachLeaf()` function, except this one has
        a lambda for both leaf-test *and* for distance computation as
        well. The "NodeLambda" gets passed a pointer to a node, and is
        supposed to return a float (for conservative distance to said
        node). If the returned distnace is INF that node is guaranteed
        to not be traversed) */
    template<typename T, int D, typename PrimLambda, typename NodeLambda>
    inline __cubql_both
    void forEachPrim(/*! lambda that gets called for each prim that
                       may contain any primitives. if this lamdba does
                       find a new, better result than whatever the
                       query had before this lambda MUST return the
                       SQUARE of the new culling radius */
                     const PrimLambda &lambdaToExecuteForEachCandidatePrim,
                     const NodeLambda &lambdaToExecuteForEachVisitedNode,
                     /* the bvh we're querying into */
                     bvh_t<T,D> bvh,
                     /*! the SQUARE of the maximum query radius to
                       which we want to restrict the search; can be
                       CUBQL_INF for unrestricted searches */
                     float sqrMaxSearchRadius=CUBQL_INF);
    

    // ******************************************************************
    /*! performs a 'shrinking radius (leaf-)query', which iterate
      through all bvh leaves that overlap a given query ball that is
      centered around a fixed point in space, and whose radius may by
      successively reduced (shrunk) durin that query. Every time the
      query reaches a new candidate leaf it calls the provided
      callback function, which, after processing the given leaf, can
      then return a new maximum query radius which, if smaller than
      the radius of the query ball at that point in time, will from
      that point on be used as new query radius. Note that the query
      radius can only be *shrunk* during traversal; if the
      user-provided callback returns a radius larger than what the
      query ball has already been shrunk to the existing smaller value
      will be used.

      Expected signature of the lambda is

      float perLeafLambda(const uint32_t *primIDsInLeaf,
                          int numPrimsInLeaf);

      which is expected to return a new maximum search radius (or a
      value larger than the current search radius, in which case this
      value gets ignored)
    */
    template<typename T, int D, typename Lambda>
    inline __cubql_both
    void forEachLeaf(/*! lambda that gets called for each leaf that
                       may contain any primitives. if this lamdba does
                       find a new, better result than whatever the
                       query had before this lambda MUST return the
                       SQUARE of the new culling radius */
                     const Lambda &lambdaToExecuteForEachCandidateLeaf,
                     /* the bvh we're querying into */
                     bvh_t<T,D> bvh,
                     /*! the center of out query ball */
                     vec_t<T,D> queryPoint,
                     /*! the SQUARE of the maximum query radius to
                       which we want to restrict the search; can be
                       CUBQL_INF for unrestricted searches */
                     float sqrMaxSearchRadius=CUBQL_INF);
    
    /*! performs a 'shrinking radius (leaf-)query', which iterate
      through all bvh leaves that overlap a given query ball that is
      centered around a fixed point in space, and whose radius may by
      successively reduced (shrunk) durin that query. Every time the
      query reaches a new candidate leaf it calls the provided
      callback function, which, after processing the given leaf, can
      then return a new maximum query radius which, if smaller than
      the radius of the query ball at that point in time, will from
      that point on be used as new query radius. Note that the query
      radius can only be *shrunk* during traversal; if the
      user-provided callback returns a radius larger than what the
      query ball has already been shrunk to the existing smaller value
      will be used. */
    template<typename T, int D, int W, typename Lambda>
    inline __cubql_both
    void forEachLeaf(/*! lambda that gets called for each leaf that
                       may contain any primitives. if this lamdba does
                       find a new, better result than whatever the
                       query had before this lambda MUST return the
                       SQUARE of the new culling radius */
                     const Lambda &lambdaToExecuteForEachCandidateLeaf,
                     /* the bvh we're querying into */
                     WideBVH<T,D,W> bvh,
                     /*! the center of out query ball */
                     vec_t<T,D> queryPoint,
                     /*! the SQUARE of the maximum query radius to
                       which we want to restrict the search; can be
                       CUBQL_INF for unrestricted searches */
                     float sqrMaxSearchRadius=CUBQL_INF);

    /*! same as regular `forEachLeaf()` function, except this one has
        a lambda for both leaf-test *and* for distance computation as
        well. The "NodeLambda" gets passed a pointer to a node, and is
        supposed to return a float (for conservative distance to said
        node). If the returned distnace is INF that node is guaranteed
        to not be traversed) */
    template<typename T, int D, typename LeafLambda, typename NodeLambda>
    inline __cubql_both
    void forEachLeaf(/*! lambda that gets called for each leaf that
                       may contain any primitives. if this lamdba does
                       find a new, better result than whatever the
                       query had before this lambda MUST return the
                       SQUARE of the new culling radius */
                     const LeafLambda &lambdaToExecuteForEachCandidateLeaf,
                     const NodeLambda &lambdaToExecuteForEachVisitedNode,
                     /* the bvh we're querying into */
                     bvh_t<T,D> bvh,
                     /*! the SQUARE of the maximum query radius to
                       which we want to restrict the search; can be
                       CUBQL_INF for unrestricted searches */
                     float sqrMaxSearchRadius=CUBQL_INF);
    
    
  
    // ******************************************************************
    // IMPLEMENTATION
    // ******************************************************************
    
    /*! compute square distance with round-down where necessary. Use
      this in traversal query-to-node distance checks, because the
      rounding down will mean we'll only ever, if anything,
      _under_estimate the distance to the node, and thus never wrongly
      cull any subtrees) */
    inline __cubql_both
    float fSqrDistance_rd(vec2f point, box2f box)
    {
      vec2f projected = min(max(point,box.lower),box.upper);
      vec2f v = projected - point;
      return dot(v,v);
    }
    /*! compute square distance with round-down where necessary. Use
      this in traversal query-to-node distance checks, because the
      rounding down will mean we'll only ever, if anything,
      _under_estimate the distance to the node, and thus never wrongly
      cull any subtrees) */
    inline __cubql_both
    float fSqrDistance_rd(vec3f point, box3f box)
    {
      vec3f projected = min(max(point,box.lower),box.upper);
      vec3f v = projected - point;
      return dot(v,v);
    }
    /*! compute square distance with round-down where necessary. Use
      this in traversal query-to-node distance checks, because the
      rounding down will mean we'll only ever, if anything,
      _under_estimate the distance to the node, and thus never wrongly
      cull any subtrees) */
    inline __cubql_both
    float fSqrDistance_rd(vec4f point, box4f box)
    {
      vec4f projected = min(max(point,box.lower),box.upper);
      vec4f v = projected - point;
      return dot(v,v);
    }


    /*! compute square distance with round-down where necessary. Use
      this in traversal query-to-node distance checks, because the
      rounding down will mean we'll only ever, if anything,
      _under_estimate the distance to the node, and thus never wrongly
      cull any subtrees) */
    inline __cubql_both
    float fSqrDistance_rd(vec2i point, box2i box)
    {
      vec2i projected = min(max(point,box.lower),box.upper);
      vec2i v = projected - point;
#ifdef __CUDA_ARCH__
      return __ll2float_rd(dot(v,v));
#else
      return host::__ull2float_rd(dot(v,v));
#endif
    }
    /*! compute square distance with round-down where necessary. Use
      this in traversal query-to-node distance checks, because the
      rounding down will mean we'll only ever, if anything,
      _under_estimate the distance to the node, and thus never wrongly
      cull any subtrees) */
    inline __cubql_both
    float fSqrDistance_rd(vec3i point, box3i box)
    {
      vec3i projected = min(max(point,box.lower),box.upper);
      vec3i v = projected - point;
#ifdef __CUDA_ARCH__
      return __ll2float_rd(dot(v,v));
#else
      return host::__ull2float_rd(dot(v,v));
#endif
    }
    /*! compute square distance with round-down where necessary. Use
      this in traversal query-to-node distance checks, because the
      rounding down will mean we'll only ever, if anything,
      _under_estimate the distance to the node, and thus never wrongly
      cull any subtrees) */
    inline __cubql_both
    float fSqrDistance_rd(vec4i point, box4i box)
    {
      vec4i projected = min(max(point,box.lower),box.upper);
      vec4i v = projected - point;
#ifdef __CUDA_ARCH__
      return __ll2float_rd(dot(v,v));
#else
      return host::__ull2float_rd(dot(v,v));
#endif
    }



    /*! compute square distance with round-down where necessary. Use
      this in traversal query-to-node distance checks, because the
      rounding down will mean we'll only ever, if anything,
      _under_estimate the distance to the node, and thus never wrongly
      cull any subtrees) */
    inline __cubql_both
    float fSqrDistance_rd(vec2d point, box2d box)
    {
      vec2d projected = min(max(point,box.lower),box.upper);
      vec2d v = projected - point;
#ifdef __CUDA_ARCH__
      return __ll2float_rd(dot(v,v));
#else
      return host::__udouble2float_rd(dot(v,v));
#endif
    }
    
    /*! compute square distance with round-down where necessary. Use
      this in traversal query-to-node distance checks, because the
      rounding down will mean we'll only ever, if anything,
      _under_estimate the distance to the node, and thus never wrongly
      cull any subtrees) */
    inline __cubql_both
    float fSqrDistance_rd(vec3d point, box3d box)
    {
      vec3d projected = min(max(point,box.lower),box.upper);
      vec3d v = projected - point;
#ifdef __CUDA_ARCH__
      return __double2float_rd(dot(v,v));
#else
      return host::__udouble2float_rd(dot(v,v));
#endif
    }
    
    /*! compute square distance with round-down where necessary. Use
      this in traversal query-to-node distance checks, because the
      rounding down will mean we'll only ever, if anything,
      _under_estimate the distance to the node, and thus never wrongly
      cull any subtrees) */
    inline __cubql_both
    float fSqrDistance_rd(vec4d point, box4d box)
    {
      vec4d projected = min(max(point,box.lower),box.upper);
      vec4d v = projected - point;
#ifdef __CUDA_ARCH__
      return __double2float_rd(dot(v,v));
#else
      return host::__udouble2float_rd(dot(v,v));
#endif
    }

    template<int D>
    inline __cubql_both
    float fSqrDistance_rd(vec_t<long long int, D> a,
                          vec_t<long long int, D> b)
    {
      float sum = 0.f;
      for (int i=0;i<D;i++) {
        long long lo = min(a[i],b[i]);
        long long hi = max(a[i],b[i]);
        unsigned long long diff = hi - lo;
#ifdef __CUDA_ARCH__
        float fDiff = __ll2float_rd(diff);
#else
        float fDiff = host::__ull2float_rd(diff);
#endif
        sum += fDiff*fDiff;
      }
      return sum;
    }
    
    /*! performs a 'shrinking radius (leaf-)query', which iterate
      through all bvh leaves that overlap a given query ball that is
      centered around a fixed point in space, and whose radius may by
      successively reduced (shrunk) durin that query. Every time the
      query reaches a new candidate leaf it calls the provided
      callback function, which, after processing the given leaf, can
      then return a new maximum query radius which, if smaller than
      the radius of the query ball at that point in time, will from
      that point on be used as new query radius. Note that the query
      radius can only be *shrunk* during traversal; if the
      user-provided callback returns a radius larger than what the
      query ball has already been shrunk to the existing smaller value
      will be used. */
    template<typename T, int D, typename Lambda>
    inline __cubql_both
    void forEachLeaf(/*! lambda that gets called for each leaf that
                       may contain any primitives. if this lamdba does
                       find a new, better result than whatever the
                       query had before this lambda MUST return the
                       SQUARE of the new culling radius */
                     const Lambda &lambdaToExecuteForEachCandidateLeaf,
                     bvh_t<T,D> bvh,
                     vec_t<T,D> queryPoint,
                     float sqrMaxSearchRadius)
    {
      float sqrCullDist = sqrMaxSearchRadius;
      struct StackEntry {
        union {
          struct {
            uint32_t idx;
            float    dist;
          };
          uint64_t forALign;
        };
      };
      const int stackSize = 64;
      StackEntry traversalStack[stackSize], *stackPtr = traversalStack;
      typename bvh_t<T,D>::node_t::Admin node = bvh.nodes[0].admin;
      // ------------------------------------------------------------------
      // traverse until there's nothing left to traverse:
      // ------------------------------------------------------------------
      while (true) {

        // ------------------------------------------------------------------
        // traverse INNER nodes downward; breaking out if we either find
        // a leaf within the current search radius, or found a dead-end
        // at which we need to pop
        // ------------------------------------------------------------------
        while (true) {
          if (node.count != 0)
            // it's a boy! - seriously: this is not a inner node, step
            // out of down-travesal and let leaf code pop in.
            break;

          uint32_t n0Idx = (uint32_t)node.offset+0;
          uint32_t n1Idx = (uint32_t)node.offset+1;
          typename bvh_t<T,D>::node_t n0 = bvh.nodes[n0Idx];
          typename bvh_t<T,D>::node_t n1 = bvh.nodes[n1Idx];
          float d0 = fSqrDistance_rd(queryPoint,n0.bounds);
          float d1 = fSqrDistance_rd(queryPoint,n1.bounds);
          if (min(d0,d1) >= sqrCullDist) {
            // both children are too far away; this is a dead end
            node.count = 0;
            break;
          }

          uint32_t farID;
          if (d0 < d1) {
            // go left side, possibly pop right side
            node = n0.admin;
            farID = n1Idx;
          } else {
            // go left side, possibly pop right side
            node = n1.admin;
            farID = n0Idx;
          }
          float farChildDist = max(d0,d1);
          bool  bothInRange = (farChildDist < sqrCullDist);
          if (bothInRange) {
            if (stackPtr >= traversalStack+stackSize) {
              // printf("stack overflow\n");
              return;
            }
            *stackPtr++ = StackEntry{ farID, farChildDist }; }
        }
      
        if (node.count != 0) {
          // we're at a valid leaf: call the lambda and see if that gave
          // us a enw, closer cull radius
          float leafResult
            = lambdaToExecuteForEachCandidateLeaf(bvh.primIDs+node.offset,node.count);
          if (leafResult < 0.f) return;
          sqrCullDist = min(sqrCullDist,leafResult);
        }
        // ------------------------------------------------------------------
        // pop next un-traversed node from stack, discarding any nodes
        // that are more distant than whatever query radius we now have
        // ------------------------------------------------------------------
        while (true) {
          if (stackPtr == traversalStack)
            return;
          StackEntry fromStack = *--stackPtr;
          if (fromStack.dist <= sqrCullDist) {
            node = bvh.nodes[fromStack.idx].admin;
            // found one!
            break;
          }
          // discard this one, it's too far away (apparently the search
          // radius has shrunk since we pushed this node
          continue;
        }
      }
    }

#ifdef __CUDA_ARCH__
#else
    inline float __int_as_float(int i) { return (const float &)i; }
    inline int __float_as_int(float f) { return (const int &)f; }
#endif
    
    template<int N>
    struct ChildOrder {
      inline __cubql_both void clear(int i) { v[i] = (uint64_t)-1; }
      inline __cubql_both void set(int i, float dist, uint32_t payload)
      { v[i] = (uint64_t(__float_as_int(dist))<<32) | payload; }
      
      uint64_t v[N];
    };
    
    template<int N>
    inline __cubql_both void sort(ChildOrder<N> &children)
    {
#pragma unroll
      for (int i=N-1;i>0;--i) {
#pragma unroll
        for (int j=0;j<i;j++) {
          uint64_t c0 = children.v[j+0];
          uint64_t c1 = children.v[j+1];
          children.v[j+0] = min(c0,c1);
          children.v[j+1] = max(c0,c1);
        }
      }
    }

    /*! performs a 'shrinking radius (leaf-)query', which iterate
      through all bvh leaves that overlap a given query ball that is
      centered around a fixed point in space, and whose radius may by
      successively reduced (shrunk) durin that query. Every time the
      query reaches a new candidate leaf it calls the provided
      callback function, which, after processing the given leaf, can
      then return a new maximum query radius which, if smaller than
      the radius of the query ball at that point in time, will from
      that point on be used as new query radius. Note that the query
      radius can only be *shrunk* during traversal; if the
      user-provided callback returns a radius larger than what the
      query ball has already been shrunk to the existing smaller value
      will be used. */
    template<typename T, int D, int W, typename Lambda>
    inline __cubql_both
    void forEachLeaf(/*! lambda that gets called for each leaf that
                       may contain any primitives. if this lamdba does
                       find a new, better result than whatever the
                       query had before this lambda MUST return the
                       SQUARE of the new culling radius */
                     const Lambda &lambdaToExecuteForEachCandidateLeaf,
                     WideBVH<T,D,W> bvh,
                     vec_t<T,D> queryPoint,
                     float /* SQUARE OF maxmimum search radius */maxQueryDistSquare)
    {
      enum { stackSize = 64 };
      uint64_t stackBase[stackSize], *stackPtr = stackBase;
      int nodeID = 0;
      ChildOrder<W> childOrder;
      while (true) {
        while (true) {
          while (nodeID == -1) {
            if (stackPtr == stackBase)
              return;
            uint64_t tos = *--stackPtr;
            if (__int_as_float(tos>>32) > maxQueryDistSquare)
              continue;
            nodeID = (uint32_t)tos;
            // pop....
          }
          if (nodeID & (1<<31))
            break;
          
          const typename WideBVH<float,3,W>::Node &node = bvh.nodes[nodeID];
#pragma unroll
          for (int c=0;c<W;c++) {
            const auto child = node.children[c];
            if (!node.children[c].valid)
              childOrder.clear(c);
            else {
              float dist2 = sqrDistance(child.bounds,queryPoint);
              if (dist2 > maxQueryDistSquare) 
                childOrder.clear(c);
              else {
                uint32_t payload
                  = child.count
                  ? ((1<<31)|(nodeID<<log_of<W>::value)|c)
                  : child.offset;
                childOrder.set(c,dist2,payload);
              }
            }
          }
          sort(childOrder);
#pragma unroll
          for (int c=W-1;c>0;--c) {
            uint64_t coc = childOrder.v[c];
            if (coc != uint64_t(-1)) {
              *stackPtr++ = coc;
              // if (stackPtr - stackBase == stackSize)
              //   printf("stack overrun!\n");
            }
          }
          if (childOrder.v[0] == uint64_t(-1)) {
            nodeID = -1;
            continue;
          }
          nodeID = uint32_t(childOrder.v[0]);
        }
      
        int c = nodeID & ((1<<log_of<W>::value)-1);
        int n = (nodeID & 0x7fffffff)  >> log_of<W>::value;
        int offset = bvh.nodes[n].children[c].offset;
        int count  = bvh.nodes[n].children[c].count;

        float maxSquareRadiusFromLeaf
          = lambdaToExecuteForEachCandidateLeaf(bvh.primIDs+offset,count);
        maxQueryDistSquare = min(maxQueryDistSquare,maxSquareRadiusFromLeaf);
        // for (int i=0;i<count;i++) {
        //   int primID = bvh.primIDs[offset+i];
        //   float dist2 = sqrDistance(dataPoints[primID],query);
        //   if (dist2 >= maxQueryDistSquare) continue;
        //   maxQueryDistSquare = dist2;
        //   result             = primID;
        // }
        nodeID = -1;
      }
    }


    
    /*! performs a 'shrinking radius (leaf-)query', which iterate
      through all bvh leaves that overlap a given query ball that is
      centered around a fixed point in space, and whose radius may by
      successively reduced (shrunk) durin that query. Every time the
      query reaches a new candidate leaf it calls the provided
      callback function, which, after processing the given leaf, can
      then return a new maximum query radius which, if smaller than
      the radius of the query ball at that point in time, will from
      that point on be used as new query radius. Note that the query
      radius can only be *shrunk* during traversal; if the
      user-provided callback returns a radius larger than what the
      query ball has already been shrunk to the existing smaller value
      will be used. */
    template<typename T, int D, typename LeafLambda, typename NodeLambda>
    inline __cubql_both
    void forEachLeaf(/*! lambda that gets called for each leaf that may contain any
                       primitives. if this lamdba does find a new, better result than
                       whatever the query had before this lambda MUST return the SQUARE
                       of the new culling radius */
                     const LeafLambda &lambdaToExecuteForEachCandidateLeaf,
                     const NodeLambda &lambdaToExecuteForEachCandidateNode,
                     bvh_t<T,D> bvh,
                     float sqrMaxSearchRadius)
    {
      float sqrCullDist = sqrMaxSearchRadius;
      struct StackEntry {
        union {
          struct {
            uint32_t idx;
            float    dist;
          };
          uint64_t forALign;
        };
      };
      const int stackSize = 64;
      StackEntry traversalStack[stackSize], *stackPtr = traversalStack;
      typename bvh_t<T,D>::node_t::Admin node = bvh.nodes[0].admin;
      // ------------------------------------------------------------------
      // traverse until there's nothing left to traverse:
      // ------------------------------------------------------------------
      while (true) {

        // ------------------------------------------------------------------
        // traverse INNER nodes downward; breaking out if we either find
        // a leaf within the current search radius, or found a dead-end
        // at which we need to pop
        // ------------------------------------------------------------------
        while (true) {
          if (node.count != 0)
            // it's a boy! - seriously: this is not a inner node, step
            // out of down-travesal and let leaf code pop in.
            break;

          uint32_t n0Idx = (uint32_t)node.offset+0;
          uint32_t n1Idx = (uint32_t)node.offset+1;
          typename bvh_t<T,D>::node_t &n0 = bvh.nodes[n0Idx];
          typename bvh_t<T,D>::node_t &n1 = bvh.nodes[n1Idx];
          float d0 = lambdaToExecuteForEachCandidateNode(n0);
          float d1 = lambdaToExecuteForEachCandidateNode(n1);
          if (min(d0,d1) >= sqrCullDist) {
            // both children are too far away; this is a dead end
            node.count = 0;
            break;
          }

          uint32_t farID;
          if (d0 < d1) {
            // go left side, possibly pop right side
            node = n0.admin;
            farID = n1Idx;
          } else {
            // go left side, possibly pop right side
            node = n1.admin;
            farID = n0Idx;
          }
          float farChildDist = max(d0,d1);
          bool  bothInRange = (farChildDist < sqrCullDist);
          if (bothInRange) {
            if (stackPtr >= traversalStack+stackSize) {
              // printf("stack overflow\n");
              return;
            }
            *stackPtr++ = StackEntry{ farID, farChildDist }; }
        }
      
        if (node.count != 0) {
          // we're at a valid leaf: call the lambda and see if that gave
          // us a enw, closer cull radius
          float leafResult
            = lambdaToExecuteForEachCandidateLeaf(bvh.primIDs+node.offset,node.count);
          if (leafResult < 0.f) return;
          sqrCullDist = min(sqrCullDist,leafResult);
        }
        // ------------------------------------------------------------------
        // pop next un-traversed node from stack, discarding any nodes
        // that are more distant than whatever query radius we now have
        // ------------------------------------------------------------------
        while (true) {
          if (stackPtr == traversalStack)
            return;
          StackEntry fromStack = *--stackPtr;
          if (fromStack.dist <= sqrCullDist) {
            node = bvh.nodes[fromStack.idx].admin;
            // found one!
            break;
          }
          // discard this one, it's too far away (apparently the search
          // radius has shrunk since we pushed this node
          continue;
        }
      }
    }

    /*! performs a 'shrinking radius (primitive-)query', which iterate
      through all bvh leaves that overlap a given query ball that is
      centered around a fixed point in space, and whose radius may by
      successively reduced (shrunk) durin that query. Every time the
      query reaches a new candidate leaf it calls the provided
      callback function, which, after processing the given leaf, can
      then return a new maximum query radius which, if smaller than
      the radius of the query ball at that point in time, will from
      that point on be used as new query radius. Note that the query
      radius can only be *shrunk* during traversal; if the
      user-provided callback returns a radius larger than what the
      query ball has already been shrunk to the existing smaller value
      will be used. */
    template<typename T, int D, typename Lambda>
    inline __cubql_both
    void forEachPrim(/*! lambda that gets called for each candidate
                       primitmive index that may contain any
                       primitmives. if this lamdba does find a new,
                       better result than whatever the query had
                       before this lambda MUST return the SQUARE of
                       the new culling radius */
                     const Lambda &lambdaToExecuteForEachCandidate,
                     /* the bvh we're querying into */
                     bvh_t<T,D> bvh,
                     /*! the center of out query ball */
                     vec_t<T,D> queryPoint,
                     /*! the SQUARE of the maximum query radius to
                       which we want to restrict the search; can be
                       CUBQL_INF for unrestricted searches */
                     float sqrMaxSearchRadius)
    {
      /* the code we want to have executed for each leaf that may
         contain candidates. we loop over each prim in a given leaf,
         and return the minimum culling distance returned by any of
         the per-prim lambdas */
      auto leafCode
        = [lambdaToExecuteForEachCandidate](const uint32_t *leafPrims,
                                            size_t numPrims)->float
        {
          float leafResult = CUBQL_INF;
          for (int i=0;i<numPrims;i++) {
            float primResult
              = lambdaToExecuteForEachCandidate(leafPrims[i]);
            leafResult = min(leafResult,primResult);
          }
          return leafResult;
        };
      forEachLeaf(leafCode,bvh,queryPoint,sqrMaxSearchRadius);
    }

    /*! performs a 'shrinking radius (primitive-)query', which iterate
      through all bvh leaves that overlap a given query ball that is
      centered around a fixed point in space, and whose radius may by
      successively reduced (shrunk) durin that query. Every time the
      query reaches a new candidate leaf it calls the provided
      callback function, which, after processing the given leaf, can
      then return a new maximum query radius which, if smaller than
      the radius of the query ball at that point in time, will from
      that point on be used as new query radius. Note that the query
      radius can only be *shrunk* during traversal; if the
      user-provided callback returns a radius larger than what the
      query ball has already been shrunk to the existing smaller value
      will be used. */
    template<typename T, int D, int W, typename Lambda>
    inline __cubql_both
    void forEachPrim(/*! lambda that gets called for each candidate
                       primitmive index that may contain any
                       primitmives. if this lamdba does find a new,
                       better result than whatever the query had
                       before this lambda MUST return the SQUARE of
                       the new culling radius */
                     const Lambda &lambdaToExecuteForEachCandidate,
                     /* the bvh we're querying into */
                     WideBVH<T,D,W> bvh,
                     /*! the center of out query ball */
                     vec_t<T,D> queryPoint,
                     /*! the SQUARE of the maximum query radius to
                       which we want to restrict the search; can be
                       CUBQL_INF for unrestricted searches */
                     float sqrMaxSearchRadius)
    {
      /* the code we want to have executed for each leaf that may
         contain candidates. we loop over each prim in a given leaf,
         and return the minimum culling distance returned by any of
         the per-prim lambdas */
      auto leafCode
        = [lambdaToExecuteForEachCandidate](const uint32_t *leafPrims,
                                            size_t numPrims)->float
        {
          float leafResult = CUBQL_INF;
          for (int i=0;i<numPrims;i++) {
            float primResult
              = lambdaToExecuteForEachCandidate(leafPrims[i]);
            leafResult = min(leafResult,primResult);
          }
          return leafResult;
        };
      forEachLeaf(leafCode,bvh,queryPoint,sqrMaxSearchRadius);
    }

    /*! performs a 'shrinking radius (primitive-)query', which iterate
      through all bvh leaves that overlap a given query ball that is
      centered around a fixed point in space, and whose radius may by
      successively reduced (shrunk) durin that query. Every time the
      query reaches a new candidate leaf it calls the provided
      callback function, which, after processing the given leaf, can
      then return a new maximum query radius which, if smaller than
      the radius of the query ball at that point in time, will from
      that point on be used as new query radius. Note that the query
      radius can only be *shrunk* during traversal; if the
      user-provided callback returns a radius larger than what the
      query ball has already been shrunk to the existing smaller value
      will be used. */
    template<typename T, int D, typename PrimLambda, typename NodeLambda>
    inline __cubql_both
    void forEachPrim(/*! lambda that gets called for each candidate
                       primitmive index that may contain any
                       primitmives. if this lamdba does find a new,
                       better result than whatever the query had
                       before this lambda MUST return the SQUARE of
                       the new culling radius */
                     const PrimLambda &lambdaToExecuteForEachCandidate,
                     const NodeLambda &lambdaToExecuteForEachVisitedNode,
                     /* the bvh we're querying into */
                     bvh_t<T,D> bvh,
                     /*! the SQUARE of the maximum query radius to
                       which we want to restrict the search; can be
                       CUBQL_INF for unrestricted searches */
                     float sqrMaxSearchRadius)
    {
      /* the code we want to have executed for each leaf that may
         contain candidates. we loop over each prim in a given leaf,
         and return the minimum culling distance returned by any of
         the per-prim lambdas */
      auto leafCode
        = [lambdaToExecuteForEachCandidate](const uint32_t *leafPrims,
                                            size_t numPrims)->float
        {
          float leafResult = CUBQL_INF;
          for (int i=0;i<numPrims;i++) {
            float primResult
              = lambdaToExecuteForEachCandidate(leafPrims[i]);
            leafResult = min(leafResult,primResult);
          }
          return leafResult;
        };
      forEachLeaf(leafCode,lambdaToExecuteForEachVisitedNode,
                  bvh,sqrMaxSearchRadius);
    }
  
    
  } // ::cuBQL::shrinkingRadiusQuery
} // ::cuBQL 
