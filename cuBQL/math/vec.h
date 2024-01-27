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
#include <type_traits>

#ifdef _MSC_VER
# define CUBQL_PRAGMA_UNROLL /* nothing */
#else
# define CUBQL_PRAGMA_UNROLL _Pragma("unroll")
#endif

namespace cuBQL {

  template<typename /* scalar type */T, int /*! dimensoins */D>
  struct vec_t_data {
    inline __cubql_both T  operator[](int i) const { return v[i]; }
    inline __cubql_both T &operator[](int i)       { return v[i]; }
    T v[D];
  };

  /*! defines a "invalid" type to allow for using as a paramter where
    no "actual" type for something exists. e.g., a vec<float,4> has
    a cuda equivalent type of float4, but a vec<float,5> does not */
  struct invalid_t {};
  
  /*! defines the "cuda equivalent type" for a given vector type; i.e.,
    a vec3f=vec_t<float,3> has a equivalent cuda built-in type of
    float3. to also allow vec_t's that do not have a cuda
    equivalent, let's also create a 'invalid_t' to be used by
    default */ 
  template<typename T, int D> struct cuda_eq_t { using type = invalid_t; };
  template<> struct cuda_eq_t<float,2> { using type = float2; };
  template<> struct cuda_eq_t<float,3> { using type = float3; };
  template<> struct cuda_eq_t<float,4> { using type = float4; };
  template<> struct cuda_eq_t<int,2> { using type = int2; };
  template<> struct cuda_eq_t<int,3> { using type = int3; };
  template<> struct cuda_eq_t<int,4> { using type = int4; };
  template<> struct cuda_eq_t<double,2> { using type = double2; };
  template<> struct cuda_eq_t<double,3> { using type = double3; };
  template<> struct cuda_eq_t<double,4> { using type = double4; };

  template<typename T>
  struct vec_t_data<T,2> {
    using cuda_t = typename cuda_eq_t<T,2>::type;
    inline __cubql_both T  operator[](int i) const { return i?y:x; }
    inline __cubql_both T &operator[](int i)       { return i?y:x; }
    /*! auto-cast to equivalent cuda type */
    inline __cubql_both operator cuda_t() { cuda_t t; t.x = x; t.y = y; return t; }
    T x, y;
  };
  template<typename T>
  struct vec_t_data<T,3> {
    using cuda_t = typename cuda_eq_t<T,3>::type;
    inline __cubql_both T  operator[](int i) const { return (i==2)?z:(i?y:x); }
    inline __cubql_both T &operator[](int i)       { return (i==2)?z:(i?y:x); }
    /*! auto-cast to equivalent cuda type */
    inline __cubql_both operator cuda_t() { cuda_t t; t.x = x; t.y = y; return t; }
    T x, y, z;
  };
  template<typename T>
  struct vec_t_data<T,4> {
    using cuda_t = typename cuda_eq_t<T,4>::type;
    inline __cubql_both T  operator[](int i) const { return (i>=2)?(i==2?z:w):(i?y:x); }
    inline __cubql_both T &operator[](int i)       { return (i>=2)?(i==2?z:w):(i?y:x); }
    /*! auto-cast to equivalent cuda type */
    inline __cubql_both operator cuda_t() { cuda_t t; t.x = x; t.y = y; return t; }
    T x, y, z, w;
  };
  
  template<typename T, int D>
  struct vec_t : public vec_t_data<T,D> {
    enum { numDims = D };
    using scalar_t = T;
    using cuda_t = typename cuda_eq_t<T,D>::type;

    inline __cubql_both vec_t() {}
    inline __cubql_both vec_t(const T &t)
    {
      CUBQL_PRAGMA_UNROLL
        for (int i=0;i<D;i++) (*this)[i] = t;
    }
    inline __cubql_both vec_t(const vec_t_data<T,D> &o)
    {
      CUBQL_PRAGMA_UNROLL
        for (int i=0;i<D;i++) (*this)[i] = o[i];
    }
    inline __cubql_both vec_t(const cuda_t &o)
    {
      CUBQL_PRAGMA_UNROLL
        for (int i=0;i<D;i++) (*this)[i] = (&o.x)[i];
    }

    template<typename OT>
    explicit __cubql_both vec_t(const vec_t_data<OT,D> &o)
    {
      CUBQL_PRAGMA_UNROLL
        for (int i=0;i<D;i++) (*this)[i] = (T)o[i];
    }
    
    inline __cubql_both vec_t &operator=(cuda_t v)
    {
      CUBQL_PRAGMA_UNROLL
        for (int i=0;i<numDims;i++) (*this)[i] = (&v.x)[i]; 
      return *this;
    }
  };

  template<typename T>
  struct vec_t<T,2> : public vec_t_data<T,2> {
    enum { numDims = 2 };
    using scalar_t = T;
    using cuda_t = typename cuda_eq_t<T,2>::type;
    using vec_t_data<T,2>::x;
    using vec_t_data<T,2>::y;

    inline __cubql_both vec_t() {}
    inline __cubql_both vec_t(const T &t) { x = y = t; }
    inline __cubql_both vec_t(T x, T y)
    { this->x = x; this->y = y; }
    inline __cubql_both vec_t(const vec_t_data<T,2> &o)
    { this->x = (o.x); this->y = (o.y); }
    inline __cubql_both vec_t(const cuda_t &o) 
    { this->x = (o.x); this->y = (o.y); }

    template<typename OT>
    explicit __cubql_both vec_t(const vec_t_data<OT,2> &o)
    { this->x = (o.x); this->y = (o.y); }
    
    inline __cubql_both vec_t &operator=(cuda_t o)
    { this->x = (o.x); this->y = (o.y); }
  };
  
  template<typename T>
  struct vec_t<T,3> : public vec_t_data<T,3> {
    enum { numDims = 3 };
    using scalar_t = T;
    using cuda_t = typename cuda_eq_t<T,3>::type;
    using vec_t_data<T,3>::x;
    using vec_t_data<T,3>::y;
    using vec_t_data<T,3>::z;

    inline __cubql_both vec_t() {}
    inline __cubql_both vec_t(const T &t) { x = y = z = t; }
    inline __cubql_both vec_t(T x, T y, T z)
    { this->x = x; this->y = y; this->z = z; }
    inline __cubql_both vec_t(const vec_t_data<T,3> &o)
    { this->x = (o.x); this->y = (o.y); this->z = (o.z); }
    inline __cubql_both vec_t(const cuda_t &o) 
    { this->x = (o.x); this->y = (o.y); this->z = (o.z); }

    template<typename OT>
    explicit __cubql_both vec_t(const vec_t_data<OT,3> &o)
    { this->x = (o.x); this->y = (o.y); this->z = (o.z); }
    
    inline __cubql_both vec_t &operator=(cuda_t o)
    { this->x = (o.x); this->y = (o.y); this->z = (o.z); }
  };
  
  template<typename T>
  struct vec_t<T,4> : public vec_t_data<T,4> {
    enum { numDims = 4 };
    using scalar_t = T;
    using cuda_t = typename cuda_eq_t<T,4>::type;
    using vec_t_data<T,4>::x;
    using vec_t_data<T,4>::y;
    using vec_t_data<T,4>::z;
    using vec_t_data<T,4>::w;

    inline __cubql_both vec_t() {}
    inline __cubql_both vec_t(const T &t) { x = y = z = w = t; }
    inline __cubql_both vec_t(T x, T y, T z, T w)
    { this->x = x; this->y = y; this->z = z; this->w = w; }
    inline __cubql_both vec_t(const vec_t_data<T,4> &o)
    { this->x = (o.x); this->y = (o.y); this->z = (o.z); this->w = (o.w); }
    inline __cubql_both vec_t(const cuda_t &o) 
    { this->x = (o.x); this->y = (o.y); this->z = (o.z); this->w = o.w; }

    template<typename OT>
    explicit __cubql_both vec_t(const vec_t_data<OT,4> &o)
    { this->x = (o.x); this->y = (o.y); this->z = (o.z); this->w = o.w; }
    
    inline __cubql_both vec_t &operator=(cuda_t o)
    { this->x = (o.x); this->y = (o.y); this->z = (o.z); this->w = o.w; }
  };
  
  using vec2f = vec_t<float,2>;
  using vec3f = vec_t<float,3>;
  using vec4f = vec_t<float,4>;

  using vec2i = vec_t<int,2>;
  using vec3i = vec_t<int,3>;
  using vec4i = vec_t<int,4>;
  
  using vec2ui = vec_t<uint32_t,2>;
  using vec3ui = vec_t<uint32_t,3>;
  using vec4ui = vec_t<uint32_t,4>;
  
  template<typename T>
  inline __cubql_both vec_t<T,3> cross(vec_t<T,3> a, vec_t<T,3> b)
  {
    vec_t<T,3> v;
    v.x = a.y*b.z;
    v.y = a.z*b.x;
    v.z = a.x*b.y;
    return v;
  }

  /*! traits for a vec_t */
  template<typename vec_t>
  struct our_vec_t_traits {
    enum { numDims = vec_t::numDims };
    using scalar_t = typename vec_t::scalar_t;
  };


  /*! vec_traits<T> describe the scalar type and number of dimensions
    of whatever wants to get used as a vector/point type in
    cuBQL. By default cuBQL will use its own vec_t<T,D> for that,
    but this should allow to also describe the traits of external
    types such as CUDA's float3 */
  template<typename T> struct vec_traits : public our_vec_t_traits<T> {};
  
  template<> struct vec_traits<float3> { enum { numDims = 3 }; using scalar_t = float; };



  template<typename vec_t>
  inline __cubql_both vec_t make(typename vec_t::scalar_t v)
  {
    vec_t r;
    CUBQL_PRAGMA_UNROLL
      for (int i=0;i<vec_t::numDims;i++)
        r[i] = v;
    return r;
  }

  template<typename vec_t>
  inline __cubql_both vec_t make(typename cuda_eq_t<typename vec_t::scalar_t,vec_t::numDims>::type v)
  {
    vec_t r;
    CUBQL_PRAGMA_UNROLL
      for (int i=0;i<vec_t::numDims;i++)
        r[i] = (&v.x)[i];
    return r;
  }

#define CUBQL_OPERATOR(long_op, op)                                     \
  /* vec:vec */                                                         \
  template<typename T, int D>                                           \
  inline __cubql_both                                                   \
  vec_t<T,D> long_op(vec_t<T,D> a, vec_t<T,D> b)                        \
  {                                                                     \
    vec_t<T,D> r;                                                       \
    CUBQL_PRAGMA_UNROLL                                                 \
      for (int i=0;i<D;i++) r[i] = a[i] op b[i];                        \
    return r;                                                           \
  }                                                                     \
    /* scalar-vec */                                                    \
    template<typename T, int D>                                         \
    inline __cubql_both                                                 \
    vec_t<T,D> long_op(T a, vec_t<T,D> b)                               \
    {                                                                   \
      vec_t<T,D> r;                                                     \
      CUBQL_PRAGMA_UNROLL                                               \
        for (int i=0;i<D;i++) r[i] = a op b[i];                         \
      return r;                                                         \
    }                                                                   \
    /* vec:scalar */                                                    \
    template<typename T, int D>                                         \
    inline __cubql_both                                                 \
    vec_t<T,D> long_op(vec_t<T,D> a, T b)                               \
    {                                                                   \
      vec_t<T,D> r;                                                     \
      CUBQL_PRAGMA_UNROLL                                               \
        for (int i=0;i<D;i++) r[i] = a[i] op b;                         \
      return r;                                                         \
    }                                                                   \
    /* cudaVec:vec */                                                   \
    template<typename T, int D>                                         \
    inline __cubql_both                                                 \
    vec_t<T,D> long_op(typename cuda_eq_t<T,D>::type a, vec_t<T,D> b)   \
    {                                                                   \
      vec_t<T,D> r;                                                     \
      CUBQL_PRAGMA_UNROLL                                               \
        for (int i=0;i<D;i++) r[i] = (&a.x)[i] op b[i];                 \
      return r;                                                         \
    }                                                                   \
    /* vec:cudaVec */                                                   \
    template<typename T, int D>                                         \
    inline __cubql_both                                                 \
    vec_t<T,D> long_op(vec_t<T,D> a,  typename cuda_eq_t<T,D>::type b)  \
    {                                                                   \
      vec_t<T,D> r;                                                     \
      CUBQL_PRAGMA_UNROLL                                               \
        for (int i=0;i<D;i++) r[i] = a[i] op (&b.x)[i];                 \
      return r;                                                         \
    }                                                                   \

  CUBQL_OPERATOR(operator+,+)
  CUBQL_OPERATOR(operator-,-)
  CUBQL_OPERATOR(operator*,*)
  CUBQL_OPERATOR(operator/,/)
#undef CUBQL_OPERATOR

#define CUBQL_UNARY(op)                         \
  template<typename T, int D>                   \
  inline __cubql_both                           \
  vec_t<T,D> rcp(vec_t<T,D> a)                  \
  {                                             \
    vec_t<T,D> r;                               \
    CUBQL_PRAGMA_UNROLL                         \
      for (int i=0;i<D;i++) r[i] = op(a[i]);    \
    return r;                                   \
  }

  CUBQL_UNARY(rcp)
#undef CUBQL_FUNCTOR
  
#define CUBQL_BINARY(op)                                \
  template<typename T, int D>                           \
  inline __cubql_both                                   \
  vec_t<T,D> op(vec_t<T,D> a, vec_t<T,D> b)             \
  {                                                     \
    vec_t<T,D> r;                                       \
    CUBQL_PRAGMA_UNROLL                                 \
      for (int i=0;i<D;i++) r[i] = op(a[i],b[i]);       \
    return r;                                           \
  }

  CUBQL_BINARY(min)
  CUBQL_BINARY(max)
#undef CUBQL_FUNCTOR

  template<typename T> struct dot_result_t;
  template<> struct dot_result_t<float> { using type = float; };
  template<> struct dot_result_t<int32_t> { using type = int64_t; };
  template<> struct dot_result_t<double> { using type = double; };

  template<typename T, int D> inline __cubql_both
  typename dot_result_t<T>::type dot(vec_t<T,D> a, vec_t<T,D> b)
  {
    typename dot_result_t<T>::type result = 0;
    CUBQL_PRAGMA_UNROLL
      for (int i=0;i<D;i++)
        result += a[i]*b[i];
    return result;
  }

  /*! approximate-conservative square distance between two
    points. whatever type the points are, the result will be
    returned in floats, including whatever rounding error that might
    incur. we will, however, always round downwars, so if this is
    used for culling it will, if anything, under-estiamte the
    distance to a subtree (and thus, still traverse it) rather than
    wrongly skipping it*/
  template<typename T> inline __device__ float fSqrLength(T v);
  template<> inline __device__ float fSqrLength<float>(float v)
  { return v*v; }

#ifdef __CUDA_ARCH__
  template<> inline __device__ float fSqrLength<int>(int _v)
  { float v = __int2float_rz(_v); return v*v; }
#else
  template<> inline __device__ float fSqrLength<int>(int _v);
#endif

  /*! accurate square-length of a vector; due to the 'square' involved
    in computing the distance this may need to change the type from
    int to long, etc - so a bit less rounding issues, but a bit more
    complicated to use with the right typenames */
  template<typename T, int D> inline __cubql_both
  typename dot_result_t<T>::type sqrLength(vec_t<T,D> v)
  {
    return dot(v,v);
  }


  // ------------------------------------------------------------------
  // *square* distance between two points (can always be computed w/o
  // a square root, so makes sense even for non-float types)
  // ------------------------------------------------------------------

  /*! accurate square-distance between two points; due to the 'square'
    involved in computing the distance this may need to change the
    type from int to long, etc - so a bit less rounding issues, but
    a bit more complicated to use with the right typenames */
  template<typename T, int D> inline __cubql_both
  typename dot_result_t<T>::type sqrDistance(vec_t<T,D> a, vec_t<T,D> b)
  {
    return sqrLength(a-b);
  }

  /*! approximate-conservative square distance between two
    points. whatever type the points are, the result will be
    returned in floats, including whatever rounding error that might
    incur. we will, however, always round downwars, so if this is
    used for culling it will, if anything, under-estiamte the
    distance to a subtree (and thus, still traverse it) rather than
    wrongly skipping it*/
  template<typename T, int D> inline __cubql_both
  float fSqrDistance(vec_t<T,D> a, vec_t<T,D> b)
  {
    float sum = 0.f;
    CUBQL_PRAGMA_UNROLL
      for (int i=0;i<D;i++)
        sum += fSqrLength(a[i]-b[i]);
    return sum;
  }

  
  // ------------------------------------------------------------------
  // 'length' of a vector - may only make sense for certain types
  // ------------------------------------------------------------------
  template<int D>
  inline __cubql_both float length(const vec_t<float,D> &v)
  { return sqrtf(dot(v,v)); }
    
  
  template<typename T, int D>
  inline std::ostream &operator<<(std::ostream &o,
                                  const vec_t_data<T,D> &v)
  {
    o << "(";
    for (int i=0;i<D;i++) {
      if (i) o << ",";
      o << v[i];
    }
    o << ")";
    return o;
  }



  template<typename /* scalar type */T, int /*! dimensoins */D>
  inline __cubql_both bool operator==(const vec_t_data<T,D> &a,
                                      const vec_t_data<T,D> &b)
  {
#pragma unroll
    for (int i=0;i<D;i++)
      if (a[i] != b[i]) return false;
    return true;
  }
                                     

  template<typename T>
  inline __cubql_both
  T reduce_max(vec_t<T,2> v) { return max(v.x,v.y); }
    
  template<typename T>
  inline __cubql_both
  T reduce_min(vec_t<T,2> v) { return min(v.x,v.y); }
    

  template<typename T>
  inline __cubql_both
  T reduce_max(vec_t<T,3> v) { return max(max(v.x,v.y),v.z); }
    
  template<typename T>
  inline __cubql_both
  T reduce_min(vec_t<T,3> v) { return min(min(v.x,v.y),v.z); }
    

  template<typename T>
  inline __cubql_both
  T reduce_max(vec_t<T,4> v) { return max(max(v.x,v.y),max(v.z,v.w)); }
    
  template<typename T>
  inline __cubql_both
  T reduce_min(vec_t<T,4> v) { return min(min(v.x,v.y),min(v.z,v.w)); }
    

  inline __cubql_both
  vec2ui operator<<(vec2ui v, vec2ui b) {
    return {
      v.x << b.x,
      v.y << b.y
    };
  }
  inline __cubql_both
  vec3ui operator<<(vec3ui v, vec3ui b) {
    return {
      v.x << b.x,
      v.y << b.y,
      v.z << b.z
    };
  }
  inline __cubql_both
  vec4ui operator<<(vec4ui v, vec4ui b) {
    return {
      v.x << b.x,
      v.y << b.y,
      v.z << b.z,
      v.w << b.w
    };
  }

  inline __cubql_both
  vec2ui operator>>(vec2ui v, vec2ui b) {
    return {
      v.x >> b.x,
      v.y >> b.y
    };
  }
  inline __cubql_both
  vec3ui operator>>(vec3ui v, vec3ui b) {
    return {
      v.x >> b.x,
      v.y >> b.y,
      v.z >> b.z
    };
  }
  inline __cubql_both
  vec4ui operator>>(vec4ui v, vec4ui b) {
    return {
      v.x >> b.x,
      v.y >> b.y,
      v.z >> b.z,
      v.w >> b.w
    };
  }
}

