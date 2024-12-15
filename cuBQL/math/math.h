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

#include "cuBQL/math/common.h"

namespace cuBQL {
  
#ifdef __CUDACC__
  // make sure we use the built-in cuda functoins that use floats, not
  // the c-stdlib ones that use doubles.
  using ::min;
  using ::max;
#else
  using std::min;
  using std::max;
#endif
  
  template<int N> struct log_of;
  template<> struct log_of< 2> { enum { value = 1 }; };
  template<> struct log_of< 4> { enum { value = 2 }; };
  template<> struct log_of< 8> { enum { value = 3 }; };
  template<> struct log_of<16> { enum { value = 4 }; };
  template<> struct log_of<32> { enum { value = 5 }; };

  /*! square of a value */
  inline __cubql_both float sqr(float f) { return f*f; }
  
  /*! unary functors on scalar types, so we can lift them to vector types later on */
  inline __cubql_both float  rcp(float f)     { return 1.f/f; }
  inline __cubql_both double rcp(double d)    { return 1./d; }

  template<typename T>
  inline __cubql_both T clamp(T t, T lo=T(0), T hi=T(1))
  { return min(max(t,lo),hi); }

  inline __cubql_both float saturate(float f) { return clamp(f,0.f,1.f); }
  inline __cubql_both double saturate(double f) { return clamp(f,0.,1.); }

  // inline __cubql_both float sqrt(float f) { return ::sqrtf(f); }
  // inline __cubql_both double sqrt(double d) { return ::sqrt(d); }
}

