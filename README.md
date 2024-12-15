# cuBQL - A CUDA "BVH Build-and-Query" Library

CuBQL (say: "cubicle") is a (mostly) header-only CUDA/C++ library for the
easy and efficient GPU-construction and -traversal of bounding volume
hierarchies (BVHes), with the ultimate goal of providing the tools and
infrastructure to realize a wide range of (GPU-accelerated) spatial
queries over various geometric primitives.

CuBQL is largely inspired by two libraries: the standard template
library (`STL`), and `cub`. Like those two libraries cuBQL largely
relies on header-only CUDA/C++ code, and on the use of templates and
lambda functions to make sure that certain key operations (like
traversing a BVH) can work for different primititive, different data
type and dimensionality (e.g., `float3` vs `int2`), multiple different
but similar geometric queries (e.g., `find closest point` vs
`k-nearest neighbor (kNN)` vs `signed distance functions (SDF)`, etc).

Throughout cuBQL, the main driving goal are robustness, generality,
and ease of use: each builder for each BVH type should always work for
all input types and dimensionality, and even for numerically
challenging input data.

# cuBQL Functionality - Overview

CuBQL offers four separate layers of functionality:

- `Abstract BVH Type` layer: defines the basic (GPU friendly) type(s)
  for different kinds of BVHes. In particular, the cuBQL bvh types is
  templated over what geometric space the BVH is to be built over;
  i.e., you can realize not only BVHes over `float3` data, but also
  BVHes over `int4`, `double2`, etc (cuBQL spans the entire space of
  {int,float,double,long}x{2,3,4,N}).

- `BVH builders` layer: provides a set of primarily GPU-side (but also
  some simple host side) builder(s) for the underlying BVH
  type(s). This level offers multiple different builders with
  different speed/quality tradeoffs (though the default `gpuBuilder`
  should work well for most cases).
  
- `BVH Traversal Templates` layer: though different types of geometric
  queries are often *similar in concept*, nevertheless they often
  slightly *differ in detail*. Instead of only providing a fixed set
  of very specific geometric queries cuBQL focusses on providing a set
  of traversal *templates* that, though the use of lambda functions,
  can easily be modified in their details. E.g., both a kNN and a find
  closest point query will build on the same `shrinking radius query`,
  with just different way of processing a given candidate primitive
  encountered during traversal.
  
- `Various (specific) Geometric Queries`, realized with the underlying
   layers. cuBQL provides these queries more as *samples* than
   anything else, fully assuming that many users will have
   requirements that the existing samples will not capture---but which
   these samples's use of the traversal templates should show how to
   realize.

# Supported BVH Type(s)

The main BVH type of this library is a binary BVH, where each node
contains that node's bounding box, as well as two ints, `count` and
`offset`.

```
  template<typename /*ScalarType*/T, 
           int /*Dimensionality*/D>
  struct BinaryBVH {
    struct CUBQL_ALIGN(16) Node {
      box_t<T,D> bounds;
      uint64_t   offset : 48;
      uint64_t   count  : 16;
    };

    Node     *nodes;
    uint32_t  numNodes;
    uint32_t *primIDs;
    uint32_t  numPrims;
  };
```

The `count` value is 0 for inner nodes, and for leaf nodes specifies
the number of primitives in this leaf. For inner nodes, the `offset`
indexes into the `BinaryBVH::nodes[]` array (the current node's two
children are at `nodes[offset]`, and `nodes[offset+1]`, respectively);
for leaf nodes it points into the `BinaryBVH::primIDs[]` array (i.e.,
that leaf contains `primID[offset+0]`, `primID[offset+1]`, etc).

A `WideBVH<N>` type (templated over BVH width) is supported as
well. WideBVH'es always have a fixed branching factor of N (i.e., a
fixed number of `N` children in each inner node); however, some of
these may be 'null' (marked as not valid). Note that most builders
will only work for binary BVHes; these can then "collapsed" into
Wide-BVHes.

Though most of the algorithms and data types in this library could
absolutely be templated over both dimensionality and underlying data
type (i.e., a BVH over `double4` data rather than `float3`), for sake
of readability in this particular implementation this has not been
done (yet?). If this is a feature you would like to have, please let
me know.

# (on-GPU) BVH Construction

The main workhorse of this library is a CUDA-accelerated and `on
device` parallel BVH builder (with spatial median splits). The primary
feature of the BVH builder is its simplicity; i.e., it is still
"reasonably fast", but it is much simpler than other variants. Though
performance will obviously vary for different data types, data
distributions, etc..., right now this builder builds a BinaryBVH over
10 million uniformly distributed random points in under 13ms; that's
not the fastest builder I have, but IMHO quite reasonable for most
applications. In addition to this `cuBQL::gpuBuilder()` there are also
various other builders, including a regular morton/radix builder, a
wide GPU builder (for BVHes with branching factors greater than 2), a
surface-area-heuristic (SAH) builder, and a modified morton/radix
builder that for numerically challenging inputs is significantly more
robust than a regular morton/radix builder.

For all builders, the overall build process is always the same: Create
an array of bounding boxes (one box per primitive), and call the
builder with a pointer to this array, and the number of
primitives. For GPU-side builders this array has to live in device (or
managed) memory; for host side builds it has to be in host
memory. Obiously, device side builders will create node and primitmive
ID arrays in device memory, the host builder will create these in host
memory.

Given such an array, the builder (in this case, for `float3` data)
gets invoked as follows:

```
#include "cuBQL/bvh.h"
...
box3f *d_boxes  = 0;
int    numBoxes = 0;
userCodeForGeneratingPrims(&d_boxes,&numBoxes, ...);
...
cuBQL::BinaryBVH<float,3> bvh;
cuBQL::BuildConfig buildParams;
cuBQL::gpuBuilder(bvh,d_boxes,numBoxes,buildParams);
...
```
Builds for other data types (such as, e.g., `<int,4>` or <double,2>`)
work exactly the same way (though obviously, the scalar type and dimensionality of the
boxes has to be the same as that for the BVH).

The builder will not modify the `d_boxes[]` array; after the build is
complete the `bvh.primIDs[]` array contains ints referring to indices
in this array. This builder will properly handle "invalid prims" and
"empty boxes": Primitives that are not supposed to be included in the
BVH can simply use a box for which `lower.x > upper.x`; such
primitives will be detected during the build, and will simply get
excluded from the build process - i.e., they will simply not appear in
any of the leaves, but also not influence any of the (valid) bounding
boxes. However, behavior for NaNs, denorms, etc. is not
defined. Zero-volume primitives (ie, those with `box.lower ==
box.upper`) are considered valid primitives, and will get included in
the BVH.

The `BuildConfig` class can be used to influence things like whether
the BVH should be built with a surface area heuristic (SAH) cost
metric (more expensive build, but faster queries for some types of
inputs and query operations), or how coarse vs how fine the BVH should
be built (ie, at which point to make a leaf).

A few notes:

- For GPU builders one can optionally also pass a `cudaStream_t` if
  desired. All operations, synchronization, and memory allocs should
  happen in that stream.

- By default the GPU side builder(s) will allocate device memory; but
  it is also be possible to make them use managed memory or async
  device memory by passing the appropriate `cuBQL::GpuMemoryResource`
  to the builder.

- Following the same pattern as other libraries like tinyOBJ or STB,
  this library *can* be used in a header-only form: By default a
  included header file will only pull in the type and function
  *declaration*s, but specifying `CUBQL_GPU_BUILDER_IMPLEMENTATION` to
  1 will also pull in the implementation, so using this in one of
  one's source files allows the user to compile the builders with
  exactly the cmd-line flags, CUDA architecture, etc, that he or she
  desires. *Alternatively* (and purely optionally), when using `cmake`
  one can also link to one (or more) of specific pre-defined targets
  such as, for example, `cuBQL_cuda_float3` or `cuBQL_host_int4` that
  will then build that specific device and type specific builder(s).
  
# Traversal Templates

CuBQL is the fourth one of different libraries that all aimed at
providing fast, GPU-accelerated geometric queries. Throughout these
previous predecessor libraries, a common theme that emerged was that
whatever exact implementation the library provided, the user(s)
typically required something that, though similar, was often just a
little bit different. For example, a "find closest point point" kernel
on point data is very similar to "find closest *surface* point" (on a
set of triangles), but the actual point-to-primitive test is
nevertheless different. Similarly, k-nearest-neighbor (kNN) queries
might want to exclude certain points (e.g., based on surface normal
for photon mapping), or a "find all points that overlap this box"
kernel might only actually require the *number* of points vs another
use case that might require the actual primitive IDs, etc.
 
Based on this experience `cuBQL` decided to not only provide a set of
very specific geometric kernels, but to *also* provide a set of what
it calls `traversal templates` that can be used to easily generate
different variants of queries using lamdba functions. For example,
both kNN (with or without culling by surface normal) and find closest
point (on points *or* on triangles) in principle work the same way, by
performing a ball-shaped query where the radius of that ball *shrinks*
during traversal, based on what primitive(s) have already been
found. What exactly the query wants to do with a given primitmive that
the traversal encounters depends on the actual query code, but as long
as the traversal knows what range it has to look for *after* a given
primitmive has been processed it does not actually need to know what
specific operation that query need to do.q

In cuBQL, this pattern is realized through what we call a "shrinking
radius query": In this query, the user provides a query point (as the
origin of that query), a (intital) maximum search radius, and a lambda
function (ie, a "callback") that it wants to get called for any
"candidate" primitive encountered by the traversal. This lambda can
then do whatever that type of query needs to do with that primitive,
and can additionally return a new maximum query radius that the
traversal can then use for subsequent traversal steps. To do this, the
user would simply define a lambda function that implements the
specific per-primitive callback code, and pass that to a
`cuBQL::shrinkingRadiusQuery::forEachPrim(...)` traversal template:

```
inline __device__ 
void myQuery(bvh3f myBVH, 
             MyPrim *myPrims, 
             vec3f myQueryPoint,
             ...) 
{
  auto myQueryLambda = [...](int primID) -> float {
   ...
  };
  cuBQL::shrinkingRadiusQuery::forEachPrim
     (myQueryLambda,myQueryPoint,myBVH, ...);
}
```

For example, a `find closest point` kernel can then be realized
by having the lamdba callback simply keep track of the currently 
closest point:

```
void findClosestPoint(...)
{
   float closestDist = INFINITY;
   int   closestID   = -1;
   auto myQueryLambda = [&closestDist,&closestID,...](int primID) -> float {
      float dist = distance(queryPoint,myPrims[primID]);
      if (dist < closestDist) 
         { closestDist = dist; closestID = primID; }
      return closestDist;
   };
}
```
Note that this same patters works for both point-to-point or point-to-triangular-surface 
data! Also, the exact same pattern works for `float3` data as for `int2`, etc.

# Specific geometric queries

As described above, by far the main focus of cuBQL is the BVH
builders, and the traversal templates, for users to be able to write
their own specific queries. In addition, cuBQL also contains a
small(!) set of very specific queries such as find closest point among
float3 points, find closest surface point on triangle-mesh surface,
k-nearest (float3 point) neighbor, etc. These should be considered
more "samples" (for how to use the builder and traversal templates),
but can of course also be used directly for those that require that
exact kernel.

# Code organization:

- all of the cuBQL *library* are under `~/cuBQL/`
  - `~/cuBQL/bvh.h` defines the various BVH types
  - `~/cuBQL/builder/cuda.h` and `~/cuBQL/builder/host.h` provides the
    (header-only) source for the various builder(s)
  - `~/cuBQL/traversal/...` provides the traversal templates
- various specific queries are under `~/cuBQL/queries/` (also all header-only)
- some specific sample codes are under `~/samples/`, and some tools for more 
  comprehensive testing are under `~/testing/`
  
Most users should only every need what is under `~/cuBQL/`, in fact
most should only need the builder(s) and possibly traversal
templates. Everything else should predominantly be viewed as examples
of how to use those.

  
# Dependencies

To use `cuBQL`, you need:

- CUDA, version 12 and up. In theory some versions of CUDA 11 should work too, but 
  using 12.2 and upwards is highly recommended.
- `cmake`


# Building

As all of cuBQL's BVH builders and traversers *can* be used in a
header-only form, cuBQL can be used from within any compiler and build
system, by simply providing the proper include paths and including the
`cuBQL/bvh.h` or other header files as required.

However, we strongly suggest to use `cmake`, include cuBQL as a cmake
`add_subdirectory(...)`, and then `target_link_libraries(...)` with
the desired cuBQL cmake target.

## Building in Header-only (explicit instantiation) mode:

- in your own CUDA sources (say, `userMain.cu`):
``` 
#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1
#include <cuBQL/bvh.h>
...
void foo(...) {
	cuBQL::gpuBuilder(...)
}
```

- in your own `CMakeLists.txt`:
```
add_subdirectory(<pathTo>/cuBQL)
	
add_executable(userExec ... 
    userMain.cu ...)
	
target_link_libraries(userExec ...
    cuBQL)
```

In this case, the 'cuBQL' target that we link to is only a cmake
`INTERFACE` target that merely sets up the right include paths, but
does not build any actual library.

## Building with predefined target (eg, for float3 data)

- in your own CUDA sources (say, `userMain.cu`):
```
// do NOT define CUBQL_GPU_BUILDER_IMPLEMENTATION 
#include <cuBQL/bvh.h>
...
void foo(...) {
   cuBQL::gpuBuilder(...)
}
```

- in your own `CMakeLists.txt`:
```
add_subdirectory(<pathTo>/cuBQL)
	
add_executable(userExec ... 
   userMain.cu ...)
	
target_link_libraries(userExec ...
   cuBQL_cuda_float3)
```

