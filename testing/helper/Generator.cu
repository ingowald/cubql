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

#include "testing/helper/Generator.h"
#include "cuBQL/math/random.h"
#include <exception>

namespace cuBQL {
  namespace test_rig {

    namespace tokenizer {
      std::string findFirst(const char *curr, const char *&endOfFound)
      {
        if (curr == 0)
          return "";
      
        while (*curr && strchr(" \t\r\n",*curr)) {
          ++curr;
        }
        if (*curr == 0) {
          endOfFound = curr;
          return "";
        }
      
        std::stringstream ss;
        if (strchr("[]{}():,",*curr)) {
          ss << *curr++;
        } else if (isalnum(*curr) || *curr && strchr("+-.",*curr)) {
          while (isalnum(*curr) || *curr && strchr("+-.",*curr)) {
            ss << *curr;
            curr++;
          }
        }
        else
          throw std::runtime_error("unable to parse ... '"+std::string(curr)+"'");
      
        endOfFound = curr;
        return ss.str();
      }
    };

    template<typename T> inline T to_scalar(const std::string &s);
    template<> inline float to_scalar<float>(const std::string &s)
    { return std::stof(s); }
    template<> inline double to_scalar<double>(const std::string &s)
    { return std::stof(s); }
    template<> inline int to_scalar<int>(const std::string &s)
    { return std::stoi(s); }

    template<typename T, int D>
    vec_t<T,D> parseVector(const char *&curr)
    {
      const char *next = 0;
      std::string tok = tokenizer::findFirst(curr,next);
      assert(tok != "");
      curr = next;
      if (tok == "[") {
        std::vector<T> values;
        while(1) {
          tok = tokenizer::findFirst(curr,next);
          assert(tok != "");
          curr = next;
          if (tok == "]") break;
          values.push_back(to_scalar<T>(tok));
          std::string tok = tokenizer::findFirst(curr,next);
        }
        assert(!values.empty());
        vec_t<T,D> ret;
        for (int i=0;i<D;i++)
          ret[i] = values[i % values.size()];
        return ret;
      } else
        return vec_t<T,D>(std::stof(tok));
    }
    
    // ==================================================================
    // point generator base
    // ==================================================================
    template<typename T, int D>
    typename PointGenerator<T,D>::SP
    PointGenerator<T,D>::createAndParse(const char *&curr)
    {
      const char *next = 0;
      std::string type = tokenizer::findFirst(curr,next);
      if (type == "") throw std::runtime_error("could not parse generator type");

      typename PointGenerator<T,D>::SP gen;
      if (type == "uniform")
        gen = std::make_shared<UniformPointGenerator<T,D>>();
      else if (type == "clustered")
        gen = std::make_shared<ClusteredPointGenerator<T,D>>();
      else if (type == "nrooks")
        gen = std::make_shared<NRooksPointGenerator<T,D>>();
      else if (type == "mixture")
        gen = std::make_shared<MixturePointGenerator<T,D>>();
      else if (type == "remap")
        gen = std::make_shared<RemapPointGenerator<T,D>>();
      else
        throw std::runtime_error("un-recognized point generator type '"+type+"'");
      curr = next;
      gen->parse(curr);
      return gen;
    }


    template<typename T, int D>
    typename PointGenerator<T,D>::SP
    PointGenerator<T,D>::createFromString(const std::string &wholeString)
    {
      const char *curr = wholeString.c_str(), *next = 0;
      SP generator = createAndParse(curr);
      std::string trailing = tokenizer::findFirst(curr,next);
      if (!trailing.empty())
        throw std::runtime_error("un-recognized trailing stuff '"
                                 +std::string(curr)
                                 +"' at end of point generator string");
      return generator;
    }
  
    template<typename T, int D>
    void PointGenerator<T,D>::parse(const char *&currentParsePos)
    {}

    template struct PointGenerator<float,2>;
    template struct PointGenerator<float,3>;
    template struct PointGenerator<float,4>;
#if CUBQL_TEST_N
    template struct PointGenerator<float,CUBQL_TEST_N>;
#endif
    template struct PointGenerator<double,2>;
    template struct PointGenerator<double,3>;
    template struct PointGenerator<double,4>;
#if CUBQL_TEST_N
    template struct PointGenerator<double,CUBQL_TEST_N>;
#endif
    template struct PointGenerator<int,2>;
    template struct PointGenerator<int,3>;
    template struct PointGenerator<int,4>;
#if CUBQL_TEST_N
    template struct PointGenerator<int,CUBQL_TEST_N>;
#endif
    // ==================================================================



  
    // ==================================================================
    // box generator base
    // ==================================================================
    template<typename T, int D>
    typename BoxGenerator<T,D>::SP
    BoxGenerator<T,D>::createAndParse(const char *&curr)
    {
      const char *next = 0;
      std::string type = tokenizer::findFirst(curr,next);
      if (type == "") throw std::runtime_error("could not parse generator type");

      typename BoxGenerator<T,D>::SP gen;
      if (type == "uniform")
        gen = std::make_shared<UniformBoxGenerator<T,D>>();
      else if (type == "clustered")
        gen = std::make_shared<ClusteredBoxGenerator<T,D>>();
      else if (type == "nrooks")
        gen = std::make_shared<NRooksBoxGenerator<T,D>>();
      else if (type == "remap")
        gen = std::make_shared<RemapBoxGenerator<T,D>>();
      else if (type == "mixture")
        gen = std::make_shared<MixtureBoxGenerator<T,D>>();
      else
        throw std::runtime_error("un-recognized box generator type '"+type+"'");
      curr = next;
      gen->parse(curr);
      return gen;
    }


    template<typename T, int D>
    typename BoxGenerator<T,D>::SP
    BoxGenerator<T,D>::createFromString(const std::string &wholeString)
    {
      const char *curr = wholeString.c_str(), *next = 0;
      SP generator = createAndParse(curr);
      std::string trailing = tokenizer::findFirst(curr,next);
      if (!trailing.empty())
        throw std::runtime_error("un-recognized trailing stuff '"
                                 +std::string(curr)
                                 +"' at end of box generator string");
      return generator;
    }
  
    template<typename T, int D>
    void BoxGenerator<T,D>::parse(const char *&currentParsePos)
    {}

    template struct BoxGenerator<float,2>;
    template struct BoxGenerator<float,3>;
    template struct BoxGenerator<float,4>;
#if CUBQL_TEST_N
    template struct BoxGenerator<float,CUBQL_TEST_N>;
#endif
    template struct BoxGenerator<double,2>;
    template struct BoxGenerator<double,3>;
    template struct BoxGenerator<double,4>;
#if CUBQL_TEST_N
    template struct BoxGenerator<double,CUBQL_TEST_N>;
#endif
    template struct BoxGenerator<int,2>;
    template struct BoxGenerator<int,3>;
    template struct BoxGenerator<int,4>;
#if CUBQL_TEST_N
    template struct BoxGenerator<int,CUBQL_TEST_N>;
#endif
    // ==================================================================
  
  
    // ==================================================================
    // uniform points
    // ==================================================================

    template<typename T> inline __cubql_both T uniform_default_lower();
    template<typename T> inline __cubql_both T uniform_default_upper();
    template<> inline __cubql_both float uniform_default_lower<float>() { return 0.f; }
    template<> inline __cubql_both float uniform_default_upper<float>() { return 1.f; }
    template<> inline __cubql_both double uniform_default_lower<double>() { return 0.f; }
    template<> inline __cubql_both double uniform_default_upper<double>() { return 1.f; }
    template<> inline __cubql_both int uniform_default_lower<int>() { return -100000; }
    template<> inline __cubql_both int uniform_default_upper<int>() { return +100000; }

    template<typename T> inline __cubql_both T uniform_domain_size()
    { return uniform_default_upper<T>()-uniform_default_lower<T>(); }
    
    template<typename T, int D>
    __global__
    void uniformPointGenerator(vec_t<T,D> *d_points, int count, int seed)
    {
      int tid = threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= count) return;
      vec_t<T,D> &mine = d_points[tid];

      LCG<8> rng(seed,tid);

      T lo = uniform_default_lower<T>();
      T hi = uniform_default_upper<T>();
    
      for (int i=0;i<D;i++)
        mine[i] = T(lo + rng() * (hi-lo));
    }

    template<typename T, int D>
    void UniformPointGenerator<T,D>::generate(CUDAArray<vec_t<T,D>> &res,
                                              int count, int seed)
    {
      if (count <= 0)
        throw std::runtime_error("UniformPointGenerator<T,D>::generate(): invalid count...");
      res.resize(count);
      int bs = 1024;
      int nb = divRoundUp(int(count),bs);
      uniformPointGenerator<T,D><<<nb,bs>>>(res.data(),count,seed);
    }

    // ------------------------------------------------------------------
    template<typename T, int D>
    __global__
    void uniformBoxGenerator(box_t<T,D> *d_boxes, int count, int seed,
                             float size)
    {
      int tid = threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= count) return;
      box_t<T,D> &mine = d_boxes[tid];

      LCG<8> rng(seed,tid);

      T lo = uniform_default_lower<T>();
      T hi = uniform_default_upper<T>();

      vec_t<T,D> center;
      for (int i=0;i<D;i++)
        center[i] = T(lo + rng() * (hi-lo));

      for (int i=0;i<D;i++) {
        T scalarSize = T((hi - lo) * size);
        mine.lower[i] = center[i] - scalarSize/2;
        mine.upper[i] = mine.lower[i] + scalarSize;
      }
    }

    template<typename T, int D>
    void UniformBoxGenerator<T,D>::generate(CUDAArray<box_t<T,D>> &res,
                                            int count, int seed)
    {
      if (count <= 0)
        throw std::runtime_error("UniformBoxGenerator<T,D>::generate(): invalid count...");
      res.resize(count);
      int bs = 1024;
      int nb = divRoundUp(int(count),bs);
      float size = 0.5f / powf((float)count,float(1.f/D));
      // float size = (0.5f*uniform_domain_size<T>()) / powf((float)count,float(1.f/D));
      uniformBoxGenerator<T,D><<<nb,bs>>>(res.data(),count,seed,size);
    }

    // ==================================================================
    // re-mapping
    // ==================================================================
  
    template<typename T, int D>
    __global__
    void remapPointsKernel(vec_t<T,D> *points,
                           int count,
                           vec_t<T,D> lower,
                           vec_t<T,D> upper)
    {
      int tid = threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= count) return;

      auto &point = points[tid];
      for (int d=0;d<D;d++)
        point[d]
          = lower[d]
          + T(point[d]
              * (typename dot_result_t<T>::type)(upper[d]-lower[d])
              / (uniform_default_upper<T>() - uniform_default_lower<T>()));
    }
  
    template<typename T, int D>
    RemapPointGenerator<T,D>::RemapPointGenerator()
    {
      for (int d=0;d<D;d++) {
        lower[d] = uniform_default_lower<T>();
        upper[d] = uniform_default_upper<T>();
      }
    }

    template<typename T, int D>
    void RemapPointGenerator<T,D>::generate(CUDAArray<vec_t<T,D>> &pts,
                                            int count, int seed)
    {
      if (!source)
        throw std::runtime_error("RemapPointGenerator: no source defined");
      source->generate(pts,count,seed);

      int bs = 128;
      int nb = divRoundUp(count,bs);
      remapPointsKernel<<<nb,bs>>>(pts.get(),count,lower,upper);
    }

    template<typename T, int D>
    void RemapPointGenerator<T,D>::parse(const char *&currentParsePos)
    {
      // const char *next = 0;
      // for (int d=0;d<D;d++) {
      //   std::string tok = tokenizer::findFirst(currentParsePos,next);
      //   assert(tok != "");
      //   lower[d] = to_scalar<T>(tok);
      //   currentParsePos = next;
      // }
      // for (int d=0;d<D;d++) {
      //   std::string tok = tokenizer::findFirst(currentParsePos,next);
      //   assert(tok != "");
      //   upper[d] = to_scalar<T>(tok);
      //   currentParsePos = next;
      // }
      lower = parseVector<T,D>(currentParsePos);
      upper = parseVector<T,D>(currentParsePos);
      source = PointGenerator<T,D>::createAndParse(currentParsePos);
    }

    // ------------------------------------------------------------------



    template<typename T, int D>
    __global__
    void remapBoxsKernel(box_t<T,D> *boxs,
                         int count,
                         vec_t<T,D> lower,
                         vec_t<T,D> upper)
    {
      int tid = threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= count) return;

      auto &box = boxs[tid];
      for (int d=0;d<D;d++)
        box.lower[d]
          = lower[d]
          + T(box.lower[d]
              * (typename dot_result_t<T>::type)(upper[d]-lower[d])
              / (uniform_default_upper<T>() - uniform_default_lower<T>()));
      for (int d=0;d<D;d++)
        box.upper[d]
          = lower[d]
          + T(box.upper[d]
              * (typename dot_result_t<T>::type)(upper[d]-lower[d])
              / (uniform_default_upper<T>() - uniform_default_lower<T>()));
    }
  
    template<typename T, int D>
    RemapBoxGenerator<T,D>::RemapBoxGenerator()
    {
      for (int d=0;d<D;d++) {
        lower[d] = uniform_default_lower<T>();
        upper[d] = uniform_default_upper<T>();
      }
    }

    template<typename T, int D>
    void RemapBoxGenerator<T,D>::generate(CUDAArray<box_t<T,D>> &boxes,
                                          int count, int seed)
    {
      if (!source)
        throw std::runtime_error("RemapBoxGenerator: no source defined");
      source->generate(boxes,count,seed);

      int bs = 128;
      int nb = divRoundUp(count,bs);
      remapBoxsKernel<<<nb,bs>>>(boxes.get(),count,lower,upper);
    }

    template<typename T, int D>
    void RemapBoxGenerator<T,D>::parse(const char *&currentParsePos)
    {
      // const char *next = 0;
      lower = parseVector<T,D>(currentParsePos);
      upper = parseVector<T,D>(currentParsePos);
      // for (int d=0;d<D;d++) {
      //   std::string tok = tokenizer::findFirst(currentParsePos,next);
      //   assert(tok != "");
      //   lower[d] = to_scalar<T>(tok);
      //   currentParsePos = next;
      // }
      // for (int d=0;d<D;d++) {
      //   std::string tok = tokenizer::findFirst(currentParsePos,next);
      //   assert(tok != "");
      //   upper[d] = to_scalar<T>(tok);
      //   currentParsePos = next;
      // }
      source = BoxGenerator<T,D>::createAndParse(currentParsePos);
    }


  
    // ==================================================================
    // clustered points
    // ==================================================================

    template<typename T, int D>
    void ClusteredPointGenerator<T,D>::generate(CUDAArray<vec_t<T,D>> &res,
                                                int count, int seed)
    {
      std::default_random_engine rng;
      rng.seed(seed);
      std::uniform_real_distribution<double>
      //   uniform(0.f,1.f);
        uniform(uniform_default_lower<T>(),
                uniform_default_upper<T>());
      
      int numClusters = int(1+powf((float)count,(D-1.f)/D));
      // = int(1+powf(count/50.f);
      // = int(1+sqrtf(count));
      // = this->numClusters
      // ? this->numClusters
      // : int(1+sqrtf(count));
      std::vector<vec_t<float,D>> clusterCenters;
      for (int cc=0;cc<numClusters;cc++) {
        vec_t<float,D> c;
        for (int i=0;i<D;i++)
          c[i] = (T)uniform(rng);
        clusterCenters.push_back(c);
      }
    
      float sigma = 2.f/numClusters;
      std::normal_distribution<double> gaussian(0.f,sigma);
      std::vector<vec_t<T,D>> points;
      for (int sID=0;sID<count;sID++) {
        int clusterID = int(uniform(rng)*numClusters) % numClusters;
        vec_t<T,D> pt;
        for (int i=0;i<D;i++)
          pt[i] = T(gaussian(rng) + clusterCenters[clusterID][i]);
        points.push_back(pt);
      }
    
      res.upload(points);
    }
  
    template<typename T, int D>
    void ClusteredBoxGenerator<T,D>::parse(const char *&currentParsePos)
    {
      const char *next = 0;
      while (true) {
        const std::string tag = tokenizer::findFirst(currentParsePos,next);
        if (tag == "")
          break;

        if (tag == "gaussian") {
          currentParsePos = next;
          
          std::string sMean = tokenizer::findFirst(currentParsePos,next);
          currentParsePos = next;
        
          std::string sSigma = tokenizer::findFirst(currentParsePos,next);
          currentParsePos = next;
        
          gaussianSize.mean = std::stof(sMean);
          gaussianSize.sigma = std::stof(sSigma);
        } else if (tag == "gaussian.scale") {
          currentParsePos = next;
          
          std::string scale = tokenizer::findFirst(currentParsePos,next);
          assert(scale != "");
          currentParsePos = next;
          gaussianSize.scale = std::stof(scale); 
        } else {
          break;
        }
      }
    }
    
    template<typename T, int D>
    void ClusteredBoxGenerator<T,D>::generate(CUDAArray<box_t<T,D>> &res,
                                              int count, int seed)
    {
      std::default_random_engine rng;
      rng.seed(seed);
      std::uniform_real_distribution<double> uniform(0.f,1.f);
  
      int numClusters = int(1+powf((float)count,(D-1.f)/D));
      // int numClusters
      //   = int(1+count/50.f);
      std::vector<vec_t<T,D>> clusterCenters;
      for (int cc=0;cc<numClusters;cc++) {
        vec_t<T,D> c;
        for (int i=0;i<D;i++)
          c[i] = (T)uniform(rng);
        clusterCenters.push_back(c);
      }

      float sigma = 2.f/numClusters;
    
      std::normal_distribution<double> gaussian(0.f,sigma);

      float sizeMean = -1.f, sizeSigma = 0.f;
      if (gaussianSize.mean > 0) {
        std::cout << "choosing size using gaussian distribution..." << std::endl;
        sizeMean = gaussianSize.mean*gaussianSize.scale;
        sizeSigma = gaussianSize.sigma;
      } else if (uniformSize.min > 0) {
        std::cout << "choosing size using uniform min/max distribution..." << std::endl;
        sizeMean = -1.f;
      } else {
        // std::cout << "choosing size using auto-chosen gaussian..." << std::endl;
        // int avgBoxesPerCluster = count / numClusters;
        // // float avgClusterWidth = 4.f*sigma;
        // sizeMean = 1.f/powf(avgBoxesPerCluster,1.f/D);
        // PRINT(sizeMean);
        // sizeSigma = sizeMean/5.f;
        std::cout << "choosing size using auto-chosen gaussian..." << std::endl;
        float avgClusterWidth = 4*sigma;
        // int avgBoxesPerCluster = count / numClusters;
        sizeMean = .5f*avgClusterWidth*gaussianSize.scale;//powf(avgBoxesPerCluster,1.f/D);
        sizeSigma = sizeMean/3.f;
        std::cout << "choosing size using auto-config'ed gaussian"
                  << " mean=" << sizeMean
                  << " sigma=" << sizeSigma
                  << std::endl;
      }
    
      std::normal_distribution<double> sizeGaussian(sizeMean,sizeSigma);
      std::vector<box_t<T,D>> boxes;
      for (int sID=0;sID<count;sID++) {
        int clusterID = int(uniform(rng)*numClusters) % numClusters;
        vec_t<T,D> center, halfSize;
        for (int i=0;i<D;i++)
          center[i] = T(gaussian(rng) + clusterCenters[clusterID][i]);

        if (sizeMean > 0) {
          for (int d=0;d<D;d++)
            halfSize[d] = (T)fabsf(0.5f*(float)sizeGaussian(rng));
        } else {
          for (int d=0;d<D;d++)
            halfSize[d]
              = T(uniformSize.min
                  + (uniformSize.max-uniformSize.min) * uniform(rng));
        }
        box_t<T,D> box;
        box.lower = center - halfSize;
        box.upper = center + halfSize;
        boxes.push_back(box);
      }

      res.upload(boxes);
    }
  


    // ==================================================================
    // "nrooks": generate N clusters of ~50 points each, then arrange
    // these N clusters in a NxNx...xN grid with a N-rooks pattern. Each
    // of these clusters has ~50 uniformly distributed points inside of
    // that clusters "field"
    // ==================================================================
    template<typename T, int D>
    void NRooksPointGenerator<T,D>::generate(CUDAArray<vec_t<T,D>> &res,
                                             int count, int seed)
    {
      int numClusters = (int)(1+powf((float)count,0.5f*(D-1.f)/D));
      LCG<8> rng(seed,290374);
      std::vector<vec_t<T,D>> clusterLower(numClusters);
      for (int d=0;d<D;d++) {
        for (int i=0;i<numClusters;i++) {
          clusterLower[i][d] = i/(float)numClusters;
        }
        for (int i=numClusters-1;i>0;--i) {
          int o = rng.ui32() % (i+1);
          if (i != o)
            std::swap(clusterLower[i][d],clusterLower[o][d]);
        }
      }
    
      std::vector<vec_t<T,D>> points(count);
      for (int i=0;i<count;i++) {
        int clusterID = rng.ui32() % numClusters;
        for (int d=0;d<D;d++)
          points[i][d] = clusterLower[clusterID][d] + (1.f/numClusters)*rng();
      }
    
      res.resize(count);
      res.upload(points);
    }

    template<typename T, int D>
    void NRooksBoxGenerator<T,D>::parse(const char *&currentParsePos)
    {
      const char *next = 0;
      while (true) {
        const std::string tag = tokenizer::findFirst(currentParsePos,next);
        if (tag == "")
          break;

        if (tag == "gaussian") {
          currentParsePos = next;
          
          std::string sMean = tokenizer::findFirst(currentParsePos,next);
          currentParsePos = next;
        
          std::string sSigma = tokenizer::findFirst(currentParsePos,next);
          currentParsePos = next;
        
          gaussianSize.mean = std::stof(sMean);
          gaussianSize.sigma = std::stof(sSigma);
        } else if (tag == "gaussian.scale") {
          currentParsePos = next;
          
          std::string scale = tokenizer::findFirst(currentParsePos,next);
          assert(scale != "");
          currentParsePos = next;
          gaussianSize.scale = std::stof(scale); 
        } else {
          break;
        }
      }
    }
  
    template<typename T, int D>
    void NRooksBoxGenerator<T,D>::generate(CUDAArray<box_t<T,D>> &res,
                                           int count, int seed)
    {
      int numClusters = (int)(1+powf((float)count,0.5f*(D-1.f)/D));
      LCG<8> lcg(seed,290374);
      std::vector<vec_t<T,D>> clusterLower(numClusters);
      for (int d=0;d<D;d++) {
        for (int i=0;i<numClusters;i++) {
          clusterLower[i][d] = i/(float)numClusters;
        }
        for (int i=numClusters-1;i>0;--i) {
          int o = lcg.ui32() % (i+1);
          if (i != o)
            std::swap(clusterLower[i][d],clusterLower[o][d]);
        }
      }

      float sizeMean = -1.f, sizeSigma = 0.f;
      if (gaussianSize.mean > 0) {
        sizeMean = gaussianSize.mean;
        sizeSigma = gaussianSize.sigma;
        std::cout << "choosing size using user-supplied gaussian"
                  << " mean=" << sizeMean
                  << " sigma=" << sizeSigma
                  << std::endl;
      } else if (uniformSize.min > 0) {
        std::cout << "choosing size using uniform min/max distribution..." << std::endl;
        sizeMean = -1.f;
      } else {
        std::cout << "choosing size using auto-chosen gaussian..." << std::endl;
        float avgClusterWidth = 1.f/numClusters;
        // float avgClusterWidth = 4*sigma;
        // int avgBoxesPerCluster = count / numClusters;
        sizeMean = .5f*avgClusterWidth*gaussianSize.scale;//powf(avgBoxesPerCluster,1.f/D);
        sizeSigma = sizeMean/3.f;
        std::cout << "choosing size using auto-config'ed gaussian"
                  << " mean=" << sizeMean
                  << " sigma=" << sizeSigma
                  << std::endl;
        // std::cout << "choosing size using auto-chosen gaussian..." << std::endl;
        // // int avgBoxesPerCluster = count / numClusters;
        // float avgClusterWidth = 1.f/numClusters;
        // sizeMean = .1f*avgClusterWidth;//powf(avgBoxesPerCluster,1.f/D);
        // sizeSigma = sizeMean/3.f;
        // std::cout << "choosing size using auto-config'ed gaussian"
        //           << " mean=" << sizeMean
        //           << " sigma=" << sizeSigma
        //           << std::endl;
      }
    
      std::normal_distribution<double> sizeGaussian(sizeMean,sizeSigma);
      std::default_random_engine reng;
      std::uniform_real_distribution<double> uniform(0.f,1.f);
      reng.seed(seed+29037411);

      std::vector<box_t<T,D>> boxes;

      for (int i=0;i<count;i++) {
        int clusterID = lcg.ui32() % numClusters;
        vec_t<T,D> center;
        for (int d=0;d<D;d++) {
          center[d] = clusterLower[clusterID][d] + (1.f/numClusters)*lcg();
        }
      
        vec_t<T,D> halfSize;
        if (sizeMean > 0) {
          for (int d=0;d<D;d++)
            halfSize[d] = fabsf(0.5f*(float)sizeGaussian(reng));
        } else {
          for (int d=0;d<D;d++)
            halfSize[d]
              = T(uniformSize.min
                  + (uniformSize.max-uniformSize.min) * uniform(reng));
        }
        box_t<T,D> box;
        box.lower = center - halfSize;
        box.upper = center + halfSize;
        boxes.push_back(box);
      }
    
      res.upload(boxes);
    }


    // ==================================================================
    template<typename T, int D>
    void TrianglesBoxGenerator<T,D>::generate(CUDAArray<box_t<T,D>> &d_boxes,
                                              int numRequested, int seed)
    {
      throw std::runtime_error("can generate boxes from triangles only "
                               "for T=float and D=3");
    }

    template<>
    void TrianglesBoxGenerator<float,3>::generate(CUDAArray<box_t<float,3>> &d_boxes,
                                                  int numRequested, int seed)
    {
      std::vector<box3f> boxes;
      for (auto tri : triangles)
        boxes.push_back(make_box<float,3>(tri.a).grow(tri.b).grow(tri.c));
      d_boxes.upload(boxes);
    }
    
    template<typename T, int D>
    void TrianglesBoxGenerator<T,D>::parse(const char *&currentParsePos)
    {
      const char *next = 0;

      std::string format = tokenizer::findFirst(currentParsePos,next);
      if (format == "") throw std::runtime_error("no triangles file format specified");
      currentParsePos = next;

      std::string fileName = tokenizer::findFirst(currentParsePos,next);
      if (fileName == "") throw std::runtime_error("no file name specified");

      std::cout << "going to start reading triangles from '"
                << fileName << "'" << std::endl;
      triangles = loadData<Triangle>(fileName);
      std::cout << "done loading " << prettyNumber(triangles.size())
                << " triangles..." << std::endl;
      currentParsePos = next;
    }



    // ==================================================================
    template<typename T, int D>
    void 
    TrianglesPointGenerator<T,D>::generate(CUDAArray<vec_t<T,D>> &d_points,
                                           int numRequested, int seed)
    {
      throw std::runtime_error("can generate sample points from triangles only "
                               "for T=float and D=3");
    }

    template<>
    void
    TrianglesPointGenerator<float,3>::generate(CUDAArray<vec_t<float,3>> &d_points,
                                               int numRequested, int seed)
    {
      float sumAreas = 0.f;
      std::vector<float> areas;
      std::vector<box3f> boxes;
      for (auto tri : triangles) {
        boxes.push_back(make_box<float,3>(tri.a).grow(tri.b).grow(tri.c));
        float a = area(tri);
        areas.push_back(a);
        sumAreas += a;
      }

      std::vector<vec3f> points = sample(triangles,numRequested,seed);
      assert(points.size() == numRequested);
      
      d_points.upload(points);
    }
    
    template<typename T, int D>
    void TrianglesPointGenerator<T,D>::parse(const char *&currentParsePos)
    {
      const char *next = 0;

      std::string format = tokenizer::findFirst(currentParsePos,next);
      if (format == "") throw std::runtime_error("no triangles file format specified");
      currentParsePos = next;

      std::string fileName = tokenizer::findFirst(currentParsePos,next);
      if (fileName == "") throw std::runtime_error("no file name specified");

      std::cout << "going to start reading triangles from '"
                << fileName << "'" << std::endl;
      triangles = loadData<Triangle>(fileName);
      std::cout << "done loading " << prettyNumber(triangles.size())
                << " triangles..." << std::endl;
      currentParsePos = next;
    }

    template<typename T, int D>
    __global__ void mixKernel(box_t<T,D> *out, int outCount,
                              float prob_a,
                              int seed,
                              const box_t<T,D> *a, int aCount,
                              const box_t<T,D> *b, int bCount)
    {
      int tid = threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= outCount) return;
      
      box_t<T,D> &mine = out[tid];
      LCG<8> rng(seed,tid);
      
      bool use_a
        = (prob_a < 1.f)
        ? (rng() < prob_a)
        : (tid < (int)prob_a);
      const box_t<T,D> *in = (use_a ? a : b);
      int inCount = (use_a ? aCount : bCount);
      
      int which
        = (inCount == outCount)
        ? tid
        : (rng.ui32() & inCount);
      mine = in[which];
    }
    
    // ==================================================================
    /*! "mixture" generator - generates a new distributoin based by
      randomly picking between two input distributions */
    template<typename T, int D>
    void MixtureBoxGenerator<T,D>::generate(CUDAArray<box_t<T,D>> &boxes,
                                            int numRequested, int seed)
    {
      assert(gen_a);
      assert(gen_b);
      CUDAArray<box_t<T,D>>  boxes_a;
      gen_a->generate(boxes_a,numRequested,3*seed+0);
      CUDAArray<box_t<T,D>>  boxes_b;
      gen_b->generate(boxes_b,numRequested,3*seed+1);

      boxes.resize(numRequested);
      
      int bs = 128;
      int nb = divRoundUp(numRequested,bs);
      mixKernel<<<nb,bs>>>(boxes.data(),(int)boxes.size(),
                           prob_a,
                           3*seed+2,
                           boxes_a.data(),(int)boxes_a.size(),
                           boxes_b.data(),(int)boxes_b.size());
    }
    
    template<typename T, int D>
    void MixtureBoxGenerator<T,D>::parse(const char *&currentParsePos)
    {
      const char *next = 0;

      std::string prob = tokenizer::findFirst(currentParsePos,next);
      currentParsePos = next;
      
      if (prob == "") throw std::runtime_error("no mixture probabilty specified");

      prob_a = std::stof(prob);
      gen_a = BoxGenerator<T,D>::createAndParse(currentParsePos);
      gen_b = BoxGenerator<T,D>::createAndParse(currentParsePos);
    }
    
    template<typename T, int D>
    __global__ void mixKernel(vec_t<T,D> *out, int outCount,
                              float prob_a,
                              int seed,
                              const vec_t<T,D> *a, int aCount,
                              const vec_t<T,D> *b, int bCount)
    {
      int tid = threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= outCount) return;
      
      vec_t<T,D> &mine = out[tid];
      LCG<8> rng(seed,tid);
      
      bool use_a
        = (prob_a < 1.f)
        ? (rng() < prob_a)
        : (tid < (int)prob_a);
      const vec_t<T,D> *in = (use_a ? a : b);
      int inCount = (use_a ? aCount : bCount);
      
      int which
        = (inCount == outCount)
        ? tid
        : (rng.ui32() & inCount);
      mine = in[which];
    }
    
    // ==================================================================
    /*! "mixture" generator - generates a new distributoin based by
      randomly picking between two input distributions */
    template<typename T, int D>
    void MixturePointGenerator<T,D>::generate(CUDAArray<vec_t<T,D>> &points,
                                              int numRequested, int seed)
    {
      assert(gen_a);
      assert(gen_b);
      CUDAArray<vec_t<T,D>>  points_a;
      gen_a->generate(points_a,numRequested,3*seed+0);
      CUDAArray<vec_t<T,D>>  points_b;
      gen_b->generate(points_b,numRequested,3*seed+1);

      points.resize(numRequested);
      
      int bs = 128;
      int nb = divRoundUp(numRequested,bs);
      mixKernel<<<nb,bs>>>(points.data(), (int)points.size(),
                           prob_a,
                           3*seed+2,
                           points_a.data(), (int)points_a.size(),
                           points_b.data(), (int)points_b.size());
    }
    
    template<typename T, int D>
    void MixturePointGenerator<T,D>::parse(const char *&currentParsePos)
    {
      const char *next = 0;

      std::string prob = tokenizer::findFirst(currentParsePos,next);
      currentParsePos = next;
      
      if (prob == "") throw std::runtime_error("no mixture probabilty specified");

      prob_a = std::stof(prob);
      gen_a = PointGenerator<T,D>::createAndParse(currentParsePos);
      gen_b = PointGenerator<T,D>::createAndParse(currentParsePos);
    }
    
  } // ::cuBQL::test_rig
} // ::cuBQL
