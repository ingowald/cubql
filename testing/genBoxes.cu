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

// #define CUBQL_GPU_BUILDER_IMPLEMENTATION 1
#include "cuBQL/bvh.h"
#include "cuBQL/computeSAH.h"
#include "cuBQL/queries/fcp.h"
#include "cuBQL/queries/knn.h"

#include "testing/helper/CUDAArray.h"
#include "testing/helper.h"
#include "testing/helper/Generator.h"
#include <fstream>
#include <sstream>

#define PLOT_BOXES 1

namespace cuBQL {
  namespace test_rig {

    std::vector<float> reference;

    template<typename T, int D>
    std::ostream &operator<<(std::ostream &o, const vec_t<T,D> &v)
    {
      o << "(";
      for (int i=0;i<D;i++) {
        if (i) o << ",";
        o << v[i];
      }
      o << ")";
      return o;      
    }
    template<typename T, int D>
    std::ostream &operator<<(std::ostream &o, const box_t<T,D> &b)
    {
      o << "{" << b.lower << "," << b.upper << "}";
      return o;
    }
    
    struct TestConfig {
      /* knn_k = 0 means fcp, every other number means knn-query with this k */

      float maxTimeThreshold = 10.f;
      float maxQueryRadius = INFINITY;

      bool useBoxGenerator = false;
      std::string dataGen = "uniform";
      int dataCount = 100000;

      std::string outFileName;
    };
  
    void usage(const std::string &error = "")
    {
      if (!error.empty()) {
        std::cerr << error << "\n\n";
      }
      std::cout << "./cuBQL_fcpAndKnn <args>\n\n";
      std::cout << "w/ args:\n";
      std::cout << "-dc <data_count>\n";
      std::cout << "-dg <data_generator_string> (see generator strings)\n";
      std::cout << "-qc <guery_count> (see generator strings)\n";
      std::cout << "-qg <query_generator_string> (see generator strings)\n";
      
      exit(error.empty()?0:1);
    }

    template<typename T, int D>
    void saveBoxes(const std::vector<cuBQL::box_t<T,D>> &boxes,
                   const std::string &fileName)
    {
      std::cout << "saving " << prettyNumber(boxes.size()) << " boxes to " << fileName << std::endl;

      std::cout << "for reference, here are a few of them:" << std::endl;
      for (int log_i=0;true;log_i++) {
        int i = 1ull<<log_i;
        bool done = false;
        if (i >= boxes.size()) { done = true; i = boxes.size()-1; }

        std::cout << "box #" << i << " : " << boxes[i] << std::endl;
        if (done) break;
      }
      std::ofstream f(fileName.c_str(),std::ios::binary);
      f.write((const char *)boxes.data(),boxes.size()*sizeof(boxes[0]));
    }
    
    template<typename T, int D>
    __global__
    void makeBoxes(cuBQL::box_t<T,D> *boxes,
                   cuBQL::vec_t<T,D> *points,
                   int numPoints)
    {
      int tid = threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= numPoints) return;
      vec_t<T,D> point = points[tid];
      boxes[tid].lower = point;
      boxes[tid].upper = point;
    }

    // ------------------------------------------------------------------
  
    template<typename T, int D>
    void genData(TestConfig testConfig,
                 BuildConfig buildConfig)
    {
      using point_t = cuBQL::vec_t<T,D>;
      using box_t = cuBQL::box_t<T,D>;
      if (testConfig.useBoxGenerator) {
        typename BoxGenerator<T,D>::SP dataGenerator
          = BoxGenerator<T,D>::createFromString(testConfig.dataGen);
        CUDAArray<box_t> data;
        dataGenerator->generate(data,testConfig.dataCount,0x1345);
        auto &boxes = data;
        std::vector<box_t> h_data = boxes.download();
        saveBoxes<T,D>(h_data,testConfig.outFileName);
      } else {
        typename PointGenerator<T,D>::SP dataGenerator
          = PointGenerator<T,D>::createFromString(testConfig.dataGen);
        CUDAArray<point_t> data;
        dataGenerator->generate(data,testConfig.dataCount,0x1345);
        
        CUDAArray<box_t> boxes(data.size());
        {
          int bs = 256;
          int nb = divRoundUp((int)data.size(),bs);
          makeBoxes<<<nb,bs>>>(boxes.data(),data.get(),(int)data.size());
        };
        std::vector<box_t> h_data = boxes.download();
        saveBoxes<T,D>(h_data,testConfig.outFileName);
      }
    }
  
    template<int D>
    void genData(const std::string &type,
                 TestConfig testConfig,
                 BuildConfig buildConfig)
    {
      if (type == "float")
        genData<float,D>(testConfig,buildConfig);
      else if (type == "int")
        genData<int,D>(testConfig,buildConfig);
      else if (type == "double")
        genData<double,D>(testConfig,buildConfig);
      else
        throw std::runtime_error("un-handled type '"+type+"'");
    }

  } // ::cuBQL::test_rig
} // ::cuBQL

using namespace ::cuBQL::test_rig;

int main(int ac, char **av)
{ 
  BuildConfig buildConfig;
  std::string bvhType = "binary";
  TestConfig testConfig;
  int numDims = 3;
  std::string type = "float";
  for (int i=1;i<ac;i++) {
    const std::string arg = av[i];
    if (arg == "-dg" || arg == "--data-dist" || arg == "--data-generator") {
      testConfig.dataGen = av[++i];
    } else if (arg == "-dc" || arg == "--data-count") {
      testConfig.dataCount = std::stoi(av[++i]);
    } else if (arg == "-t" || arg == "--type") {
      type = av[++i];
    } else if (arg == "-o") {
      testConfig.outFileName = av[++i];
    } else if (arg == "-b" || arg == "--boxes") {
      testConfig.useBoxGenerator = true;
    } else if (arg == "-d" || arg == "-nd" || arg == "--num-dims") {
      if (std::string(av[i+1]) == "n") {
        numDims = CUBQL_TEST_N;
        ++i;
      } else
        numDims = std::stoi(av[++i]);
    } else
      usage("unknown cmd-line argument '"+arg+"'");
  }

  if (testConfig.outFileName.empty())
    throw std::runtime_error("no output filename specified");
  
  if (numDims == 2)
    genData<2>(type,testConfig,buildConfig);
  else if (numDims == 3)
    genData<3>(type,testConfig,buildConfig);
  else if (numDims == 4)
    genData<4>(type,testConfig,buildConfig);
#if CUBQL_TEST_N
  else if (numDims == CUBQL_TEST_N)
    genData<CUBQL_TEST_N>(type,testConfig,buildConfig);
#endif
  else
    throw std::runtime_error("unsupported number of dimensions "+std::to_string(numDims));
  return 0;
}
