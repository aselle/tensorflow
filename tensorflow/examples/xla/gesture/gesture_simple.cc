/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#define EIGEN_USE_THREADS
#define EIGEN_USE_CUSTOM_THREAD_POOL
#include "tensorflow/examples/xla/gesture/gesturelib.h"
#include <iostream>
#include <cmath>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"


constexpr int BATCH_SIZE=256;
constexpr int N=32;
int main(int argc, char** argv) {
  Eigen::ThreadPool tp(2);  // Size the thread pool as appropriate.
  Eigen::ThreadPoolDevice device(&tp, tp.NumThreads());
  Examples::Gesture gesture;
  gesture.set_thread_pool(&device);
  float* vals = gesture.arg0_data();
  std::fill(vals,vals+BATCH_SIZE*N*2,0.f);
  float theta=0.f;
  int idx=0;
  for(int i=0;i<N;i++){
    vals[idx++] = cos(theta);
    vals[idx++] = sin(theta);
    theta += i/(2*M_PI);
  }
  {
    float x = 0.f;
    float theta = 0.f;
    float y = 0.f;
    for(int i=0;i<N;i++){
      float thetaDelta = .4*(float(rand())/RAND_MAX-0.5f);
      theta += thetaDelta;
      x += cos(thetaDelta);
      y += sin(thetaDelta);
      vals[idx++] = x;
      vals[idx++] = y;
    }
  }
  for(int i=0;i<N;i++){
    vals[idx++] = i;
    vals[idx++] = 2*i;
  }
  
  gesture.Run();
  tensorflow::int64* data=gesture.result0_data();
  for(int i=0;i<3;i++){
    std::cout<<"test "<<i<<" is "<<data[i]<<std::endl;
  }
}
