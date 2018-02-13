// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
#include "../raja.hpp"

// *****************************************************************************
#ifdef __NVCC__
#include <cub/cub.cuh>

// *****************************************************************************
static double cub_vector_min(const int N,
                             const double* __restrict vec) {
  static double *h_min = NULL;
  if (!h_min){
    const int flag = CU_MEMHOSTALLOC_PORTABLE;
    checkCudaErrors(cuMemHostAlloc((void**)&h_min,sizeof(double),flag));
    // should be freed with cuMemFreeHost()
  }
  static double *d_min = NULL;
  if (!d_min)
    checkCudaErrors(cuMemAlloc((CUdeviceptr*)&d_min, 1*sizeof(double)));
  static void *d_storage = NULL;
  static size_t storage_bytes = 0;
  if (!d_storage){
    cub::DeviceReduce::Min(d_storage, storage_bytes, vec, d_min, N);
    cuMemAlloc((CUdeviceptr*)&d_storage, storage_bytes);
  }
  //printf(" \033[33;1m%d\033[m", storage_bytes);fflush(stdout);
  cub::DeviceReduce::Min(d_storage, storage_bytes, vec, d_min, N);
  //D2H
  checkCudaErrors(cuMemcpyDtoH(h_min,(CUdeviceptr)d_min,1*sizeof(double)));
  return *h_min;
}
#endif // __NVCC__


// *****************************************************************************
double vector_min(const int N,
                  const double* __restrict vec) {
  push(min,Cyan);
#ifdef __NVCC__
  if (mfem::rconfig::Get().Cuda()){
    const double result = cub_vector_min(N,vec);
    pop();
    return result;
  }
#endif
  ReduceDecl(Min,red,vec[0]);
  ReduceForall(i,N,red.min(vec[i]););
  pop();
  return red;
}

