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
#ifndef LAGHOS_RAJA_KERNELS_FORALL
#define LAGHOS_RAJA_KERNELS_FORALL

// *****************************************************************************
extern "C" bool is_managed;

// RAJA build ******************************************************************
#ifdef __RAJA__
#warning RAJA
const int CUDA_BLOCK_SIZE = 256;
#define cu_device __device__
#define cu_exec RAJA::cuda_exec<CUDA_BLOCK_SIZE>
#define cu_reduce RAJA::cuda_reduce<CUDA_BLOCK_SIZE>

#define sq_device __host__
#define sq_exec RAJA::seq_exec
#define sq_reduce RAJA::seq_reduce

// RAJA reduce *****************************************************************
#define ReduceDecl(type,var,ini) \
  RAJA::Reduce ## type<cu_reduce, RAJA::Real_type> var(ini);

// RAJA forall *****************************************************************
#define forall(i,max,body)                                              \
  if (is_managed)                                                       \
    RAJA::forall<cu_exec>(0,max,[=]cu_device(RAJA::Index_type i) {body}); \
  else                                                                  \
    RAJA::forall<sq_exec>(0,max,[=]sq_device(RAJA::Index_type i) {body});

#else // __RAJA__
//#warning KERNELS
// Kernels forall **************************************************************
#ifdef __NVCC__
#define forall(i,max,body) \
  if (is_managed)                                                     \
    for(int i=0;i<max;[=]__device__(int i) {body});                   \
  else                                                                \
    for(int i=0;i<max;i++){body}
#else // __NVCC__
// Kernels reduce **************************************************************
class ReduceSum{
private:
  double s;
public:
  inline ReduceSum(double d):s(d){}
  inline operator double() { return s; }
  inline ReduceSum& operator +=(const double d) { return *this = (s+d); }
};
class ReduceMin{
private:
  double m;
public:
  inline ReduceMin(double d):m(d){}
  inline operator double() { return m; }
  inline ReduceMin& min(const double d) { return *this=(m<d)?m:d; }
};
#define ReduceDecl(type,var,ini) Reduce##type var(ini);
#define forall(i,max,body) for(int i=0;i<max;i++){body}
#endif //__NVCC__

#endif // __RAJA__

#endif // LAGHOS_RAJA_KERNELS_FORALL
