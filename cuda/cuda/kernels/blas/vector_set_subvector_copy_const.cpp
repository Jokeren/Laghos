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
#include "../cuda.hpp"

// *****************************************************************************
extern "C" kernel
void vector_set_subvector_copy_const0(const int N,
                                 const int ess_tdofs_count,
                                 double* __restrict data,
                                 double* __restrict x,
                                 const int* __restrict tdofs)
{
   const int i = blockDim.x * blockIdx.x + threadIdx.x;
   data[i] = x[i];
   if (i >= ess_tdofs_count) { return; }
   const int dof_i = tdofs[i];
   data[dof_i] = 0.0;
   if (dof_i >= 0)
   {
      data[dof_i] = 0.0;
   }
   else
   {
      data[-dof_i-1] = 0.0;
   }
}

// *****************************************************************************
void vector_set_subvector_copy_const(const int N,
                                     const int ess_tdofs_count,
                                     double* __restrict data,
                                     double* __restrict x,
                                     const int* __restrict tdofs)
{
   cuKer(vector_set_subvector_copy_const, N, ess_tdofs_count, data, x, tdofs);
}

