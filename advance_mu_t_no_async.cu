// make sure 'jds==jps', etc.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include "config_flags.h"
#include "advance_mu_t_cu.h"

#define BLOCKSIZE 64
#define GPUs 3

#define min(a,b)  ((a)<(b)?(a):(b))

//int dev_id[GPUs] = {0};
//int dev_id[GPUs] = {0,2};
//int dev_id[GPUs] = {0,2,3};
int dev_id[GPUs] = {0,1,2};
//int dev_id[GPUs] = {0,1,2,3};

static void HandleError(cudaError_t err,
                        const char *file,
                        int line) {
    if(err != cudaSuccess){
      printf("%s in %s at line %d\n", cudaGetErrorString(err),
             file, line);
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

//--------------------------------------------------------------------
void advance_mu_t( float *ww, float *ww_1, float *u, float *u_1, 
                   float *v, float *v_1,            
                   float *mu, float *mut, float *muave, float *muts, 
                   float *muu,float *muv,
                   float *mudf, float *t, float *t_1,                      
                   float *t_ave, float *ft, float *mu_tend,                  
                   float rdx, float rdy, float dts, float epssm,               
                   float *dnw, float *fnm, float *fnp, float *rdnw,              
                   float *msfuy, float *msfvx_inv,                 
                   float *msftx, float *msfty,                  
                   config_flags config,                      
                   int ids, int ide, int jds, int jde, int kds, int kde,            
                   int ims, int ime, int jms, int jme, int kms, int kme,     
                   int its, int ite, int jts, int jte, int kts, int kte )
{
    size_t idim = ime-ims+1;
    size_t kdim = kme-kms+1;
    size_t jdim = jme-jms+1;

    dim3 Griddim(idim/BLOCKSIZE+1, jdim, 1);
    dim3 Blockdim(BLOCKSIZE, 1, 1);

    ide = ide - ids + 1;
    jde = jde - jds + 1;
    kde = kde - kds + 1; 
    ite = ite - its + 1; // i_size
    jte = jte - jts + 1; // j_size
    
    ids = ids - ims;
    jds = jds - jms;
    kds = kds - kms;
    its = its - ims; // i_start index in memory
    jts = jts - jms; // j_start index in memory
    
    ide = ide + ids - 1;
    jde = jde + jds - 1;
    kde = kde + kds - 1;
    ite = ite + its - 1; // i_end index in memory
    jte = jte + jts - 1; // j_end index in memory

    kte = kte - kts; 
    kts = 0;

    kme = kme - kms; 
    kms = 0;


    if((jts!=jds || jte != jde)){
      printf("jts!=jds || jte != jde\n");
      exit(1);
    }

// domain decomposition is performed equally in 'j' dimension 
// each GPU gets one 'j' row both before and after the output domain
    int start_address_2d[GPUs]; // 2d data's starting address on CPU for a GPU
    int start_address[GPUs]; // 3d data
    int start_address_2d_output[GPUs];
    int start_address_output[GPUs]; // output starting address doesn't need extra 'j' row
    int d_start_address_2d_output[GPUs];
    int d_start_address_output[GPUs];

    int num_rows[GPUs];
    int num_rows_output[GPUs];

    int jds_g[GPUs];
    int jde_g[GPUs];
    int jts_g[GPUs];
    int jte_g[GPUs];

    int total_rows = jme-jms+1;
    int acc_rows = 0; // accumulator for # of rows

    int j;
    if(GPUs == 1){
      num_rows[0] = total_rows;
      jds_g[0] = jds;
      jts_g[0] = jts; 
      jde_g[0] = jde;
      jte_g[0] = jte; 
      start_address[0] = 0;
      start_address_2d[0] = 0;
      start_address_output[0] = 0;
      start_address_2d_output[0] = 0;
      num_rows_output[0] = num_rows[0];
    }
    else
    for(j=0; j<GPUs; j++){
        if(j<GPUs-1){
          num_rows[j] = total_rows / GPUs;
        }
        else{
          num_rows[j] = total_rows - acc_rows;
        }

        if(j == 0){
          jds_g[j] = jds; // jts == jds -> set_physical_bc2d() is OK
          jts_g[j] = jts; 
          start_address[j] = 0;
          start_address_2d[j] = 0; 
        }
        else{
          jds_g[j] = 0; // if jds <=jts-1 then coriolis(), max( jds+1, jts ) is OK
          jts_g[j] = 3;
          start_address[j] = (acc_rows-3) * idim * kdim; // transfer three rows before 'jts'
          start_address_2d[j] = (acc_rows-3) * idim;
        }

        start_address_output[j] = (acc_rows) * idim * kdim;
        start_address_2d_output[j] = (acc_rows) * idim;
        acc_rows += num_rows[j];
        num_rows_output[j] = num_rows[j];

        if(j == 0){
          jde_g[j]=(num_rows[j]+3); // +3: three input row afters the first output row
          jte_g[j]=(num_rows[j]); // if jde >= jte+2 then min( jde-2, jte ) in coriolis() is OK
          num_rows[j] += 3;
        }
        else if(j == GPUs-1){
          jde_g[j]=(num_rows[j]+3); // +3: three input rows before the first output row
          jte_g[j]=(num_rows[j]+3); // jte == jde -> set_physical_bc2d() is OK
          num_rows[j] += 3;
        }
        else{
          jde_g[j]=(num_rows[j]+6); // +6: three input rows before the first output row, three input rows after the last output row
          jte_g[j]=(num_rows[j]+3) ; 
          num_rows[j] += 6; // three input rows before the first output row, three input rows after the last output row
        }
    }

#ifdef COALESCED
    size_t pitch_f;
#endif
    float *d_u[GPUs], *d_u_1[GPUs], *d_v[GPUs], *d_v_1[GPUs]; 
	float *d_t_1[GPUs], *d_ft[GPUs];
	float *d_ww[GPUs], *d_ww_1[GPUs], *d_t[GPUs], *d_t_ave[GPUs]; 
    float *d_mut[GPUs], *d_muu[GPUs], *d_muv[GPUs], *d_mu_tend[GPUs], *d_msfuy[GPUs]; 
    float *d_msfvx_inv[GPUs], *d_msftx[GPUs], *d_msfty[GPUs];
    float *d_mu[GPUs];
    float *d_muave[GPUs], *d_muts[GPUs], *d_mudf[GPUs];    
	float *d_dnw[GPUs], *d_fnm[GPUs], *d_fnp[GPUs], *d_rdnw[GPUs];
    float *d_wdtn[GPUs], *d_dvdxi[GPUs];
    float *d_dmdt[GPUs];	
	
    for(j=0; j<GPUs; j++){
        HANDLE_ERROR( cudaSetDevice(dev_id[j]) );
        HANDLE_ERROR( cudaDeviceSetCacheConfig(cudaFuncCachePreferL1) );

#ifdef COALESCED
        HANDLE_ERROR( cudaMallocPitch(&d_u[j]    , &pitch_f, idim * sizeof(float), kdim * num_rows[j]) );
        HANDLE_ERROR( cudaMallocPitch(&d_u_1[j]  , &pitch_f, idim * sizeof(float), kdim * num_rows[j]) );
        HANDLE_ERROR( cudaMallocPitch(&d_v[j]    , &pitch_f, idim * sizeof(float), kdim * num_rows[j]) );
        HANDLE_ERROR( cudaMallocPitch(&d_v_1[j]  , &pitch_f, idim * sizeof(float), kdim * num_rows[j]) );
        HANDLE_ERROR( cudaMallocPitch(&d_t_1[j]  , &pitch_f, idim * sizeof(float), kdim * num_rows[j]) );
        HANDLE_ERROR( cudaMallocPitch(&d_ft[j]   , &pitch_f, idim * sizeof(float), kdim * num_rows[j]) );
        HANDLE_ERROR( cudaMallocPitch(&d_ww[j]   , &pitch_f, idim * sizeof(float), kdim * num_rows[j]) );
        HANDLE_ERROR( cudaMallocPitch(&d_ww_1[j] , &pitch_f, idim * sizeof(float), kdim * num_rows[j]) );
        HANDLE_ERROR( cudaMallocPitch(&d_t[j]    , &pitch_f, idim * sizeof(float), kdim * num_rows[j]) );
        HANDLE_ERROR( cudaMallocPitch(&d_t_ave[j], &pitch_f, idim * sizeof(float), kdim * num_rows[j]) );
        HANDLE_ERROR( cudaMallocPitch(&d_dvdxi[j], &pitch_f, idim * sizeof(float), kdim * num_rows[j]) );
        HANDLE_ERROR( cudaMallocPitch(&d_wdtn[j] , &pitch_f, idim * sizeof(float), kdim * num_rows[j]) );
       
        HANDLE_ERROR( cudaMallocPitch(&d_mut[j]      , &pitch_f, idim * sizeof(float), num_rows[j]) );
        HANDLE_ERROR( cudaMallocPitch(&d_muu[j]      , &pitch_f, idim * sizeof(float), num_rows[j]) );
        HANDLE_ERROR( cudaMallocPitch(&d_muv[j]      , &pitch_f, idim * sizeof(float), num_rows[j]) );
        HANDLE_ERROR( cudaMallocPitch(&d_mu_tend[j]  , &pitch_f, idim * sizeof(float), num_rows[j]) );
        HANDLE_ERROR( cudaMallocPitch(&d_msfuy[j]    , &pitch_f, idim * sizeof(float), num_rows[j]) );
        HANDLE_ERROR( cudaMallocPitch(&d_msfvx_inv[j], &pitch_f, idim * sizeof(float), num_rows[j]) );
        HANDLE_ERROR( cudaMallocPitch(&d_msftx[j]    , &pitch_f, idim * sizeof(float), num_rows[j]) );
        HANDLE_ERROR( cudaMallocPitch(&d_msfty[j]    , &pitch_f, idim * sizeof(float), num_rows[j]) );
        HANDLE_ERROR( cudaMallocPitch(&d_mu[j]       , &pitch_f, idim * sizeof(float), num_rows[j]) );
        HANDLE_ERROR( cudaMallocPitch(&d_muave[j]    , &pitch_f, idim * sizeof(float), num_rows[j]) );
        HANDLE_ERROR( cudaMallocPitch(&d_muts[j]     , &pitch_f, idim * sizeof(float), num_rows[j]) );
        HANDLE_ERROR( cudaMallocPitch(&d_mudf[j]     , &pitch_f, idim * sizeof(float), num_rows[j]) );
        HANDLE_ERROR( cudaMallocPitch(&d_dmdt[j]     , &pitch_f, idim * sizeof(float), num_rows[j]) );

#else
        HANDLE_ERROR( cudaMalloc(&d_u[j]    , idim * sizeof(float) * kdim * num_rows[j]) );
        HANDLE_ERROR( cudaMalloc(&d_u_1[j]  , idim * sizeof(float) * kdim * num_rows[j]) );
        HANDLE_ERROR( cudaMalloc(&d_v[j]    , idim * sizeof(float) * kdim * num_rows[j]) );
        HANDLE_ERROR( cudaMalloc(&d_v_1[j]  , idim * sizeof(float) * kdim * num_rows[j]) );
        HANDLE_ERROR( cudaMalloc(&d_t_1[j]  , idim * sizeof(float) * kdim * num_rows[j]) );
        HANDLE_ERROR( cudaMalloc(&d_ft[j]   , idim * sizeof(float) * kdim * num_rows[j]) );
        HANDLE_ERROR( cudaMalloc(&d_ww[j]   , idim * sizeof(float) * kdim * num_rows[j]) );
        HANDLE_ERROR( cudaMalloc(&d_ww_1[j] , idim * sizeof(float) * kdim * num_rows[j]) );
        HANDLE_ERROR( cudaMalloc(&d_t[j]    , idim * sizeof(float) * kdim * num_rows[j]) );
        HANDLE_ERROR( cudaMalloc(&d_t_ave[j], idim * sizeof(float) * kdim * num_rows[j]) );
        HANDLE_ERROR( cudaMalloc(&d_dvdxi[j], idim * sizeof(float) * kdim * num_rows[j]) );
        HANDLE_ERROR( cudaMalloc(&d_wdtn[j] , idim * sizeof(float) * kdim * num_rows[j]) );
       
        HANDLE_ERROR( cudaMalloc(&d_mut[j]      , idim * sizeof(float) * num_rows[j]) );
        HANDLE_ERROR( cudaMalloc(&d_muu[j]      , idim * sizeof(float) * num_rows[j]) );
        HANDLE_ERROR( cudaMalloc(&d_muv[j]      , idim * sizeof(float) * num_rows[j]) );
        HANDLE_ERROR( cudaMalloc(&d_mu_tend[j]  , idim * sizeof(float) * num_rows[j]) );
        HANDLE_ERROR( cudaMalloc(&d_msfuy[j]    , idim * sizeof(float) * num_rows[j]) );
        HANDLE_ERROR( cudaMalloc(&d_msfvx_inv[j], idim * sizeof(float) * num_rows[j]) );
        HANDLE_ERROR( cudaMalloc(&d_msftx[j]    , idim * sizeof(float) * num_rows[j]) );
        HANDLE_ERROR( cudaMalloc(&d_msfty[j]    , idim * sizeof(float) * num_rows[j]) );
        HANDLE_ERROR( cudaMalloc(&d_mu[j]       , idim * sizeof(float) * num_rows[j]) );
        HANDLE_ERROR( cudaMalloc(&d_muave[j]    , idim * sizeof(float) * num_rows[j]) );
        HANDLE_ERROR( cudaMalloc(&d_muts[j]     , idim * sizeof(float) * num_rows[j]) );
        HANDLE_ERROR( cudaMalloc(&d_mudf[j]     , idim * sizeof(float) * num_rows[j]) );
        HANDLE_ERROR( cudaMalloc(&d_dmdt[j]     , idim * sizeof(float) * num_rows[j]) );
        
#endif
        HANDLE_ERROR( cudaMalloc(&d_dnw[j] , kdim * sizeof(float)) );
        HANDLE_ERROR( cudaMalloc(&d_fnm[j] , kdim * sizeof(float)) );
        HANDLE_ERROR( cudaMalloc(&d_fnp[j] , kdim * sizeof(float)) );
        HANDLE_ERROR( cudaMalloc(&d_rdnw[j], kdim * sizeof(float)) );
         
    }
    for(j=0; j<GPUs; j++){
        HANDLE_ERROR( cudaSetDevice(dev_id[j]) );
    // copy input from CPU to GPU
#ifdef COALESCED

        HANDLE_ERROR( cudaMemcpy2D(d_u[j]    , pitch_f, &u[start_address[j]]    , idim * sizeof(float), idim * sizeof(float), kdim * num_rows[j], cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy2D(d_u_1[j]  , pitch_f, &u_1[start_address[j]]  , idim * sizeof(float), idim * sizeof(float), kdim * num_rows[j], cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy2D(d_v[j]    , pitch_f, &v[start_address[j]]    , idim * sizeof(float), idim * sizeof(float), kdim * num_rows[j], cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy2D(d_v_1[j]  , pitch_f, &v_1[start_address[j]]  , idim * sizeof(float), idim * sizeof(float), kdim * num_rows[j], cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy2D(d_t_1[j]  , pitch_f, &t_1[start_address[j]]  , idim * sizeof(float), idim * sizeof(float), kdim * num_rows[j], cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy2D(d_ft[j]   , pitch_f, &ft[start_address[j]]   , idim * sizeof(float), idim * sizeof(float), kdim * num_rows[j], cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy2D(d_ww[j]   , pitch_f, &ww[start_address[j]]   , idim * sizeof(float), idim * sizeof(float), kdim * num_rows[j], cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy2D(d_ww_1[j] , pitch_f, &ww_1[start_address[j]] , idim * sizeof(float), idim * sizeof(float), kdim * num_rows[j], cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy2D(d_t[j]    , pitch_f, &t[start_address[j]]    , idim * sizeof(float), idim * sizeof(float), kdim * num_rows[j], cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy2D(d_t_ave[j], pitch_f, &t_ave[start_address[j]], idim * sizeof(float), idim * sizeof(float), kdim * num_rows[j], cudaMemcpyHostToDevice) );
       
        HANDLE_ERROR( cudaMemcpy2D(d_mut[j]      , pitch_f, &mut[start_address_2d[j]]      , idim * sizeof(float), idim * sizeof(float), num_rows[j], cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy2D(d_muu[j]      , pitch_f, &muu[start_address_2d[j]]      , idim * sizeof(float), idim * sizeof(float), num_rows[j], cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy2D(d_muv[j]      , pitch_f, &muv[start_address_2d[j]]      , idim * sizeof(float), idim * sizeof(float), num_rows[j], cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy2D(d_mu_tend[j]  , pitch_f, &mu_tend[start_address_2d[j]]  , idim * sizeof(float), idim * sizeof(float), num_rows[j], cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy2D(d_msfuy[j]    , pitch_f, &msfuy[start_address_2d[j]]    , idim * sizeof(float), idim * sizeof(float), num_rows[j], cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy2D(d_msfvx_inv[j], pitch_f, &msfvx_inv[start_address_2d[j]], idim * sizeof(float), idim * sizeof(float), num_rows[j], cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy2D(d_msftx[j]    , pitch_f, &msftx[start_address_2d[j]]    , idim * sizeof(float), idim * sizeof(float), num_rows[j], cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy2D(d_msfty[j]    , pitch_f, &msfty[start_address_2d[j]]    , idim * sizeof(float), idim * sizeof(float), num_rows[j], cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy2D(d_mu[j]       , pitch_f, &mu[start_address_2d[j]]       , idim * sizeof(float), idim * sizeof(float), num_rows[j], cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy2D(d_muave[j]    , pitch_f, &muave[start_address_2d[j]]    , idim * sizeof(float), idim * sizeof(float), num_rows[j], cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy2D(d_muts[j]     , pitch_f, &muts[start_address_2d[j]]     , idim * sizeof(float), idim * sizeof(float), num_rows[j], cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy2D(d_mudf[j]     , pitch_f, &mudf[start_address_2d[j]]     , idim * sizeof(float), idim * sizeof(float), num_rows[j], cudaMemcpyHostToDevice) );

#else

        HANDLE_ERROR( cudaMemcpy(d_u[j]    , &u[start_address[j]]    , idim * sizeof(float) * kdim * num_rows[j], cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy(d_u_1[j]  , &u_1[start_address[j]]  , idim * sizeof(float) * kdim * num_rows[j], cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy(d_v[j]    , &v[start_address[j]]    , idim * sizeof(float) * kdim * num_rows[j], cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy(d_v_1[j]  , &v_1[start_address[j]]  , idim * sizeof(float) * kdim * num_rows[j], cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy(d_t_1[j]  , &t_1[start_address[j]]  , idim * sizeof(float) * kdim * num_rows[j], cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy(d_ft[j]   , &ft[start_address[j]]   , idim * sizeof(float) * kdim * num_rows[j], cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy(d_ww[j]   , &ww[start_address[j]]   , idim * sizeof(float) * kdim * num_rows[j], cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy(d_ww_1[j] , &ww_1[start_address[j]] , idim * sizeof(float) * kdim * num_rows[j], cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy(d_t[j]    , &t[start_address[j]]    , idim * sizeof(float) * kdim * num_rows[j], cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy(d_t_ave[j], &t_ave[start_address[j]], idim * sizeof(float) * kdim * num_rows[j], cudaMemcpyHostToDevice) );
       
        HANDLE_ERROR( cudaMemcpy(d_mut[j]      , &mut[start_address_2d[j]]      , idim * sizeof(float) * num_rows[j], cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy(d_muu[j]      , &muu[start_address_2d[j]]      , idim * sizeof(float) * num_rows[j], cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy(d_muv[j]      , &muv[start_address_2d[j]]      , idim * sizeof(float) * num_rows[j], cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy(d_mu_tend[j]  , &mu_tend[start_address_2d[j]]  , idim * sizeof(float) * num_rows[j], cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy(d_msfuy[j]    , &msfuy[start_address_2d[j]]    , idim * sizeof(float) * num_rows[j], cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy(d_msfvx_inv[j], &msfvx_inv[start_address_2d[j]], idim * sizeof(float) * num_rows[j], cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy(d_msftx[j]    , &msftx[start_address_2d[j]]    , idim * sizeof(float) * num_rows[j], cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy(d_msfty[j]    , &msfty[start_address_2d[j]]    , idim * sizeof(float) * num_rows[j], cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy(d_mu[j]       , &mu[start_address_2d[j]]       , idim * sizeof(float) * num_rows[j], cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy(d_muave[j]    , &muave[start_address_2d[j]]    , idim * sizeof(float) * num_rows[j], cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy(d_muts[j]     , &muts[start_address_2d[j]]     , idim * sizeof(float) * num_rows[j], cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy(d_mudf[j]     , &mudf[start_address_2d[j]]     , idim * sizeof(float) * num_rows[j], cudaMemcpyHostToDevice) );

#endif
        HANDLE_ERROR( cudaMemcpy(d_dnw[j] , dnw , kdim * sizeof(float), cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy(d_fnm[j] , fnm , kdim * sizeof(float), cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy(d_fnp[j] , fnp , kdim * sizeof(float), cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy(d_rdnw[j], rdnw, kdim * sizeof(float), cudaMemcpyHostToDevice) );
    
    }

    for(j=0; j<GPUs; j++){
        if(j == 0){
          d_start_address_output[j] = 0;        
          d_start_address_2d_output[j] = 0;
        }
        else{
#ifdef COALESCED
          d_start_address_output[j] = 3 * pitch_f / sizeof(float) * kdim; // skip first three rows
          d_start_address_2d_output[j] = 3 * pitch_f / sizeof(float);
#else
          d_start_address_output[j] = 3 * idim * kdim;        
          d_start_address_2d_output[j] = 3 * idim;
#endif
        }
    }

    struct timeval ta, tb;
    long mseca, msecb;
    gettimeofday( &ta, NULL );
    mseca = ta.tv_sec * 1000000 + ta.tv_usec;
    
    for(j=0; j<GPUs; j++){
        HANDLE_ERROR( cudaSetDevice(dev_id[j]) );
     
        advance_mu_t_kernel<<<Griddim, Blockdim>>>
                                    (d_ww[j], d_ww_1[j], d_u[j], d_u_1[j],
                                     d_v[j], d_v_1[j],
                                     d_mu[j], d_mut[j], d_muave[j], d_muts[j],
                                     d_muu[j], d_muv[j],
                                     d_mudf[j], d_t[j], d_t_1[j],
                                     d_t_ave[j], d_ft[j], d_mu_tend[j],
                                     rdx, rdy, dts, epssm,
                                     d_dnw[j], d_fnm[j], d_fnp[j], d_rdnw[j],
                                     d_msfuy[j], d_msfvx_inv[j],
                                     d_msftx[j], d_msfty[j],
                                     d_wdtn[j], d_dvdxi[j], d_dmdt[j],
                                     config,
                                     ids, ide, jds_g[j], jde_g[j], kds, kde,
#ifdef COALESCED
                                     pitch_f / sizeof(float), jdim, kdim,
#else
                                     idim, jdim, kdim,
#endif
                                     its, ite, jts_g[j], jte_g[j],
                                     kts,kte);
    }
    for(j=0; j<GPUs; j++){
        HANDLE_ERROR( cudaSetDevice(dev_id[j]) );
        HANDLE_ERROR( cudaThreadSynchronize() );
    }


    gettimeofday( &tb, NULL );
    msecb = tb.tv_sec * 1000000 + tb.tv_usec;
    msecb -=mseca;
    printf("advance_mu_t GPU time is\t%.3f ms\n", (float)msecb/1000);

  // copy output from GPU to CPU
    for(j=0; j<GPUs; j++){
        HANDLE_ERROR( cudaSetDevice(dev_id[j]) );

#ifdef COALESCED
        HANDLE_ERROR( cudaMemcpy2D(&ww[start_address_output[j]]   , idim * sizeof(float), &d_ww[j][d_start_address_output[j]]   , pitch_f, idim * sizeof(float), kdim * num_rows_output[j], cudaMemcpyDeviceToHost) );
        HANDLE_ERROR( cudaMemcpy2D(&ww_1[start_address_output[j]] , idim * sizeof(float), &d_ww_1[j][d_start_address_output[j]] , pitch_f, idim * sizeof(float), kdim * num_rows_output[j], cudaMemcpyDeviceToHost) );
        HANDLE_ERROR( cudaMemcpy2D(&t[start_address_output[j]]    , idim * sizeof(float), &d_t[j][d_start_address_output[j]]      , pitch_f, idim * sizeof(float), kdim * num_rows_output[j], cudaMemcpyDeviceToHost) );
        HANDLE_ERROR( cudaMemcpy2D(&t_ave[start_address_output[j]], idim * sizeof(float), &d_t_ave[j][d_start_address_output[j]], pitch_f, idim * sizeof(float), kdim * num_rows_output[j], cudaMemcpyDeviceToHost) );
        
        HANDLE_ERROR( cudaMemcpy2D(&mu[start_address_2d_output[j]]   , idim * sizeof(float), &d_mu[j][d_start_address_2d_output[j]]   , pitch_f, idim * sizeof(float), num_rows_output[j], cudaMemcpyDeviceToHost) );
        HANDLE_ERROR( cudaMemcpy2D(&muave[start_address_2d_output[j]], idim * sizeof(float), &d_muave[j][d_start_address_2d_output[j]], pitch_f, idim * sizeof(float), num_rows_output[j], cudaMemcpyDeviceToHost) );
        HANDLE_ERROR( cudaMemcpy2D(&muts[start_address_2d_output[j]] , idim * sizeof(float), &d_muts[j][d_start_address_2d_output[j]] , pitch_f, idim * sizeof(float), num_rows_output[j], cudaMemcpyDeviceToHost) );
        HANDLE_ERROR( cudaMemcpy2D(&mudf[start_address_2d_output[j]] , idim * sizeof(float), &d_mudf[j][d_start_address_2d_output[j]] , pitch_f, idim * sizeof(float), num_rows_output[j], cudaMemcpyDeviceToHost) );
#else
        HANDLE_ERROR( cudaMemcpy(&ww[start_address_output[j]]   , &d_ww[j][d_start_address_output[j]]   , idim * sizeof(float) * kdim * num_rows_output[j], cudaMemcpyDeviceToHost) );
        HANDLE_ERROR( cudaMemcpy(&ww_1[start_address_output[j]] , &d_ww_1[j][d_start_address_output[j]] , idim * sizeof(float) * kdim * num_rows_output[j], cudaMemcpyDeviceToHost) );
        HANDLE_ERROR( cudaMemcpy(&t[start_address_output[j]]    , &d_t[j][d_start_address_output[j]]    , idim * sizeof(float) * kdim * num_rows_output[j], cudaMemcpyDeviceToHost) );
        HANDLE_ERROR( cudaMemcpy(&t_ave[start_address_output[j]], &d_t_ave[j][d_start_address_output[j]], idim * sizeof(float) * kdim * num_rows_output[j], cudaMemcpyDeviceToHost) );
        
        HANDLE_ERROR( cudaMemcpy(&mu[start_address_2d_output[j]]   , &d_mu[j][d_start_address_2d_output[j]]   , idim * sizeof(float) * num_rows_output[j], cudaMemcpyDeviceToHost) );
        HANDLE_ERROR( cudaMemcpy(&muave[start_address_2d_output[j]], &d_muave[j][d_start_address_2d_output[j]], idim * sizeof(float) * num_rows_output[j], cudaMemcpyDeviceToHost) );
        HANDLE_ERROR( cudaMemcpy(&muts[start_address_2d_output[j]] , &d_muts[j][d_start_address_2d_output[j]] , idim * sizeof(float) * num_rows_output[j], cudaMemcpyDeviceToHost) );
        HANDLE_ERROR( cudaMemcpy(&mudf[start_address_2d_output[j]] , &d_mudf[j][d_start_address_2d_output[j]] , idim * sizeof(float) * num_rows_output[j], cudaMemcpyDeviceToHost) );
#endif
    }

    for(j=0; j<GPUs; j++){
        HANDLE_ERROR( cudaSetDevice(dev_id[j]) );
        HANDLE_ERROR( cudaFree( d_ww[j] ));
        HANDLE_ERROR( cudaFree( d_ww_1[j] ));
        HANDLE_ERROR( cudaFree( d_u[j] ));
        HANDLE_ERROR( cudaFree( d_u_1[j] ));
        HANDLE_ERROR( cudaFree( d_v[j] ));
        HANDLE_ERROR( cudaFree( d_v_1[j] ));
        HANDLE_ERROR( cudaFree( d_mu[j] ));
        HANDLE_ERROR( cudaFree( d_mut[j] ));
        HANDLE_ERROR( cudaFree( d_muave[j] ));
        HANDLE_ERROR( cudaFree( d_muts[j] ));
        HANDLE_ERROR( cudaFree( d_muu[j] ));
        HANDLE_ERROR( cudaFree( d_muv[j] ));
        HANDLE_ERROR( cudaFree( d_mudf[j] ));
        HANDLE_ERROR( cudaFree( d_t[j] ));
        HANDLE_ERROR( cudaFree( d_t_1[j] ));
        HANDLE_ERROR( cudaFree( d_t_ave[j] ));
        HANDLE_ERROR( cudaFree( d_ft[j] ));
        HANDLE_ERROR( cudaFree( d_mu_tend[j] ));
        HANDLE_ERROR( cudaFree( d_dnw[j] ));
        HANDLE_ERROR( cudaFree( d_fnm[j] ));
        HANDLE_ERROR( cudaFree( d_fnp[j] ));
        HANDLE_ERROR( cudaFree( d_rdnw[j] ));
        HANDLE_ERROR( cudaFree( d_msfuy[j] ));
        HANDLE_ERROR( cudaFree( d_msfvx_inv[j] ));
        HANDLE_ERROR( cudaFree( d_msftx[j] ));
        HANDLE_ERROR( cudaFree( d_msfty[j] ));
        HANDLE_ERROR( cudaFree( d_wdtn[j] ));
        HANDLE_ERROR( cudaFree( d_dvdxi[j] ));
	    HANDLE_ERROR( cudaFree( d_dmdt[j] ));
    }
}


