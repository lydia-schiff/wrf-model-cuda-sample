#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include <float.h>
#include "config_flags.h"
#include "advance_mu_t_cu.h"
#include "common.h"

#define max(a,b) ((a>b) ? (a) : (b))
#define I3(i,k,j) ((j) * kdim * idim + (k) * idim + (i))

char *default_input_data_dir 
      = "/data2/WRFV3_Input_Output/V3.4.1/dyn_em/advance_mu_t/";
char input_data_dir[256];

char *default_output_data_dir
      = "./output/";
char output_data_dir[256];

char *default_compare_data_dir 
      = "/data2/WRFV3_Input_Output/V3.4.1/dyn_em/advance_mu_t/";


static void HandleError(cudaError_t err, const char *file, int line) {
    if(err != cudaSuccess){
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

int main(int argc, char **argv)
{
    if(argc == 2){
        strcpy(input_data_dir, argv[1]);
        strcpy(output_data_dir, default_output_data_dir);
    }
    else {
        if(argc > 2){
            strcpy(input_data_dir, argv[1]);
            strcpy(output_data_dir, argv[2]);
        }
        else {
            strcpy(input_data_dir, default_input_data_dir);
            strcpy(output_data_dir, default_output_data_dir);
        }
    }

    int ids, ide, jds, jde, kds, kde; // domain dims
    int ims, ime, jms, jme, kms, kme; // memory dims
    int its, ite, jts, jte, kts, kte; // tile dims

    read_dim_data(&ids, "ids.bin");
    read_dim_data(&ide, "ide.bin");
    read_dim_data(&jds, "jds.bin");
    read_dim_data(&jde, "jde.bin");
    read_dim_data(&kds, "kds.bin");
    read_dim_data(&kde, "kde.bin");
    
    read_dim_data(&ims, "ims.bin");
    read_dim_data(&ime, "ime.bin");
    read_dim_data(&jms, "jms.bin");
    read_dim_data(&jme, "jme.bin");
    read_dim_data(&kms, "kms.bin");
    read_dim_data(&kme, "kme.bin");
    
    read_dim_data(&its, "its.bin");
    read_dim_data(&ite, "ite.bin");
    read_dim_data(&jts, "jts.bin");
    read_dim_data(&jte, "jte.bin");
    read_dim_data(&kts, "kts.bin");
    read_dim_data(&kte, "kte.bin");
    
    config_flags config;
    
    read_dim_data(&config.nested    , "config_flags_nested.bin"    );
    read_dim_data(&config.periodic_x, "config_flags_periodic_x.bin");
    read_dim_data(&config.specified , "config_flags_specified.bin" );

	float grid_rdx, grid_rdy, dts_rk, grid_epssm;
    
	read_real_data( &grid_rdx  , "grid_rdx.bin"  );
	read_real_data( &grid_rdy  , "grid_rdy.bin"  );
    read_real_data( &dts_rk    , "dts_rk.bin"    );
    read_real_data( &grid_epssm, "grid_epssm.bin");


    //------reshape 1D to nD--------------
    int idim = ime - ims + 1;
    int jdim = jme - jms + 1;
    int kdim = kme - kms + 1;
 // 1d in
	float *grid_dnw, *grid_fnm, *grid_fnp, *grid_rdnw;
	
	HANDLE_ERROR( cudaHostAlloc((void**)&grid_dnw , 1 * kdim * 1 * sizeof(float), cudaHostAllocPortable) );
	HANDLE_ERROR( cudaHostAlloc((void**)&grid_fnm , 1 * kdim * 1 * sizeof(float), cudaHostAllocPortable) );
	HANDLE_ERROR( cudaHostAlloc((void**)&grid_fnp , 1 * kdim * 1 * sizeof(float), cudaHostAllocPortable) );
	HANDLE_ERROR( cudaHostAlloc((void**)&grid_rdnw, 1 * kdim * 1 * sizeof(float), cudaHostAllocPortable) );
	
	read_data( grid_dnw , "grid_dnw.bin" , 1, kdim, 1 );
	read_data( grid_fnm , "grid_fnm.bin" , 1, kdim, 1 );
	read_data( grid_fnp , "grid_fnp.bin" , 1, kdim, 1 );
	read_data( grid_rdnw, "grid_rdnw.bin", 1, kdim, 1 );
    
    // 2d in
    float *grid_mut, *grid_muu, *grid_muv, *mu_tend, *grid_msfuy; 
    float *grid_msfvx_inv, *grid_msftx, *grid_msfty;
	
	HANDLE_ERROR( cudaHostAlloc((void**)&grid_mut      , idim * 1 * jdim * sizeof(float), cudaHostAllocPortable) );
	HANDLE_ERROR( cudaHostAlloc((void**)&grid_muu      , idim * 1 * jdim * sizeof(float), cudaHostAllocPortable) );
	HANDLE_ERROR( cudaHostAlloc((void**)&grid_muv      , idim * 1 * jdim * sizeof(float), cudaHostAllocPortable) );
	HANDLE_ERROR( cudaHostAlloc((void**)&mu_tend       , idim * 1 * jdim * sizeof(float), cudaHostAllocPortable) );
	HANDLE_ERROR( cudaHostAlloc((void**)&grid_msfuy    , idim * 1 * jdim * sizeof(float), cudaHostAllocPortable) );
	HANDLE_ERROR( cudaHostAlloc((void**)&grid_msfvx_inv, idim * 1 * jdim * sizeof(float), cudaHostAllocPortable) );
	HANDLE_ERROR( cudaHostAlloc((void**)&grid_msftx    , idim * 1 * jdim * sizeof(float), cudaHostAllocPortable) );
	HANDLE_ERROR( cudaHostAlloc((void**)&grid_msfty    , idim * 1 * jdim * sizeof(float), cudaHostAllocPortable) );	
		
	read_data( grid_mut      , "grid_mut.bin"      , idim, 1, jdim );
    read_data( grid_muu      , "grid_muu.bin"      , idim, 1, jdim );
    read_data( grid_muv      , "grid_muv.bin"      , idim, 1, jdim );
    read_data( mu_tend       , "mu_tend.bin"       , idim, 1, jdim );
    read_data( grid_msfuy    , "grid_msfuy.bin"    , idim, 1, jdim );
    read_data( grid_msfvx_inv, "grid_msfvx_inv.bin", idim, 1, jdim );
    read_data( grid_msfty    , "grid_msfty.bin"    , idim, 1, jdim );
    read_data( grid_msftx    , "grid_msftx.bin"    , idim, 1, jdim );
    
    // 2d in/out
    float *grid_mu_2;
    
    HANDLE_ERROR( cudaHostAlloc((void**)&grid_mu_2, idim * 1 * jdim * sizeof(float), cudaHostAllocPortable) );
    
    read_data( grid_mu_2, "grid_mu_2.bin", idim, 1, jdim );
    
    // 2d out
    float *muave, *grid_muts, *grid_mudf;
    
    HANDLE_ERROR( cudaHostAlloc((void**)&muave    , idim * 1 * jdim * sizeof(float), cudaHostAllocPortable) );
    HANDLE_ERROR( cudaHostAlloc((void**)&grid_muts, idim * 1 * jdim * sizeof(float), cudaHostAllocPortable) );
    HANDLE_ERROR( cudaHostAlloc((void**)&grid_mudf, idim * 1 * jdim * sizeof(float), cudaHostAllocPortable) );

    // 3d in 
	float *grid_u_2, *grid_u_save, *grid_v_2, *grid_v_save; 
	float *grid_t_save, *t_tend;
	
	HANDLE_ERROR( cudaHostAlloc((void**)&grid_u_2   , idim * kdim * jdim * sizeof(float), cudaHostAllocPortable) );
	HANDLE_ERROR( cudaHostAlloc((void**)&grid_u_save, idim * kdim * jdim * sizeof(float), cudaHostAllocPortable) );
	HANDLE_ERROR( cudaHostAlloc((void**)&grid_v_2   , idim * kdim * jdim * sizeof(float), cudaHostAllocPortable) );
	HANDLE_ERROR( cudaHostAlloc((void**)&grid_v_save, idim * kdim * jdim * sizeof(float), cudaHostAllocPortable) );
	HANDLE_ERROR( cudaHostAlloc((void**)&grid_t_save, idim * kdim * jdim * sizeof(float), cudaHostAllocPortable) );
	HANDLE_ERROR( cudaHostAlloc((void**)&t_tend     , idim * kdim * jdim * sizeof(float), cudaHostAllocPortable) );
	
	read_data( grid_u_2   , "grid_u_2.bin"   , idim, kdim, jdim );
	read_data( grid_u_save, "grid_u_save.bin", idim, kdim, jdim );
	read_data( grid_v_2   , "grid_v_2.bin"   , idim, kdim, jdim );
	read_data( grid_v_save, "grid_v_save.bin", idim, kdim, jdim );
	read_data( grid_t_save, "grid_t_save.bin", idim, kdim, jdim );
    read_data( t_tend     , "t_tend.bin"     , idim, kdim, jdim );
    
    // 3d in/out 
	float *grid_ww, *ww1, *grid_t_2, *t_2save; 
	
	HANDLE_ERROR( cudaHostAlloc((void**)&grid_ww , idim * kdim * jdim * sizeof(float), cudaHostAllocPortable) );
	HANDLE_ERROR( cudaHostAlloc((void**)&ww1     , idim * kdim * jdim * sizeof(float), cudaHostAllocPortable) );
	HANDLE_ERROR( cudaHostAlloc((void**)&grid_t_2, idim * kdim * jdim * sizeof(float), cudaHostAllocPortable) );
	HANDLE_ERROR( cudaHostAlloc((void**)&t_2save , idim * kdim * jdim * sizeof(float), cudaHostAllocPortable) );
	
	read_data( grid_ww , "grid_ww.bin" , idim, kdim, jdim );
	read_data( ww1     , "ww1.bin"     , idim, kdim, jdim );
	read_data( grid_t_2, "grid_t_2.bin", idim, kdim, jdim );
	read_data( t_2save , "t_2save.bin" , idim, kdim, jdim );
	

    advance_mu_t( grid_ww, ww1, grid_u_2, grid_u_save, grid_v_2, grid_v_save,
                  grid_mu_2, grid_mut, muave, grid_muts, grid_muu, grid_muv,
                  grid_mudf,                                               
                  grid_t_2, grid_t_save, t_2save, t_tend,                 
                  mu_tend,                                                   
                  grid_rdx, grid_rdy, dts_rk, grid_epssm,                    
                  grid_dnw, grid_fnm, grid_fnp, grid_rdnw,                    
                  grid_msfuy, grid_msfvx_inv,                                
                  grid_msftx, grid_msfty,                                   
                  config,                                           
                  ids, ide, jds, jde, kds, kde,                               
                  ims, ime, jms, jme, kms, kme,                        
                  its, ite, jts, jte, kts, kte );
                  

    int i_start = max(its,ids+1);
    int i_end   = min(ite,ide-2);
    int j_start = max(jts,jds+1);
    int j_end   = min(jte,jde-2);

    compare(grid_ww , "grid_ww_output.bin" , ims, ime, kms, kme, jms, jme, its, ite, kts, kte, jts, jte-2); 
    compare(ww1     , "ww1_output.bin"     , ims, ime, kms, kme, jms, jme, its, ite, kts, kte, jts, jte-2);
    compare(grid_t_2, "grid_t_2_output.bin", ims, ime, kms, kme, jms, jme, its, ite, kts, kte, jts, jte-2);
    compare(t_2save , "t_2save_output.bin" , ims, ime, kms, kme, jms, jme, its, ite, kts, kte, jts, jte-2);
    
	compare(grid_mu_2, "grid_mu_2_output.bin", ims, ime, 1, 1, jms, jme, its, ite, 1, 1, jts, jte-2);
    compare(muave    , "muave_output.bin"    , ims, ime, 1, 1, jms, jme, i_start, i_end, 1, 1, j_start, j_end);
    compare(grid_muts, "grid_muts_output.bin", ims, ime, 1, 1, jms, jme, i_start, i_end, 1, 1, j_start, j_end);
    compare(grid_mudf, "grid_mudf_output.bin", ims, ime, 1, 1, jms, jme, i_start, i_end, 1, 1, j_start, j_end);

    cudaFreeHost( grid_ww );
    cudaFreeHost( ww1 );
    cudaFreeHost( grid_u_2 );
    cudaFreeHost( grid_u_save );
    cudaFreeHost( grid_v_2 );
    cudaFreeHost( grid_v_save );
    cudaFreeHost( grid_mu_2 );
    cudaFreeHost( grid_mut );
    cudaFreeHost( muave );
    cudaFreeHost( grid_muts );
    cudaFreeHost( grid_muu );
    cudaFreeHost( grid_muv );
    cudaFreeHost( grid_mudf );
    cudaFreeHost( grid_t_2 );
    cudaFreeHost( grid_t_save );
    cudaFreeHost( t_2save );
    cudaFreeHost( t_tend );
    cudaFreeHost( mu_tend );
    cudaFreeHost( grid_dnw );
    cudaFreeHost( grid_fnm );
    cudaFreeHost( grid_fnp );
    cudaFreeHost( grid_rdnw );
    cudaFreeHost( grid_msfuy );
    cudaFreeHost( grid_msfvx_inv );
    cudaFreeHost( grid_msftx );
    cudaFreeHost( grid_msfty );

    return 0;   
}	

