#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include "advance_mu_t.h"
#include <float.h>
#include "ulps.h"

#define max(A,B) ((A>B) ? (A) : (B))
#define min(a,b)  ((a)<(b)?(a):(b))
#define I3(i,k,j) ((j) * kdim * idim + (k) * idim + (i))
            

char *default_input_data_dir 
      = "/data2/WRFV3_Input_Output/V3.4.1/dyn_em/advance_mu_t/";
char input_data_dir[256];

char *default_output_data_dir
      = "./output/";
char output_data_dir[256];

char *default_compare_data_dir 
      = "/data2/WRFV3_Input_Output/V3.4.1/dyn_em/advance_mu_t/";

void read_dim_data(int *data, char *file_name);
void read_data(float *data, char *file_name, int x, int y, int z);
void read_data_4d(float *data, char *file_name, int I, int K, int J, int S);
void read_real_data(float *data, char *file_name);
void compare(float *data, char *file_name, int size);
void compare_2d_t(float *data, char *file_name, int i_start, int i_end, 
                  int j_start, int j_end, int idim, int jdim           );
int float_ulps(float a, float b);



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

    int ids, ide, jds, jde, kds,  kde; // domain dims
    int ims, ime, jms, jme, kms, kme; // memory dims
    int its, ite, jts, jte, kts, kte; // tile dims
    int i_start, i_end, j_start, j_end;


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
    
    size_t idim = ime - ims + 1;
    size_t kdim = kme - kms + 1;
    size_t jdim = jme - jms + 1;
    
    int ids_orig = ids;
    int ide_orig = ide;
    int jds_orig = jds;
    int jde_orig = jde;
    int ite_orig = ite;
    int jte_orig = jte;
    int its_orig = its;
    int jts_orig = jts;
    
    ide = ide - ids + 1;
    jde = jde - jds + 1;
    ite = ite - its + 1; // i_size
    jte = jte - jts + 1; // j_size

    ids = ids - ims;
    jds = jds - jms;
    its = its - ims; // i_start index in memory
    jts = jts - jms; // j_start index in memory

    ide = ide + ids - 1;
    jde = jde + jds - 1;
    ite = ite + its - 1; // i_end index in memory
    jte = jte + jts - 1; // j_end index in memory
    
    i_start = max(its,ids+1);
    i_end   = min(ite,ide-2);
    j_start = max(jts,jds+1);
    j_end   = min(jte,jde-2);
    
    ids = ids_orig;
    ide = ide_orig;
    jds = jds_orig;
    jde = jde_orig;
    jte = jte_orig;
    ite = ite_orig;
    its = its_orig;
    jts = jts_orig;

    

	float grid_rdx, grid_rdy, dts_rk, grid_epssm;
    
	read_real_data( &grid_rdx  , "grid_rdx.bin"  );
	read_real_data( &grid_rdy  , "grid_rdy.bin"  );
    read_real_data( &dts_rk    , "dts_rk.bin"    );
    read_real_data( &grid_epssm, "grid_epssm.bin");
    
    
    config_flags config;
	
	read_dim_data(&config.nested    , "config_flags_nested.bin"    );
	read_dim_data(&config.periodic_x, "config_flags_periodic_x.bin");
	read_dim_data(&config.specified , "config_flags_specified.bin" );


    
    // 1d in
	float *grid_dnw, *grid_fnm, *grid_fnp, *grid_rdnw;
	
	grid_dnw  = (float *)alloc_mem(1 * kdim * 1 * sizeof(float));
	grid_fnm  = (float *)alloc_mem(1 * kdim * 1 * sizeof(float));
	grid_fnp  = (float *)alloc_mem(1 * kdim * 1 * sizeof(float));
	grid_rdnw = (float *)alloc_mem(1 * kdim * 1 * sizeof(float));
	
	read_data( grid_dnw , "grid_dnw.bin" , 1, kdim, 1 );
	read_data( grid_fnm , "grid_fnm.bin" , 1, kdim, 1 );
	read_data( grid_fnp , "grid_fnp.bin" , 1, kdim, 1 );
	read_data( grid_rdnw, "grid_rdnw.bin", 1, kdim, 1 );
    
    // 2d in
    float *grid_mut, *grid_muu, *grid_muv, *mu_tend, *grid_msfuy; 
    float *grid_msfvx_inv, *grid_msftx, *grid_msfty;
	
	grid_mut       = (float *)alloc_mem(idim * 1 * jdim * sizeof(float));
	grid_muu       = (float *)alloc_mem(idim * 1 * jdim * sizeof(float));
	grid_muv       = (float *)alloc_mem(idim * 1 * jdim * sizeof(float));
	mu_tend        = (float *)alloc_mem(idim * 1 * jdim * sizeof(float));
	grid_msfuy     = (float *)alloc_mem(idim * 1 * jdim * sizeof(float));
	grid_msfvx_inv = (float *)alloc_mem(idim * 1 * jdim * sizeof(float));
	grid_msftx     = (float *)alloc_mem(idim * 1 * jdim * sizeof(float));
	grid_msfty     = (float *)alloc_mem(idim * 1 * jdim * sizeof(float));	
		
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
    
    grid_mu_2 = (float *)alloc_mem(idim * 1 * jdim * sizeof(float));
    
    read_data( grid_mu_2, "grid_mu_2.bin", idim, 1, jdim );
    
    // 2d out
    float *muave, *grid_muts, *grid_mudf;
    
    muave     = (float *)alloc_mem(idim * 1 * jdim * sizeof(float));
    grid_muts = (float *)alloc_mem(idim * 1 * jdim * sizeof(float));
    grid_mudf = (float *)alloc_mem(idim * 1 * jdim * sizeof(float));

    // 3d in 
	float *grid_u_2, *grid_u_save, *grid_v_2, *grid_v_save; 
	float *grid_t_save, *t_tend;
	
	grid_u_2    = (float *)alloc_mem(idim * kdim * jdim * sizeof(float));
	grid_u_save = (float *)alloc_mem(idim * kdim * jdim * sizeof(float));
	grid_v_2    = (float *)alloc_mem(idim * kdim * jdim * sizeof(float));
	grid_v_save = (float *)alloc_mem(idim * kdim * jdim * sizeof(float));
	grid_t_save = (float *)alloc_mem(idim * kdim * jdim * sizeof(float));
	t_tend      = (float *)alloc_mem(idim * kdim * jdim * sizeof(float));
	
	read_data( grid_u_2   , "grid_u_2.bin"   , idim, kdim, jdim );
	read_data( grid_u_save, "grid_u_save.bin", idim, kdim, jdim );
	read_data( grid_v_2   , "grid_v_2.bin"   , idim, kdim, jdim );
	read_data( grid_v_save, "grid_v_save.bin", idim, kdim, jdim );
	read_data( grid_t_save, "grid_t_save.bin", idim, kdim, jdim );
    read_data( t_tend     , "t_tend.bin"     , idim, kdim, jdim );
    
    // 3d in/out 
	float *grid_ww, *ww1, *grid_t_2, *t_2save; 
	
	grid_ww  = (float *)alloc_mem(idim * kdim * jdim * sizeof(float));
	ww1      = (float *)alloc_mem(idim * kdim * jdim * sizeof(float));
	grid_t_2 = (float *)alloc_mem(idim * kdim * jdim * sizeof(float));
	t_2save  = (float *)alloc_mem(idim * kdim * jdim * sizeof(float));
	
	read_data( grid_ww , "grid_ww.bin" , idim, kdim, jdim );
	read_data( ww1     , "ww1.bin"     , idim, kdim, jdim );
	read_data( grid_t_2, "grid_t_2.bin", idim, kdim, jdim );
	read_data( t_2save , "t_2save.bin" , idim, kdim, jdim );
    
    
    struct timeval ta, tb;
    long mseca, msecb;
    gettimeofday( &ta, NULL );
    mseca = ta.tv_sec * 1000000 + ta.tv_usec;
    
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

   
    gettimeofday( &tb, NULL );
    msecb = tb.tv_sec * 1000000 + tb.tv_usec;
    msecb -=mseca;
    printf(" advance_mu_t computing time(msec):\t%7.1f\n", (float)msecb/1000);

    compare(grid_ww , "grid_ww_output.bin" , idim * kdim * jdim);
    compare(ww1     , "ww1_output.bin"     , idim * kdim * jdim);
    compare(grid_t_2, "grid_t_2_output.bin", idim * kdim * jdim);
    compare(t_2save , "t_2save_output.bin" , idim * kdim * jdim);
    
	compare(grid_mu_2, "grid_mu_2_output.bin", idim * jdim);
	
    compare(muave    , "muave_output.bin"    , (i_end-i_start+1)*(j_end-j_start+1)); 

    compare_2d_t(grid_muts, "grid_muts_output.bin", i_start,i_end,j_start,j_end,idim,jdim);
    compare(grid_mudf, "grid_mudf_output.bin", (i_end-i_start+1)*(j_end-j_start+1));
	
    free( grid_ww );
    free( ww1 );
    free( grid_u_2 );
    free( grid_u_save );
    free( grid_v_2 );
    free( grid_v_save );
    free( grid_mu_2 );
    free( grid_mut );
    free( muave );
    free( grid_muts );
    free( grid_muu );
    free( grid_muv );
    free( grid_mudf );
    free( grid_t_2 );
    free( grid_t_save );
    free( t_2save );
    free( t_tend );
    free( mu_tend );
    free( grid_dnw );
    free( grid_fnm );
    free( grid_fnp );
    free( grid_rdnw );
    free( grid_msfuy );
    free( grid_msfvx_inv );
    free( grid_msftx );
    free( grid_msfty );

    
    return 0;
   
}	


void *alloc_mem(int size)
{
    void *ptr;
    if ((ptr = (void *)malloc(size)) == NULL) {
        fprintf(stderr, "Can\'t allocate memory (size = %d)!\n", (int)size);
        exit(1);
    }
    return (ptr);
}
    
void read_dim_data(int *data, char *file_name)
{
    FILE *fp;
    unsigned char tmp[4], dat[4];
  
   char input_data_dir_file_name[120];
   strcpy (input_data_dir_file_name,input_data_dir);
   strcat (input_data_dir_file_name,file_name);
   //printf("%s\n",input_data_dir_file_name);
   if((fp = fopen(input_data_dir_file_name, "rb")) == NULL){
      perror("read_dim_data: ");
      exit(1);
   }
   fread(&tmp, 4, 1, fp);
   dat[0] = tmp[3];
   dat[1] = tmp[2];
   dat[2] = tmp[1];
   dat[3] = tmp[0];
   *data = *((int *) dat);
   fclose(fp);
}


void read_data(float *data, char *file_name, int I, int K, int J)
{
   FILE *fp;
   int i,j,k;
   char tmp[4], dat[4];
  
   char input_data_dir_file_name[120];
   strcpy (input_data_dir_file_name,input_data_dir);
   strcat (input_data_dir_file_name,file_name);

   if((fp = fopen(input_data_dir_file_name, "rb")) == NULL){
      fprintf(stderr, "unable to open file %s\n", file_name);
      exit(1);
   }

   for(j=0; j<J; j++)
      for(k=0; k<K; k++)
         for(i=0; i<I; i++){
            fread(tmp, 4, 1, fp);
            dat[0]=tmp[3];
            dat[1]=tmp[2];
            dat[2]=tmp[1];
            dat[3]=tmp[0];
            data[j*K*I + k*I + i] = *((float *) dat);
         if(isnan(data[j*K*I + k*I + i])){
            printf("read_data '%s': data[%i] is not a number\n", file_name, 
                    j*K*I + k*I + i);
            exit(1);
         }
      }
   fclose(fp);
}  
    
void read_data_4d(float *data, char *file_name, int I, int K, int J, int S)
{
   FILE *fp;
   int i,j,k,s;
   char tmp[4], dat[4];
  
   char input_data_dir_file_name[120];
   strcpy (input_data_dir_file_name,input_data_dir);
   strcat (input_data_dir_file_name,file_name);

   if((fp = fopen(input_data_dir_file_name, "rb")) == NULL){
      fprintf(stderr, "unable to open file %s\n", file_name);
      exit(1);
   }
    for(s=0; s<S; s++)
       for(j=0; j<J; j++)
          for(k=0; k<K; k++)
             for(i=0; i<I; i++){
                fread(tmp, 4, 1, fp);
                dat[0]=tmp[3];
                dat[1]=tmp[2];
                dat[2]=tmp[1];
                dat[3]=tmp[0];
                data[s*J*K*I + j*K*I + k*I + i] = *((float *) dat);
             if(isnan(data[s*J*K*I + j*K*I + k*I + i])){
                printf("read_data '%s': data[%i] is not a number\n", file_name, 
                        j*K*I + k*I + i);
                exit(1);
             }
          }

   fclose(fp);
   
}  



void read_real_data(float *data, char *file_name)
{
  	FILE *fp;
  	unsigned char tmp[4], dat[4];
  
	char input_data_dir_file_name[120];
	strcpy (input_data_dir_file_name,input_data_dir);
	strcat (input_data_dir_file_name,file_name);

  	if((fp = fopen(input_data_dir_file_name, "rb")) == NULL){
    	perror("read_dim_data: ");
    	exit(1);
  	}
  	fread(&tmp, 4, 1, fp);
  	dat[0] = tmp[3];
  	dat[1] = tmp[2];
  	dat[2] = tmp[1];
  	dat[3] = tmp[0];
  	*data = *((float *) dat);
  	fclose(fp);
}

void compare_2d_t(float *data, char *file_name, int i_start, int i_end, 
        int j_start, int j_end, int idim, int jdim){
    FILE *fp;
    int i;
    float value, abs_err, rel_err;
    char tmp[4], dat[4];

    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    float min_f=FLT_MAX;
    float max_f=FLT_MIN;
    float min_c=FLT_MAX;
    float max_c=FLT_MIN;

    int equal_values=0;
    int different_values=0;
    int row = 0;
    int column = 0;
      
    int max_ulp = 0;
    int max_abs_pos=-1;
    int max_rel_pos=-1;
    float value_abs_c=-1;
    float value_abs_f=-1;
    float value_rel_c=-1;
    float value_rel_f=-1;
    float rmse = 0.0f;
    int ulp;
    
    char input_data_dir_file_name[120];
    strcpy (input_data_dir_file_name, default_compare_data_dir);
    strcat (input_data_dir_file_name,file_name);
    if((fp = fopen(input_data_dir_file_name, "r")) == NULL){
        fprintf(stderr, "unable to open file %s\n", file_name);
        exit(1);
    }

    for(i = 0; i <= idim * jdim; i++){
  
        fread(&tmp, 4, 1, fp);
        dat[0] = tmp[3];
        dat[1] = tmp[2];
        dat[2] = tmp[1];
        dat[3] = tmp[0];
        value = *((float *) dat);
        
        if(column > i_start && column < i_end && row >= j_start && row <= j_end){
        
            if(isnan(data[row*idim + column])){
                printf("compare '%s': C output - data[%i] is not a number\n", 
                    file_name, i);
                exit(1);
            }
            if(isnan(value)){
                printf("compare '%s': Fortran output value[%i] is not a number\n", 
                    file_name, i);
                exit(1);
            }

            if( fabsf(value) != 0.0f && fabsf(data[row*idim + column]) != 0.0f )
                rel_err = (fabsf(value -  data[row*idim + column]) ) 
                        / max(fabsf(value), fabsf(data[row*idim + column]));
            else 
                rel_err = max( fabsf(value), fabsf(data[row*idim + column]) );
            if(rel_err > max_rel_err) { 
                max_rel_err = rel_err;
                max_rel_pos = i;
                value_rel_f = value;
                value_rel_c = data[row*idim + column];
            }

            float_ulps_(&value, &data[row*idim + column], &ulp);
            if(ulp > max_ulp)
                max_ulp = ulp;

            abs_err = fabsf(value - data[row*idim + column]);
            rmse = rmse + powf(abs_err, 2.0f);

            if(abs_err > max_abs_err){
                max_abs_err = abs_err;
                max_abs_pos = i;
                value_abs_f = value;
                value_abs_c = data[row*idim + column];
            }
    
            if(value == data[row*idim + column])
                equal_values++;
            else{
        //      printf("Fortran: %e\tC:%e\n", value, data[i]);
                different_values++;
            }

            if(value > max_f)
                max_f = value;
            if(value < min_f)
                min_f = value;
            if(data[i] > max_c)
                max_c = data[row*idim + column];
            if(data[i] < min_c)
                min_c = data[row*idim + column];
        } //end if
        column++;
        if(column == idim) {
	        column = 0;
	        row++;
        }
          
    }
    rmse = sqrt(rmse/(equal_values+different_values));

  printf("\n");
//  printf("\n%s:\nmax relative error: %e @%i max absolute error: %e @%i\n", file_name, max_rel_err, max_rel_pos, max_abs_err, max_abs_pos);
//  printf("Max relative error: Fortran: %e, C: %e\n", value_rel_f, value_rel_c);
//  printf("Max absolute error: Fortran: %e, C: %e\n", value_abs_f, value_abs_c);
//  printf("Fortran: [%e,%e], C: [%e,%e]\n", min_f, max_f, min_c, max_c);
  printf("# of equal values: %i, # of non-equal values: %i\n", equal_values, different_values);

//  fprintf(fp_results,"max relative error: %e @%i\t max absolute error: %e @%i\t %s\n",max_rel_err,max_rel_pos, max_abs_err, max_abs_pos,file_name);
//  printf("max relative error: %e @%i\t max absolute error: %e @%i\t %s\n",max_rel_err,max_rel_pos, max_abs_err, max_abs_pos,file_name);
  printf("max relative error: %e\t max absolute error: %e\t %s\n",max_rel_err, max_abs_err, file_name);
  printf("max ulp = %i\n", max_ulp);
  printf("rmse = %e\n",rmse);

  fclose(fp);
}

void compare(float *data, char *file_name, int size){
  FILE *fp;
  int i;
  float value, abs_err, rel_err;
  char tmp[4], dat[4];

  float max_abs_err = 0.0f;
  float max_rel_err = 0.0f;
  float min_f=FLT_MAX;
  float max_f=FLT_MIN;
  float min_c=FLT_MAX;
  float max_c=FLT_MIN;

  int equal_values=0;
  int different_values=0;
  int max_ulp = 0;
  int max_abs_pos=-1;
  int max_rel_pos=-1;
  float value_abs_c=-1;
  float value_abs_f=-1;
  float value_rel_c=-1;
  float value_rel_f=-1;
  float rmse = 0.0f;
  int ulp;

  char input_data_dir_file_name[120];
  strcpy (input_data_dir_file_name, default_compare_data_dir);
  
  strcat (input_data_dir_file_name,file_name);
  if((fp = fopen(input_data_dir_file_name, "r")) == NULL){
    fprintf(stderr, "unable to open file %s\n", file_name);
    exit(1);
  }

  for(i=0; i<size; i++){
    fread(&tmp, 4, 1, fp);
    dat[0] = tmp[3];
    dat[1] = tmp[2];
    dat[2] = tmp[1];
    dat[3] = tmp[0];
    value = *((float *) dat);
    if(isnan(data[i])){
      printf("compare '%s': C output - data[%i] is not a number\n", 
            file_name, i);
      exit(1);
    }
    if(isnan(value)){
      printf("compare '%s': Fortran output value[%i] is not a number\n", 
            file_name, i);
      exit(1);
    }

    if( fabsf(value) != 0.0f && fabsf(data[i]) != 0.0f )
      rel_err = (fabsf(value -  data[i]) ) / max(fabsf(value), fabsf(data[i]));
    else 
      rel_err = max( fabsf(value), fabsf(data[i]) );
    if(rel_err > max_rel_err) { 
      max_rel_err = rel_err;
      max_rel_pos = i;
      value_rel_f = value;
      value_rel_c = data[i];
    }

    float_ulps_(&value, &data[i], &ulp);
    if(ulp > max_ulp)
      max_ulp = ulp;

    abs_err = fabsf(value - data[i]);
    rmse = rmse + powf(abs_err, 2.0f);

    if(abs_err > max_abs_err){
      max_abs_err = abs_err;
      max_abs_pos = i;
      value_abs_f = value;
      value_abs_c = data[i];
    }

    if(value == data[i])
      equal_values++;
    else{
//      printf("Fortran: %e\tC:%e\n", value, data[i]);
      different_values++;
    }

    if(value > max_f)
      max_f = value;
    if(value < min_f)
      min_f = value;
    if(data[i] > max_c)
      max_c = data[i];
    if(data[i] < min_c)
      min_c = data[i];
  }

  rmse = sqrt(rmse/(equal_values+different_values));

  printf("\n");
//  printf("\n%s:\nmax relative error: %e @%i max absolute error: %e @%i\n", file_name, max_rel_err, max_rel_pos, max_abs_err, max_abs_pos);
//  printf("Max relative error: Fortran: %e, C: %e\n", value_rel_f, value_rel_c);
//  printf("Max absolute error: Fortran: %e, C: %e\n", value_abs_f, value_abs_c);
//  printf("Fortran: [%e,%e], C: [%e,%e]\n", min_f, max_f, min_c, max_c);
  printf("# of equal values: %i, # of non-equal values: %i\n", equal_values, different_values);

//  fprintf(fp_results,"max relative error: %e @%i\t max absolute error: %e @%i\t %s\n",max_rel_err,max_rel_pos, max_abs_err, max_abs_pos,file_name);
//  printf("max relative error: %e @%i\t max absolute error: %e @%i\t %s\n",max_rel_err,max_rel_pos, max_abs_err, max_abs_pos,file_name);
  printf("max relative error: %e\t max absolute error: %e\t %s\n",max_rel_err, max_abs_err, file_name);
  printf("max ulp = %i\n", max_ulp);
  printf("rmse = %e\n",rmse);

  fclose(fp);
}


int float_ulps(float a, float b)
//int float_eq(float a, float b, int maxulps)
// ulp = units in the last place; maxulps = maximum number of
// representable floating point numbers by which x and y may differ.
// http://www.mrupp.info/Data/2007floatingcomp.pdf
{
  // maxulps small enough so NaN will not be equal to other numbers.
  // convert to integer.
  int aint = *(int*)&a;
  int bint = *(int*)&b;
  // make lexicographically ordered as a twos-complement int
  if (aint < 0) aint = 0x80000000 - aint;
  if (bint < 0) bint = 0x80000000 - bint;
  // compare.
  return abs(aint - bint);
}   
