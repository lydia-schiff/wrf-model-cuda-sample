#include <stdio.h>
//#include "common.h"

extern char *default_compare_data_dir;
extern char input_data_dir[256];
extern char output_data_dir[256];
#define MAX(A,B) ((A>B) ? (A) : (B))

// NOTE: read_data_4d -- use (i, k, m, j) order
void read_data_4d(float *data, char *file_name, int I, int K, int J, int M)
{
  FILE *fp;
  int i, j, k, m;
  char tmp[4], dat[4];
  char input_data_dir_file_name[120];

  strcpy (input_data_dir_file_name, input_data_dir);
  strcat (input_data_dir_file_name, file_name);

  if( (fp = fopen(input_data_dir_file_name, "rb")) == NULL)
    {
     fprintf(stderr, "unable to open file %s\n", file_name);
     exit(1);
    }

  for(m=0; m<M; m++)
  for(j=0; j<J; j++)
    for(k=0; k<K; k++)
      for(i=0; i<I; i++)
        {
          fread(tmp, 4, 1, fp);
          dat[0]=tmp[3];
          dat[1]=tmp[2];
          dat[2]=tmp[1];
          dat[3]=tmp[0];
          //data[m*J*K*I + j*K*I + k*I + i] = *((float *) dat);  // ikjm order
          data[j*M*K*I + m*K*I + k*I + i] = *((float *) dat);    // ikmj order

          if( isnan(data[j*M*K*I + m*K*I + k*I + i]) )   // check if it's "nan"
            {
              printf("read_data '%s': data[%i] is not a number\n",
                      file_name, j*M*K*I + m*K*I + k*I + i);
              //exit(1);
            }
        } // end of i, k, j, m loops

  fclose(fp);
}


void float_ulps(float *a, float *b, int *maxulps)
// ulp = units in the last place
// maxulps = returns maximum number of
// representable floating point numbers by which x and y may differ.
// http://www.mrupp.info/Data/2007floatingcomp.pdf
{
  // convert to integer.
  int aint = *(int*)a;
  int bint = *(int*)b;
  // make lexicographically ordered as a twos-complement int
  if (aint < 0) aint = 0x80000000 - aint;
  if (bint < 0) bint = 0x80000000 - bint;
  // compare.
  *maxulps = abs(aint - bint);
  return;
}

void compare(float *data, char *file_name, int ims, int ime, int kms, int kme, int jms, int jme, int its, int ite, int kts, int kte, int jts, int jte){
  FILE *fp;
  int i, j, k, ind;
  float value, abs_err, rel_err;
  char tmp[4], dat[4];

  float max_abs_err = 0.0f;
  float max_rel_err = 0.0f;
  int equal_values=0;
  int different_values=0;
  int max_ulp = 0;
//  int max_abs_pos=-1;
/*
  int max_rel_pos=-1;
  int max_rel_pos_i=-1;
  int max_rel_pos_k=-1;
  int max_rel_pos_j=-1;
*/
  float rmse = 0.0f;
  int ulp;

  char input_data_dir_file_name[120];
  strcpy (input_data_dir_file_name, default_compare_data_dir);
  strcat (input_data_dir_file_name,file_name);
  if((fp = fopen(input_data_dir_file_name, "r")) == NULL){
    fprintf(stderr, "unable to open file %s\n", file_name);
    exit(1);
  }

  ind = 0;
  for(j=jms; j<=jme; j++){
  for(k=kms; k<=kme; k++){
  for(i=ims; i<=ime; i++){
    fread(&tmp, 4, 1, fp);
    if(i>=its && i<=ite && k>=kts && k<=kte && j>=jts && j<=jte){
      dat[0] = tmp[3]; // change endianess
      dat[1] = tmp[2];
      dat[2] = tmp[1];
      dat[3] = tmp[0];
      value = *((float *) dat);
      if(isnan(data[ind])){
        printf("compare '%s': C output - data[%i] is not a number - Fortran index: (%i,%i,%i)\n", file_name, ind, i, k, j);
        exit(1);
      }
      if(isnan(value)){
        printf("compare '%s': Fortran output value(%i,%i,%i) is not a number - C index: [%i]\n", file_name, i, k, j, ind);
        exit(1);
      }

      if( fabsf(value) != 0.0f && fabsf(data[ind]) != 0.0f )
        rel_err = (fabsf(value -  data[ind]) ) / MAX(fabsf(value), fabsf(data[i]) ) ;
      else 
        rel_err = MAX( fabsf(value), fabsf(data[ind]) );
      if(rel_err > max_rel_err) { 
        max_rel_err = rel_err;
/*
        max_rel_pos_i = i;
        max_rel_pos_k = k;
        max_rel_pos_j = j;
        max_rel_pos = ind;
*/
      }

      float_ulps(&value, &data[ind], &ulp);
      if(ulp > max_ulp)
        max_ulp = ulp;

      abs_err = fabsf(value - data[ind]);
      rmse = rmse + powf(abs_err, 2.0f);

      if(abs_err > max_abs_err){
        max_abs_err = abs_err;
//        max_abs_pos = ind;
      }

      if(value == data[ind])
        equal_values++;
      else{
        different_values++;
      }
    }
    ind++;
  }
  }
  }

  rmse = sqrt(rmse/(equal_values+different_values));

  printf("\n");
  printf("# of equal values: %i, # of non-equal values: %i\n", equal_values, different_values);
  printf("max relative error: %e\tmax absolute error: %e\t%s\n",max_rel_err, max_abs_err, file_name);
//  if(max_rel_pos != -1)
//    printf("position of max relative error - Fortran:(%i, %i, %i), C:[%i]\n", max_rel_pos_i, max_rel_pos_k, max_rel_pos_j, max_rel_pos);
  printf("max ulp = %i\t\t\t\trmse = %e\n", max_ulp,rmse);

  fclose(fp);
}

void read_dim_data(int *data, char *file_name)
{
  FILE *fp;
  unsigned char tmp[4], dat[4];
  char input_data_dir_file_name[120];

  strcpy (input_data_dir_file_name, input_data_dir);
  strcat (input_data_dir_file_name, file_name);

  if( (fp = fopen(input_data_dir_file_name, "rb")) == NULL )
    {
      perror("read_dim_data: ");
      exit(1);
    }

  fread(&tmp, 4, 1, fp);   // 4: bytes of each element to be read
                           // 1: # of elements
  dat[0] = tmp[3];
  dat[1] = tmp[2];
  dat[2] = tmp[1];
  dat[3] = tmp[0];
  *data = *((int *) dat);     

  fclose(fp);
}


// read_real_data
void read_real_data(float *data, char *file_name)
{
  FILE *fp;
  unsigned char tmp[4], dat[4];
  char input_data_dir_file_name[120];

  strcpy (input_data_dir_file_name, input_data_dir);
  strcat (input_data_dir_file_name, file_name);

  if( (fp = fopen(input_data_dir_file_name, "rb")) == NULL)
    {
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


// read_data
void read_data(float *data, char *file_name, int I, int K, int J)
{
  FILE *fp;
  int i, j, k;
  char tmp[4], dat[4];
  char input_data_dir_file_name[120];

  strcpy (input_data_dir_file_name, input_data_dir);
  strcat (input_data_dir_file_name, file_name);

  if( (fp = fopen(input_data_dir_file_name, "rb")) == NULL)
    {
     fprintf(stderr, "unable to open file %s\n", file_name);
     exit(1);
    }

  for(j=0; j<J; j++)
    for(k=0; k<K; k++)
      for(i=0; i<I; i++)
        {
          fread(tmp, 4, 1, fp);
          dat[0]=tmp[3];
          dat[1]=tmp[2];
          dat[2]=tmp[1];
          dat[3]=tmp[0];
          data[j*K*I + k*I + i] = *((float *) dat);
     
          if( isnan(data[j*K*I + k*I + i]) )   // check if it's "nan"
            {
              printf("read_data '%s': data[%i] is not a number\n", 
                      file_name, j*K*I + k*I + i);
              //exit(1);
            }
        } // end of i, k, j-loop

  fclose(fp);
}


// read_data_int
void read_data_int(int *data, char *file_name, int I, int K, int J)
{
  FILE *fp;
  int i, j, k;
  char tmp[4], dat[4];
  char input_data_dir_file_name[120];

  strcpy (input_data_dir_file_name, input_data_dir);
  strcat (input_data_dir_file_name, file_name);

  if( (fp = fopen(input_data_dir_file_name, "rb")) == NULL)
    {
     fprintf(stderr, "unable to open file %s\n", file_name);
     exit(1);
    }

  for(j=0; j<J; j++)
    for(k=0; k<K; k++)
      for(i=0; i<I; i++)
        {
          fread(tmp, 4, 1, fp);
          dat[0]=tmp[3];
          dat[1]=tmp[2];
          dat[2]=tmp[1];
          dat[3]=tmp[0];
          data[j*K*I + k*I + i] = *((int *) dat);
  
          if( isnan(data[j*K*I + k*I + i]) )   // check if it's "nan"
            {
              printf("read_data '%s': data[%i] is not a number\n", 
                      file_name, j*K*I + k*I + i);
              exit(1);
            }
        }  // end of i,k, j-loop

  fclose(fp);
}

// write_data
void write_data(float *data, char *file_name, int I, int K, int J)
{
  FILE *fp;
  int i,j,k;
  char tmp[4], dat[4];
  char output_data_dir_file_name[120];

  strcpy(output_data_dir_file_name, output_data_dir);
  strcat(output_data_dir_file_name, file_name);

  if( (fp = fopen(output_data_dir_file_name, "wb") ) == NULL)
    {
      fprintf(stderr, "unable to open file %s\n", file_name);
       exit(1);
    }

  for(j=0; j<J; j++)
    for(k=0; k<K; k++)
      for(i=0; i<I; i++)
        {
          *((float *) dat) = data[j*K*I + k*I + i];
          tmp[3]=dat[0];
          tmp[2]=dat[1];
          tmp[1]=dat[2];
          tmp[0]=dat[3];
          fwrite(tmp, 4, 1, fp);
        }
  fclose(fp);
}

// swap_data_4d
void swap_data_4d(float *data_in, float *data_out, int I, int K, int J, int M)
{
  int i, k, j, m;

  for(m=0; m<M; m++)
  for(j=0; j<J; j++)
    for(k=0; k<K; k++)
      for(i=0; i<I; i++)
        {
          data_out[m*J*K*I + j*K*I + k*I + i] =   // ikjm order
           data_in[j*M*K*I + m*K*I + k*I + i];    // ikmj order
        } // end of i, k, j, m loops
}

void compare_4d(float *data, char *file_name, int ims, int ime, int kms, int kme, int jms, int jme, int sms, int sme, int its, int ite, int kts, int kte, int jts, int jte, int sts, int ste){
  FILE *fp;
  int i, j, k, s, ind;
  float value, abs_err, rel_err;
  char tmp[4], dat[4];

  float max_abs_err = 0.0f;
  float max_rel_err = 0.0f;
  int equal_values=0;
  int different_values=0;
  int max_ulp = 0;

  float rmse = 0.0f;
  int ulp;

  char input_data_dir_file_name[120];
  strcpy (input_data_dir_file_name, default_compare_data_dir);
  strcat (input_data_dir_file_name,file_name);
  if((fp = fopen(input_data_dir_file_name, "r")) == NULL){
    fprintf(stderr, "unable to open file %s\n", file_name);
    exit(1);
  }

  ind = 0;
  for(s=sms; s<=sme; s++){
  for(j=jms; j<=jme; j++){
  for(k=kms; k<=kme; k++){
  for(i=ims; i<=ime; i++){
    fread(&tmp, 4, 1, fp);
    if(i>=its && i<=ite && k>=kts && k<=kte && s>=sts && s<=ste && j>=jts && j<=jte){
      dat[0] = tmp[3]; // change endianess
      dat[1] = tmp[2];
      dat[2] = tmp[1];
      dat[3] = tmp[0];
      value = *((float *) dat);
      if(isnan(data[ind])){
        printf("compare '%s': C output - data[%i] is not a number - Fortran index: (%i,%i,%i,%i)\n", file_name, ind, i, k, s, j);
        exit(1);
      }
      if(isnan(value)){
        printf("compare '%s': Fortran output value(%i,%i,%i,%i) is not a number - C index: [%i]\n", file_name, i, k, j, s, ind);
        exit(1);
      }

      if( fabsf(value) != 0.0f && fabsf(data[ind]) != 0.0f )
        rel_err = (fabsf(value -  data[ind]) ) / MAX(fabsf(value), fabsf(data[i]) ) ;
      else 
        rel_err = MAX( fabsf(value), fabsf(data[ind]) );
      if(rel_err > max_rel_err) { 
        max_rel_err = rel_err;
      }

      float_ulps(&value, &data[ind], &ulp);
      if(ulp > max_ulp)
        max_ulp = ulp;

      abs_err = fabsf(value - data[ind]);
      rmse = rmse + powf(abs_err, 2.0f);

      if(abs_err > max_abs_err){
        max_abs_err = abs_err;
      }

      if(value == data[ind])
        equal_values++;
      else{
        different_values++;
      }
    }
    ind++;
  }
  }
  }
  }

  rmse = sqrt(rmse/(equal_values+different_values));

  printf("\n");
  printf("# of equal values: %i, # of non-equal values: %i\n", equal_values, different_values);
  printf("max relative error: %e\tmax absolute error: %e\t%s\n",max_rel_err, max_abs_err, file_name);
  printf("max ulp = %i\t\t\t\trmse = %e\n", max_ulp,rmse);

  fclose(fp);
}

void compare_real(float data, char *file_name)
{
  FILE *fp;
  float rel_err, abs_err;
  float value;
  char tmp[4], dat[4];
    
  char input_data_dir_file_name[120];
  strcpy (input_data_dir_file_name, default_compare_data_dir);
  
  strcat (input_data_dir_file_name,file_name);
  if((fp = fopen(input_data_dir_file_name, "r")) == NULL)
    {
      fprintf(stderr, "unable to open file %s\n", file_name);
      exit(1);
    }
    
  fread(&tmp, 4, 1, fp);
  dat[0] = tmp[3];
  dat[1] = tmp[2];
  dat[2] = tmp[1];
  dat[3] = tmp[0];
  value = *((float *) dat);
  fclose(fp);
    
  if(isnan(data))
    {
      printf("compare '%s': C output - data is not a number\n", 
          file_name);
      exit(1);
    }
  if(isnan(value))
    {
      printf("compare '%s': Fortran output value is not a number\n", 
          file_name);
      exit(1);
    }

  if( fabsf(value) != 0.0f && fabsf(data) != 0.0f )
       rel_err = (fabsf(value -  data) ) / max(fabsf(value), fabsf(data));
  else 
      rel_err = max( fabsf(value), fabsf(data) );


  abs_err = fabsf(value - data);
  printf("\n%s",file_name);
  printf("absolute/relative errorw: %f %f \n", abs_err, rel_err);
}

