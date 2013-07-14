//#ifndef __COMMON__
//#define __COMMON__

void read_data_4d(float *data, char *file_name, int I, int K, int J, int M);
void compare(float *data, char *file_name, int ims, int ime, int kms, int kme, int jms, int jme, int its, int ite, int kts, int kte, int jts, int jte);
void read_dim_data(int *data, char *file_name);
void read_real_data(float *data, char *file_name);
void read_data(float *data, char *file_name, int x, int y, int z);
void read_data_int(int *data, char *file_name, int x, int y, int z);
void write_data(float *data, char *file_name, int I, int K, int J);
void swap_data_4d(float *data_in, float *data_out, int I, int K, int J, int M);
void compare_4d(float *data, char *file_name, int ims, int ime, int kms, int kme, int jms, int jme, int sms, int sme, int its, int ite, int kts, int kte, int jts, int jte, int sts, int ste);
void compare_real(float data, char *file_name);

//#endif
