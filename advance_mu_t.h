

typedef struct _config_flags {
	int nested;
	int periodic_x;
	int specified;

} config_flags;

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
                   int its, int ite, int jts, int jte, int kts, int kte );
           
void *alloc_mem(int size);

