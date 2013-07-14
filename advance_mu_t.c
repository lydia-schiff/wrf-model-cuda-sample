#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include "advance_mu_t.h"

#define I3(i,k,j) ((j) * kdim * idim + (k) * idim + (i))
#define I2(i,j)   ((j) * idim + (i))

#define min(a,b)  ((a)<(b)?(a):(b))
#define max(a,b)  ((a)>(b)?(a):(b))
#define true 1
#define false 0


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
    size_t idim = ime - ims + 1;
    size_t kdim = kme - kms + 1;

    ide = ide - ids + 1;
    jde = jde - jds + 1;
    kde = kde - kds + 1; 
    ite = ite - its + 1; // i_size
    jte = jte - jts + 1; // j_size
	kte = kte - kts + 1; // k_size

    ids = ids - ims;
    jds = jds - jms;
    kds = kds - kms;
    its = its - ims; // i_start index in memory
    jts = jts - jms; // j_start index in memory
    kts = kts - kms; // k_start index in memory
   
    ide = ide + ids - 1;
    jde = jde + jds - 1;
    kde = kde + kds - 1;
    ite = ite + its - 1; // i_end index in memory
    jte = jte + jts - 1; // j_end index in memory
    kte = kte + kts - 1; // k_end index in memory


//  Local arrays from the stack (note tile size)

    float *wdtn, *dvdxi;
	
	wdtn  = (float *)alloc_mem(idim * kdim * 1 * sizeof(float));
	dvdxi = (float *)alloc_mem(idim * kdim * 1 * sizeof(float));
	
	float *dmdt;
	
	dmdt = (float *)alloc_mem(idim * 1 * 1 * sizeof(float));

    int i, j, k;
    int i_start, i_end, j_start, j_end, k_start, k_end;
    int i_endu, j_endv;
    
//<DESCRIPTION>
//
//  advance_mu_t advances the explicit perturbation theta equation and the mass
//  conservation equation.  In addition, the small timestep omega is updated,
//  and some quantities needed in other places are squirrelled away.
//
//</DESCRIPTION>

//  now, the real work.
//  set the loop bounds taking into account boundary conditions.

    i_start = its;
    i_end   = min(ite,ide-1);
    j_start = jts;
    j_end   = min(jte,jde-1);
    k_start = kts;
    k_end   = kte-1;
    if( !config.periodic_x ){
        if( config.specified || config.nested ) {
            i_start = max(its,ids+1);
            i_end   = min(ite,ide-2);
        }
    }
    if( config.specified || config.nested ) {
        j_start = max(jts,jds+1);
        j_end   = min(jte,jde-2);
    }

    i_endu = ite;
    j_endv = jte;

//        CALCULATION OF WW (dETA/dt)
    for(j = j_start; j <= j_end; j++){

        for(i = i_start; i <= i_end; i++){
            dmdt[i] = 0.0f;
        }
//  NOTE:  mu is not coupled with the map scale factor.
//         ww (omega) IS coupled with the map scale factor.
//         Being coupled with the map scale factor means 
//           multiplication by (1/msft) in this case.

//  Comments on map scale factors
//  ADT eqn 47: 
//  partial drho/dt = -mx*my[partial d/dx(rho u/my) + partial d/dy(rho v/mx)]
//                    -partial d/dz(rho w)
//  with rho -> mu, dividing by my, and with partial d/dnu(rho nu/my [=ww])
//  as the final term (because we're looking for d_nu_/dt)
//
//  begin by integrating with respect to nu from bottom to top
//  BCs are ww=0 at both
//  final term gives 0
//  first term gives Integral([1/my]partial d mu/dt) over total column = dm/dt
//  RHS remaining is Integral(-mx[partial d/dx(mu u/my) + 
//                                partial d/dy(mu v/mx)]) over column
//  lines below find RHS terms at each level { set dmdt = sum over all levels
//
//  [don't divide the below by msfty until find ww, since dmdt is used in
//   the meantime]

        for(k = k_start; k <= k_end; k++){
            for(i = i_start; i <= i_end; i++){
                dvdxi[I2(i,k)] = msftx[I2(i,j)]*msfty[I2(i,j)]*(                       
                     rdy*( (v[I3(i,k,j+1)]+muv[I2(i,j+1)]*v_1[I3(i,k,j+1)]*msfvx_inv[I2(i,j+1)]) 
                          -(v[I3(i,k,j  )]+muv[I2(i,j  )]*v_1[I3(i,k,j  )]*msfvx_inv[I2(i,j  )]) ) 
                    +rdx*( (u[I3(i+1,k,j)]+muu[I2(i+1,j)]*u_1[I3(i+1,k,j)]/msfuy[I2(i+1,j)]) 
                          -(u[I3(i  ,k,j)]+muu[I2(i  ,j)]*u_1[I3(i  ,k,j)]/msfuy[I2(i  ,j)]) ));
                dmdt[i] = dmdt[i] + dnw[k]*dvdxi[I2(i,k)];
            }
        }

        for(i = i_start; i <= i_end; i++){
            muave[I2(i,j)] = mu[I2(i,j)];
            mu[I2(i,j)] = mu[I2(i,j)]+dts*(dmdt[i]+mu_tend[I2(i,j)]);
            mudf[I2(i,j)] = (dmdt[i]+mu_tend[I2(i,j)]); // save tendency for div damp filter
            muts[I2(i,j)] = mut[I2(i,j)]+mu[I2(i,j)];
            muave[I2(i,j)] =0.5f*((1.0f+epssm)*mu[I2(i,j)]+(1.0f-epssm)*muave[I2(i,j)]);
        }

        for(k = 1; k <= k_end; k++){
            for(i = i_start; i <= i_end; i++){
                ww[I3(i,k,j)]=ww[I3(i,k-1,j)]-dnw[k-1]*(dmdt[i]+dvdxi[I2(i,k-1)]
                        +mu_tend[I2(i,j)])/msfty[I2(i,j)];
            }
        }

//  NOTE:  ww_1 (large timestep ww) is already coupled with the 
//         map scale factor

        for(k = 0; k <= k_end; k++){
            for(i = i_start; i <= i_end; i++){
                ww[I3(i,k,j)] = ww[I3(i,k,j)]-ww_1[I3(i,k,j)];
            }
        }

    } // j

// CALCULATION OF THETA

// NOTE: theta'' is not coupled with the map-scale factor, 
//       while the theta'' tendency is coupled (i.e., mult by 1/msft)

// Comments on map scale factors
// BUT NOTE THAT both are mass coupled
// in flux form equations (Klemp et al.) Theta = mu*theta
//
// scalar eqn: partial d/dt(rho q/my) = -mx[partial d/dx(q rho u/my) + 
//                                          partial d/dy(q rho v/mx)]
//                                      - partial d/dz(q rho w/my)
// with rho -> mu, and with partial d/dnu(q rho nu/my) as the final term
//
// adding previous tendency contribution which was map scale factor coupled
// (had been divided by msfty)
// need to uncouple before updating uncoupled Theta (by adding)

    for(j = j_start; j <= j_end; j++){
        for(k = 0; k <= k_end; k++){
            for(i = i_start; i <= i_end; i++){
                t_ave[I3(i,k,j)] = t[I3(i,k,j)];
                t[I3(i,k,j)] = t[I3(i,k,j)] + msfty[I2(i,j)]*dts*ft[I3(i,k,j)];
            }
        }
    }  

    for(j = j_start; j <= j_end; j++){

        for(i = i_start; i <= i_end; i++){
            wdtn[I2(i,0)] = 0.0f;
            wdtn[I2(i,kde)] = 0.0f;
        }
    

        for(k = 1; k <= k_end; k++){
            for(i = i_start; i <= i_end; i++){
                // for scalar eqn RHS term 3
                wdtn[I2(i,k)] = ww[I3(i,k,j)]*(fnm[k]*t_1[I3(i,k,j)]+fnp[k]
                        *t_1[I3(i,k-1,j)]);
            }
        }
    

// scalar eqn, RHS terms 1, 2 and 3
// multiply by msfty to uncouple result for Theta from map scale factor

        for(k = 0; k <= k_end; k++){
            for(i = i_start; i <= i_end; i++){
                // multiplication by msfty uncouples result for Theta
                t[I3(i,k,j)] = t[I3(i,k,j)] - dts*msfty[I2(i,j)]*(         
                              // multiplication by mx needed for RHS terms 1 & 2
                              msftx[I2(i,j)]*(         
                               0.5f*rdy*                           
                              ( v[I3(i,k,j+1)]*(t_1[I3(i,k,j+1)]+t_1[I3(i,k,j  )])  
                               -v[I3(i,k,j  )]*(t_1[I3(i,k,j  )]+t_1[I3(i,k,j-1)]) ) 
                             + 0.5f*rdx*                                 
                              ( u[I3(i+1,k,j)]*(t_1[I3(i+1,k,j)]+t_1[I3(i  ,k,j)])  
                               -u[I3(i  ,k,j)]*(t_1[I3(i  ,k,j)]+t_1[I3(i-1,k,j)]) ) )
                             + rdnw[k]*( wdtn[I2(i,k+1)]-wdtn[I2(i,k)] ) );     
            }
        }
    }
    
    
    free(wdtn);
    free(dvdxi);
	free(dmdt);
	
}


