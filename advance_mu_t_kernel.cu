#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include "config_flags.h"
#include "advance_mu_t_cu.h"

#define I3(i,k,j) ((j) * kdim * idim + (k) * idim + (i))
#define I2(i,j)   ((j) * idim + (i))

#define min(a,b)  ((a)<(b)?(a):(b))
#define max(a,b)  ((a)>(b)?(a):(b))


__global__ void advance_mu_t_kernel( float * __restrict__ ww, float * __restrict__ ww_1, float * __restrict__ u, float * __restrict__ u_1, 
                   float *v, float *v_1,            
                   float * __restrict__ mu, float * __restrict__ mut, float * __restrict__ muave, float * __restrict__ muts, 
                   float * __restrict__ muu,float * __restrict__ muv,
                   float * __restrict__ mudf, float * __restrict__ t, float * __restrict__ t_1,                      
                   float * __restrict__ t_ave, float * __restrict__ ft, float * __restrict__ mu_tend,                  
                   float rdx, float rdy, float dts, float epssm,               
                   float * __restrict__ dnw, float * __restrict__ fnm, float * __restrict__ fnp, float * __restrict__ rdnw,              
                   float * __restrict__ msfuy, float * __restrict__ msfvx_inv,                 
                   float * __restrict__ msftx, float * __restrict__ msfty,        
                   float * __restrict__ wdtn, float * __restrict__ dvdxi, float * __restrict__ dmdt,          
                   config_flags config,                      
                   int ids, int ide, int jds, int jde, int kds, int kde,            
                   int idim, int jdim, int kdim,     
                   int its, int ite, int jts, int jte, int kts, int kte )
                      
{
    int i, j, k;
    int i_start, i_end, j_start, j_end, k_start, k_end;
    
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


//        CALCULATION OF WW (dETA/dt)
    i = threadIdx.x + blockIdx.x * blockDim.x;
    j = blockIdx.y;
    
    if (j >= j_start && j <= j_end){

      if (i >= i_start && i <= i_end){

        float msftx_r = msftx[I2(i,j)];
        float msfty_r = msfty[I2(i,j)];
        float muv_r   = muv[I2(i,j)];
        float muv_r_1 = muv[I2(i,j+1)];
        float msfvx_inv_r   = msfvx_inv[I2(i,j)];
        float msfvx_inv_r_1 = msfvx_inv[I2(i,j+1)];
        float muu_r   = muu[I2(i,j)];
        float muu_r_1 = muu[I2(i+1,j)];
        float msfuy_r   = msfuy[I2(i,j)];
        float msfuy_r_1 = msfuy[I2(i+1,j)];
        
        
        
        dmdt[I2(i,j)] = 0.0f;
    
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
//  lines below find RHS terms at each level  set dmdt = sum over all levels
//
//  [don't divide the below by msfty until find ww, since dmdt is used in
//   the meantime]

        for(k = k_start; k <= k_end; k++){
            dvdxi[I3(i,k,j)] = msftx_r*msfty_r*(                       
                 rdy*( (v[I3(i,k,j+1)]+muv_r_1*v_1[I3(i,k,j+1)]*msfvx_inv_r_1) 
                      -(v[I3(i,k,j  )]+muv_r  *v_1[I3(i,k,j  )]*msfvx_inv_r) ) 
                +rdx*( (u[I3(i+1,k,j)]+muu_r_1*u_1[I3(i+1,k,j)]/msfuy_r_1) 
                      -(u[I3(i  ,k,j)]+muu_r  *u_1[I3(i  ,k,j)]/msfuy_r) ));
            dmdt[I2(i,j)] = dmdt[I2(i,j)] + dnw[k]*dvdxi[I3(i,k,j)];
        }

        muave[I2(i,j)] = mu[I2(i,j)];
        mu[I2(i,j)] = mu[I2(i,j)]+dts*(dmdt[I2(i,j)]+mu_tend[I2(i,j)]);
        mudf[I2(i,j)] = (dmdt[I2(i,j)]+mu_tend[I2(i,j)]); // save tendency for div damp filter
        muts[I2(i,j)] = mut[I2(i,j)]+mu[I2(i,j)];
        muave[I2(i,j)] = 0.5f*((1.0f+epssm)*mu[I2(i,j)]+(1.0f-epssm)*muave[I2(i,j)]);


        for(k = 1; k <= k_end; k++){
            ww[I3(i,k,j)] = ww[I3(i,k-1,j)]-dnw[k-1]*(dmdt[I2(i,j)]+dvdxi[I3(i,k-1,j)]
                    +mu_tend[I2(i,j)])/msfty_r;
        }

//  NOTE:  ww_1 (large timestep ww) is already coupled with the 
//         map scale factor

        for(k = 0; k <= k_end; k++){
            ww[I3(i,k,j)] = ww[I3(i,k,j)]-ww_1[I3(i,k,j)];
        }



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

        for(k = 0; k <= k_end; k++){
            t_ave[I3(i,k,j)] = t[I3(i,k,j)];
            t[I3(i,k,j)] = t[I3(i,k,j)] + msfty_r*dts*ft[I3(i,k,j)];
        }

        wdtn[I3(i,0,j)] = 0.0f;
        wdtn[I3(i,kde,j)] = 0.0f;



        for(k = 1; k <= k_end; k++){
                // for scalar eqn RHS term 3
            wdtn[I3(i,k,j)] = ww[I3(i,k,j)]*(fnm[k]*t_1[I3(i,k,j)]+fnp[k]
                    *t_1[I3(i,k-1,j)]);

        }


// scalar eqn, RHS terms 1, 2 and 3
// multiply by msfty to uncouple result for Theta from map scale factor

        for(k = 0; k <= k_end; k++){
            // multiplication by msfty uncouples result for Theta
            t[I3(i,k,j)] = t[I3(i,k,j)] - dts*msfty_r*(         
                    // multiplication by mx needed for RHS terms 1 & 2
                    msftx_r*(         
                    0.5f*rdy*                           
                    ( v[I3(i,k,j+1)]*(t_1[I3(i,k,j+1)]+t_1[I3(i,k,j  )])  
                     -v[I3(i,k,j  )]*(t_1[I3(i,k,j  )]+t_1[I3(i,k,j-1)]) ) 
                    + 0.5f*rdx*                                 
                    ( u[I3(i+1,k,j)]*(t_1[I3(i+1,k,j)]+t_1[I3(i  ,k,j)])  
                     -u[I3(i  ,k,j)]*(t_1[I3(i  ,k,j)]+t_1[I3(i-1,k,j)]) ) )
                    + rdnw[k]*( wdtn[I3(i,k+1,j)]-wdtn[I3(i,k,j)] ) );     

        }
        
      } // i
    } // j
        	
}


