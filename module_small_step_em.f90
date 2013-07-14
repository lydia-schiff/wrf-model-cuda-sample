MODULE module_small_step_em

USE module_configure, ONLY : grid_config_rec_type

CONTAINS

SUBROUTINE advance_mu_t( ww, ww_1, u, u_1, v, v_1,            &
                         mu, mut, muave, muts, muu, muv,      &
                         mudf, t, t_1,                        &
                         t_ave, ft, mu_tend,                  &
                         rdx, rdy, dts, epssm,                &
                         dnw, fnm, fnp, rdnw,                 &
                         msfuy, msfvx_inv,                    &
                         msftx, msfty,                        &
                         config_flags,                        &
                         ids, ide, jds, jde, kde,             &
                         ims, ime, jms, jme, kms, kme,        &
                         its, ite, jts, jte, kts, kte        )

  IMPLICIT NONE  ! religion first

! stuff coming in

  TYPE(grid_config_rec_type), INTENT(IN   ) :: config_flags

  INTEGER,      INTENT(IN   )    :: ids,ide, jds,jde, kde
  INTEGER,      INTENT(IN   )    :: ims,ime, jms,jme, kms,kme
  INTEGER,      INTENT(IN   )    :: its,ite, jts,jte, kts,kte

  REAL, DIMENSION( ims:ime , kms:kme, jms:jme ),   &
            INTENT(IN   ) ::                       &
                                              u,   &
                                              v,   &
                                              u_1, &
                                              v_1, &
                                              t_1, &
                                              ft

  REAL, DIMENSION( ims:ime , kms:kme, jms:jme ),      &
            INTENT(INOUT) ::                          &
                                              ww,     &
                                              ww_1,   &
                                              t,      &
                                              t_ave
                                              
  REAL, DIMENSION( ims:ime , jms:jme ),    INTENT(IN   ) :: muu,  &
                                                            muv,  &
                                                            mut,  &
                                                            msfuy,&
                                                            msfvx_inv,&
                                                            msftx,&
                                                            msfty,&
                                                            mu_tend

  REAL, DIMENSION( ims:ime , jms:jme ),    INTENT(  OUT) :: muave, &
                                                            muts,  &
                                                            mudf

  REAL, DIMENSION( ims:ime , jms:jme ),    INTENT(INOUT) :: mu

  REAL, DIMENSION( kms:kme ),              INTENT(IN   ) :: fnm,    &
                                                            fnp,    &
                                                            dnw,    &
                                                            rdnw


  REAL,                                    INTENT(IN   ) :: rdx,    &
                                                            rdy,    &
                                                            dts,    &
                                                            epssm

!  Local arrays from the stack (note tile size)

  REAL, DIMENSION (its:ite, kts:kte) :: wdtn, dvdxi
  REAL, DIMENSION (its:ite) :: dmdt

  INTEGER :: i,j,k, i_start, i_end, j_start, j_end, k_start, k_end
  INTEGER :: i_endu, j_endv

!<DESCRIPTION>
!
!  advance_mu_t advances the explicit perturbation theta equation and the mass
!  conservation equation.  In addition, the small timestep omega is updated,
!  and some quantities needed in other places are squirrelled away.
!
!</DESCRIPTION>

!  now, the real work.
!  set the loop bounds taking into account boundary conditions.

  i_start = its
  i_end   = min(ite,ide-1)
  j_start = jts
  j_end   = min(jte,jde-1)
  k_start = kts
  k_end   = kte-1
  IF ( .NOT. config_flags%periodic_x )THEN
     IF ( config_flags%specified .or. config_flags%nested ) then
       i_start = max(its,ids+1)
       i_end   = min(ite,ide-2)
     ENDIF
  ENDIF
  IF ( config_flags%specified .or. config_flags%nested ) then
     j_start = max(jts,jds+1)
     j_end   = min(jte,jde-2)
  ENDIF

  i_endu = ite
  j_endv = jte

!        CALCULATION OF WW (dETA/dt)
   DO j = j_start, j_end

     DO i=i_start, i_end
            dmdt(i) = 0.
     ENDDO
!  NOTE:  mu is not coupled with the map scale factor.
!         ww (omega) IS coupled with the map scale factor.
!         Being coupled with the map scale factor means 
!           multiplication by (1/msft) in this case.

!  Comments on map scale factors
!  ADT eqn 47: 
!  partial drho/dt = -mx*my[partial d/dx(rho u/my) + partial d/dy(rho v/mx)]
!                    -partial d/dz(rho w)
!  with rho -> mu, dividing by my, and with partial d/dnu(rho nu/my [=ww])
!  as the final term (because we're looking for d_nu_/dt)
!
!  begin by integrating with respect to nu from bottom to top
!  BCs are ww=0 at both
!  final term gives 0
!  first term gives Integral([1/my]partial d mu/dt) over total column = dm/dt
!  RHS remaining is Integral(-mx[partial d/dx(mu u/my) + 
!                                partial d/dy(mu v/mx)]) over column
!  lines below find RHS terms at each level then set dmdt = sum over all levels
!
!  [don't divide the below by msfty until find ww, since dmdt is used in
!   the meantime]

     DO k=k_start, k_end
     DO i=i_start, i_end
         dvdxi(i,k) = msftx(i,j)*msfty(i,j)*(                                      &
                     rdy*( (v(i,k,j+1)+muv(i,j+1)*v_1(i,k,j+1)*msfvx_inv(i,j+1))   &
                          -(v(i,k,j  )+muv(i,j  )*v_1(i,k,j  )*msfvx_inv(i,j  )) ) &
                    +rdx*( (u(i+1,k,j)+muu(i+1,j)*u_1(i+1,k,j)/msfuy(i+1,j))       &
                          -(u(i,k,j  )+muu(i  ,j)*u_1(i,k,j  )/msfuy(i  ,j)) ))
        dmdt(i)    = dmdt(i) + dnw(k)*dvdxi(i,k)
     ENDDO
     ENDDO
     
     DO i=i_start, i_end
       muave(i,j) = mu(i,j)
       mu(i,j) = mu(i,j)+dts*(dmdt(i)+mu_tend(i,j))
       mudf(i,j) = (dmdt(i)+mu_tend(i,j)) ! save tendency for div damp filter
       muts(i,j) = mut(i,j)+mu(i,j)
       muave(i,j) =.5*((1.+epssm)*mu(i,j)+(1.-epssm)*muave(i,j))
     ENDDO

     DO k=2,k_end
     DO i=i_start, i_end
       ww(i,k,j)=ww(i,k-1,j)-dnw(k-1)*(dmdt(i)+dvdxi(i,k-1)+mu_tend(i,j))/msfty(i,j)
     ENDDO
     END DO

!  NOTE:  ww_1 (large timestep ww) is already coupled with the 
!         map scale factor

     DO k=1,k_end
     DO i=i_start, i_end
       ww(i,k,j)=ww(i,k,j)-ww_1(i,k,j)
     END DO
     END DO

   ENDDO
    open(unit = 666, ACCESS="STREAM", file = "muave_before_theta.bin", convert="big_endian", action="write")
    write (666) muave
    close (666)
    open(unit = 666, ACCESS="STREAM", file = "mu_before_theta.bin", convert="big_endian", action="write")
    write (666) mu
    close (666)
    open(unit = 666, ACCESS="STREAM", file = "mudf_before_theta.bin", convert="big_endian", action="write")
    write (666) mudf
    close (666)
    open(unit = 666, ACCESS="STREAM", file = "muts_before_theta.bin", convert="big_endian", action="write")
    write (666) muts
    close (666)
    open(unit = 666, ACCESS="STREAM", file = "ww_before_theta.bin", convert="big_endian", action="write")
    write (666) ww
    close (666)
! CALCULATION OF THETA

! NOTE: theta'' is not coupled with the map-scale factor, 
!       while the theta'' tendency is coupled (i.e., mult by 1/msft)

! Comments on map scale factors
! BUT NOTE THAT both are mass coupled
! in flux form equations (Klemp et al.) Theta = mu*theta
!
! scalar eqn: partial d/dt(rho q/my) = -mx[partial d/dx(q rho u/my) + 
!                                          partial d/dy(q rho v/mx)]
!                                      - partial d/dz(q rho w/my)
! with rho -> mu, and with partial d/dnu(q rho nu/my) as the final term
!
! adding previous tendency contribution which was map scale factor coupled
! (had been divided by msfty)
! need to uncouple before updating uncoupled Theta (by adding)

   DO j=j_start, j_end
     DO k=1,k_end
     DO i=i_start, i_end
       t_ave(i,k,j) = t(i,k,j)
       t   (i,k,j) = t(i,k,j) + msfty(i,j)*dts*ft(i,k,j)
     END DO
     END DO
   ENDDO   

   DO j=j_start, j_end

     DO i=i_start, i_end
       wdtn(i,1  )=0.
       wdtn(i,kde)=0.
     ENDDO

     DO k=2,k_end
     DO i=i_start, i_end
        ! for scalar eqn RHS term 3
        wdtn(i,k)= ww(i,k,j)*(fnm(k)*t_1(i,k  ,j)+fnp(k)*t_1(i,k-1,j))
     ENDDO
     ENDDO

! scalar eqn, RHS terms 1, 2 and 3
! multiply by msfty to uncouple result for Theta from map scale factor

     DO k=1,k_end
     DO i=i_start, i_end
       ! multiplication by msfty uncouples result for Theta
       t(i,k,j) = t(i,k,j) - dts*msfty(i,j)*(              &
                          ! multiplication by mx needed for RHS terms 1 & 2
                          msftx(i,j)*(                     &
               .5*rdy*                                     &
              ( v(i,k,j+1)*(t_1(i,k,j+1)+t_1(i,k, j ))     &
               -v(i,k,j  )*(t_1(i,k, j )+t_1(i,k,j-1)) )   &
             + .5*rdx*                                     &
              ( u(i+1,k,j)*(t_1(i+1,k,j)+t_1(i  ,k,j))     &
               -u(i  ,k,j)*(t_1(i  ,k,j)+t_1(i-1,k,j)) ) ) &
             + rdnw(k)*( wdtn(i,k+1)-wdtn(i,k) ) )       
     ENDDO
     ENDDO

   ENDDO

END SUBROUTINE advance_mu_t

END MODULE module_small_step_em
