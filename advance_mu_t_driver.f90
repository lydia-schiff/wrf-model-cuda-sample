  program advance_mu_t_driver

    use module_configure, ONLY : grid_config_rec_type
    use module_small_step_em
!    use omp_lib

    implicit none

!    integer ij, id, threads, height, jts_orig, jte_orig
    integer ids, ide, jds, jde, kde      ! domain dims
    integer ims, ime, jms, jme, kms, kme ! memory dims
    integer its, ite, jts, jte, kts, kte ! tile dims
    integer i_start, i_end, j_start, j_end

    real, dimension(:, :, :), allocatable :: grid_ww, ww1
    real, dimension(:, :, :), allocatable :: grid_u_2, grid_u_save
    real, dimension(:, :, :), allocatable :: grid_v_2, grid_v_save
    real, dimension(:, :), allocatable :: grid_mu_2, grid_mut
    real, dimension(:, :), allocatable :: muave, grid_muts
    real, dimension(:, :), allocatable :: grid_muu, grid_muv
    real, dimension(:, :), allocatable :: grid_mudf
    real, dimension(:, :, :), allocatable :: grid_t_2, grid_t_save, t_2save
    real, dimension(:, :, :), allocatable :: t_tend
    real, dimension(:, :), allocatable :: mu_tend
    real, dimension(:), allocatable :: grid_dnw, grid_fnm, grid_fnp, grid_rdnw
    real, dimension(:, :), allocatable :: grid_msfuy
    real, dimension(:, :), allocatable :: grid_msfvx_inv, grid_msftx
    real, dimension(:, :), allocatable :: grid_msfty

    real :: grid_rdx, grid_rdy, dts_rk, grid_epssm
    TYPE(grid_config_rec_type) :: config_flags

    real time
    integer hz, clock1, clock0,delta

    CALL chdir("/data2/WRFV3_Input_Output/V3.4.1/dyn_em/advance_mu_t")

    CALL read_value(ids, "ids.bin")
    CALL read_value(ide, "ide.bin")
    CALL read_value(jds, "jds.bin")
    CALL read_value(jde, "jde.bin")
    CALL read_value(kde, "kde.bin")

    CALL read_value(ims, "ims.bin")
    CALL read_value(ime, "ime.bin")
    CALL read_value(jms, "jms.bin")
    CALL read_value(jme, "jme.bin")
    CALL read_value(kms, "kms.bin")
    CALL read_value(kme, "kme.bin")

    CALL read_value(its, "its.bin")
    CALL read_value(ite, "ite.bin")
    CALL read_value(jts, "jts.bin")
    CALL read_value(jte, "jte.bin")
    CALL read_value(kts, "kts.bin")
    CALL read_value(kte, "kte.bin")

    CALL read_value_real(grid_rdx, "grid_rdx.bin")
    CALL read_value_real(grid_rdy, "grid_rdy.bin")
    CALL read_value_real(dts_rk, "dts_rk.bin")
    CALL read_value_real(grid_epssm, "grid_epssm.bin")





    CALL chdir("/data2/WRFV3_Input_Output/V3.4.1/dyn_em/spec_bdy_dry")


    open(unit = 666, ACCESS="STREAM", file = "config_flags.bin", convert="big_endian", action="read")
    read (666) config_flags
    close (666)
	    open(unit = 666, ACCESS="STREAM", file = "config_flags_spec_bdy_width.bin", convert="big_endian", action="write")
    write (666) config_flags%spec_bdy_width
    close (666)
    
    	    open(unit = 666, ACCESS="STREAM", file = "config_flags_periodic_x.bin", convert="big_endian", action="write")
    write (666) config_flags%periodic_x
    close (666)
    
    	    open(unit = 666, ACCESS="STREAM", file = "config_flags_nested.bin", convert="big_endian", action="write")
    write (666) config_flags%nested
    close (666)

	CALL exit




! 1d - in
    allocate( grid_dnw (kms:kme))
    CALL read_data_1d( grid_dnw, "grid_dnw.bin", kms, kme)

    allocate( grid_fnm (kms:kme))
    CALL read_data_1d( grid_fnm, "grid_fnm.bin", kms, kme)

    allocate( grid_fnp (kms:kme))
    CALL read_data_1d( grid_fnp, "grid_fnp.bin", kms, kme)

    allocate( grid_rdnw (kms:kme))
    CALL read_data_1d( grid_rdnw, "grid_rdnw.bin", kms, kme)

! 3d - in
    allocate( grid_u_2 (ims:ime, kms:kme, jms:jme))
    CALL read_data( grid_u_2, "grid_u_2.bin", ims, ime, jms, jme, kms, kme)

    allocate( grid_u_save (ims:ime, kms:kme, jms:jme))
    CALL read_data( grid_u_save, "grid_u_save.bin", ims, ime, jms, jme, kms, kme)

    allocate( grid_v_2 (ims:ime, kms:kme, jms:jme))
    CALL read_data( grid_v_2, "grid_v_2.bin", ims, ime, jms, jme, kms, kme)

    allocate( grid_v_save (ims:ime, kms:kme, jms:jme))
    CALL read_data( grid_v_save, "grid_v_save.bin", ims, ime, jms, jme, kms, kme)

    allocate( grid_t_save (ims:ime, kms:kme, jms:jme))
    CALL read_data( grid_t_save, "grid_t_save.bin", ims, ime, jms, jme, kms, kme)

    allocate( t_tend (ims:ime, kms:kme, jms:jme))
    CALL read_data( t_tend, "t_tend.bin", ims, ime, jms, jme, kms, kme)

! 2d - in
    allocate( grid_mut(ims:ime, jms:jme) )
    CALL read_data_2d( grid_mut, "grid_mut.bin", ims, ime, jms, jme)

    allocate( grid_muu(ims:ime, jms:jme) )
    CALL read_data_2d( grid_muu, "grid_muu.bin", ims, ime, jms, jme)

    allocate( grid_muv(ims:ime, jms:jme) )
    CALL read_data_2d( grid_muv, "grid_muv.bin", ims, ime, jms, jme)

    allocate( mu_tend(ims:ime, jms:jme) )
    CALL read_data_2d( mu_tend, "mu_tend.bin", ims, ime, jms, jme)

    allocate( grid_msfuy(ims:ime, jms:jme) )
    CALL read_data_2d( grid_msfuy, "grid_msfuy.bin", ims, ime, jms, jme)

    allocate( grid_msfvx_inv(ims:ime, jms:jme) )
    CALL read_data_2d( grid_msfvx_inv, "grid_msfvx_inv.bin", ims, ime, jms, jme)

    allocate( grid_msftx(ims:ime, jms:jme) )
    CALL read_data_2d( grid_msftx, "grid_msftx.bin", ims, ime, jms, jme)

    allocate( grid_msfty(ims:ime, jms:jme) )
    CALL read_data_2d( grid_msfty, "grid_msfty.bin", ims, ime, jms, jme)

! 2d - in/out
    allocate( grid_mu_2(ims:ime, jms:jme) )
    CALL read_data_2d( grid_mu_2, "grid_mu_2.bin", ims, ime, jms, jme)

! 3d - in/out
    allocate( grid_ww (ims:ime, kms:kme, jms:jme))
    CALL read_data( grid_ww, "grid_ww.bin", ims, ime, jms, jme, kms, kme)

    allocate( ww1 (ims:ime, kms:kme, jms:jme))
    CALL read_data( ww1, "ww1.bin", ims, ime, jms, jme, kms, kme)

    allocate( grid_t_2 (ims:ime, kms:kme, jms:jme))
    CALL read_data( grid_t_2, "grid_t_2.bin", ims, ime, jms, jme, kms, kme)

    allocate( t_2save (ims:ime, kms:kme, jms:jme))
    CALL read_data( t_2save, "t_2save.bin", ims, ime, jms, jme, kms, kme)

! 2d - out
    allocate( muave(ims:ime, jms:jme) )
    allocate( grid_muts(ims:ime, jms:jme) )
    allocate( grid_mudf(ims:ime, jms:jme) )

!    jts_orig = jts
!    jte_orig = jte

    CALL system_clock(count_rate=hz)
    CALL system_clock(count=clock0)

!!$OMP PARALLEL DO &
!!$OMP PRIVATE ( ij, id, threads, height, jts, jte )

!   DO ij = 1 , omp_get_max_threads()
!        id=omp_get_thread_num()+1 ! returns ID of this thread   
!        threads=OMP_GET_NUM_THREADS() ! returns the number of threads in parallel region
!        height = (jte_orig-jts_orig+1)/threads
!        if (id .eq. 1) then
!           jts = jts_orig
!           jte = jts_orig + height
!        else if (id .eq. threads) then
!           jts = jts_orig + (threads-1) * height + 1
!           jte = jte_orig
!        else
!           jts = jts_orig + (id-1) * height + 1
!           jte = jts_orig + id * height
!        endif

      CALL advance_mu_t( grid_ww, ww1, grid_u_2, grid_u_save, grid_v_2, grid_v_save, &
                         grid_mu_2, grid_mut, muave, grid_muts, grid_muu, grid_muv,    &
                         grid_mudf,                                                    &
                         grid_t_2, grid_t_save, t_2save, t_tend,                       &
                         mu_tend,                                                      &
                         grid_rdx, grid_rdy, dts_rk, grid_epssm,                       &
                         grid_dnw, grid_fnm, grid_fnp, grid_rdnw,                      &
                         grid_msfuy, grid_msfvx_inv,                                   &
                         grid_msftx, grid_msfty,                                       &
                         config_flags,                                                 &
                         ids, ide, jds, jde, kde,                                      &
                         ims, ime, jms, jme, kms, kme,                                 &
                         its, ite, jts, jte, kts, kte )


!   end do
!!$OMP END PARALLEL DO

    CALL system_clock(count=clock1)
    delta = clock1 - clock0
    time = real(delta)/(real(hz))
    print*, "advance_mu_t computing time(msec):"," ",time*1000.0

!    jts = jts_orig
!    jte = jte_orig

    i_start = max(its,ids+1)
    i_end   = min(ite,ide-2)
    j_start = max(jts,jds+1)
    j_end   = min(jte,jde-2)

    CALL compare(     grid_ww   , "grid_ww_output.bin"   , ims, ime, jms, jme, kms, kme)
    CALL compare(         ww1   , "ww1_output.bin"       , ims, ime, jms, jme, kms, kme)
    CALL compare(     grid_t_2  , "grid_t_2_output.bin"  , ims, ime, jms, jme, kms, kme)
    CALL compare(      t_2save  , "t_2save_output.bin"   , ims, ime, jms, jme, kms, kme)
    CALL compare_2d(   grid_mu_2, "grid_mu_2_output.bin" , ims, ime, jms, jme)
    CALL compare_2d_t (    muave, "muave_output.bin"     , ims, ime, jms, jme, i_start, i_end, j_start, j_end)
    CALL compare_2d_t( grid_muts, "grid_muts_output.bin" , ims, ime, jms, jme, i_start, i_end, j_start, j_end)
    CALL compare_2d_t( grid_mudf, "grid_mudf_output.bin" , ims, ime, jms, jme, i_start, i_end, j_start, j_end)

    deallocate( grid_ww )
    deallocate( ww1 )
    deallocate( grid_u_2 )
    deallocate( grid_u_save )
    deallocate( grid_v_2 )
    deallocate( grid_v_save )
    deallocate( grid_mu_2 )
    deallocate( grid_mut )
    deallocate( muave )
    deallocate( grid_muts )
    deallocate( grid_muu )
    deallocate( grid_muv )
    deallocate( grid_mudf )
    deallocate( grid_t_2 )
    deallocate( grid_t_save )
    deallocate( t_2save )
    deallocate( t_tend )
    deallocate( mu_tend )
    deallocate( grid_dnw )
    deallocate( grid_fnm )
    deallocate( grid_fnp )
    deallocate( grid_rdnw )
    deallocate( grid_msfuy )
    deallocate( grid_msfvx_inv )
    deallocate( grid_msftx )
    deallocate( grid_msfty )

  end program advance_mu_t_driver


  subroutine compare_2d_t(data, file_name, ims, ime, jms, jme, its, ite, jts, jte)
    IMPLICIT NONE

    INTEGER,  INTENT(IN) :: ims, ime, jms, jme, its, ite, jts, jte
    real, DIMENSION( ims:ime, jms:jme ), INTENT(IN) :: data
    CHARACTER *(*), INTENT(IN) :: file_name

    INTEGER i, j
    real max_abs_err
    real max_rel_err
    real rel_err, abs_err

    real, dimension(:, :), allocatable :: data_orig
    allocate(data_orig    (ims:ime, jms:jme))

    open(unit = 666, ACCESS="STREAM", file=file_name, convert="big_endian", action="read")
    read (666) data_orig
    close (666)

    max_abs_err = 0.0
    max_rel_err = 0.0

    do i = its, ite
      do j = jts, jte

        if( abs(data(i,j)) .ne. 0.0 .and. abs(data_orig(i,j)) .ne. 0.0 ) then
          rel_err = (abs(data(i,j) - data_orig(i,j)) ) / MAX(abs(data(i,j)), abs(data_orig(i,j)) ) ;
        else 
          rel_err = MAX( abs(data(i,j)), abs(data_orig(i,j)) )
        end if
        if(rel_err .gt. max_rel_err) then
          max_rel_err = rel_err
        end if

        abs_err = abs(data(i,j) - data_orig(i,j))
        if(abs_err .gt. max_abs_err) then
          max_abs_err = abs_err
        end if

      end do
    end do

    print *
    print *, file_name
    print *, 'max relative error: ', max_rel_err
    print *, 'max absolute error: ', max_abs_err

    deallocate(data_orig)
  end subroutine compare_2d_t

  subroutine alloc_read_data_3d(data, file_name, ims, ime, jms, jme, kms, kme)
    implicit none

    integer,  intent(in) :: ims, ime, jms, jme, kms, kme
    real, dimension(:, :, :), allocatable, intent(out) :: data
    character *(*), intent(in) :: file_name

    allocate( data (ims:ime, kms:kme, jms:jme))
    CALL read_data( data, file_name, ims, ime, jms, jme, kms, kme)
  end subroutine alloc_read_data_3d

  subroutine read_value(value, file_name)
    implicit none

    integer, intent(out) :: value
    character *(*), intent(in) :: file_name

    open(unit = 666, access="stream", file=file_name, convert="big_endian", action="read")
      read (666) value
    close (666)
  end subroutine read_value

  subroutine read_value_logic(value, file_name)
    implicit none

    logical, intent(out) :: value
    character *(*), intent(in) :: file_name

    open(unit = 666, access="stream", file=file_name, convert="big_endian", action="read")
      read (666) value
    close (666)
  end subroutine read_value_logic

  subroutine read_value_real(value, file_name)
    implicit none

    real, intent(out) :: value
    character *(*), intent(in) :: file_name

    open(unit = 666, access="stream", file=file_name, convert="big_endian", action="read")
      read (666) value
    close (666)            
  end subroutine read_value_real

  subroutine read_data(data, file_name, ims, ime, jms, jme, kms, kme)
    implicit none

    integer,  intent(in) :: ims, ime, jms, jme, kms, kme
    real, dimension( ims:ime, kms:kme, jms:jme ), intent(out) :: data
    character *(*), intent(in) :: file_name

    open(unit = 666, access="stream", file=file_name, convert="big_endian", action="read")
    read (666) data
    close (666)

  end subroutine read_data

  subroutine read_data_2d(data, file_name, ims, ime, jms, jme)
    implicit none

    integer,  intent(in) :: ims, ime, jms, jme
    real, dimension( ims:ime, jms:jme ), intent(out) :: data
    character *(*), intent(in) :: file_name

    open(unit = 666, access="stream", file=file_name, convert="big_endian", action="read")
    read (666) data
    close (666)
  end subroutine read_data_2d

  subroutine read_data_int_2d(data, file_name, ims, ime, jms, jme)
    implicit none

    integer,  intent(in) :: ims, ime, jms, jme
    integer, dimension( ims:ime, jms:jme ), intent(out) :: data
    character *(*), intent(in) :: file_name

    open(unit = 666, access="stream", file=file_name, convert="big_endian", action="read")
    read (666) data
    close (666)
  end subroutine read_data_int_2d

  subroutine read_data_1d(data, file_name, ims, ime)
    implicit none

    integer,  intent(in) :: ims, ime
    real, dimension( ims:ime), intent(out) :: data
    character *(*), intent(in) :: file_name

    open(unit = 666, access="stream", file=file_name, convert="big_endian", action="read")
    read (666) data
    close (666)
  end subroutine read_data_1d

  subroutine write_data(data, file_name, ims, ime, jms, jme, kms, kme)
    implicit none

    integer,  intent(in) :: ims, ime, jms, jme, kms, kme
    real, dimension( ims:ime, kms:kme, jms:jme ), intent(in) :: data
    character *(*), intent(in) :: file_name

    open(unit = 666, access="stream", file=file_name, convert="big_endian", action="write")
    write (666) data
    close (666)
  end subroutine write_data

  subroutine write_data_2d(data, file_name, ims, ime, jms, jme)
    implicit none

    integer,  intent(in) :: ims, ime, jms, jme
    real, dimension( ims:ime, jms:jme ), intent(in) :: data
    character *(*), intent(in) :: file_name

    open(unit = 666, access="stream", file=file_name, convert="big_endian", action="write")
    write (666) data
    close (666)
  end subroutine write_data_2d

  subroutine write_data_int_2d(data, file_name, ims, ime, jms, jme)
    implicit none

    integer,  intent(in) :: ims, ime, jms, jme
    integer, dimension( ims:ime, jms:jme ), intent(in) :: data
    character *(*), intent(in) :: file_name

    open(unit = 666, access="stream", file=file_name, convert="big_endian", action="write")
    write (666) data
    close (666)
  end subroutine write_data_int_2d

  subroutine write_data_1d(data, file_name, ims, ime)
    implicit none

    integer,  intent(in) :: ims, ime
    real, dimension( ims:ime ), intent(in) :: data
    character *(*), intent(in) :: file_name

    open(unit = 666, access="stream", file=file_name, convert="big_endian", action="write")
    write (666) data
    close (666)
  end subroutine write_data_1d


  subroutine compare_2d_integer(data, file_name, ims, ime, jms, jme, its, ite, jts, jte)
    IMPLICIT NONE

    INTEGER,  INTENT(IN) :: ims, ime, jms, jme, its, ite, jts, jte
    real, DIMENSION( ims:ime, jms:jme ), INTENT(IN) :: data
    CHARACTER *(*), INTENT(IN) :: file_name

    INTEGER i, j
    real max_abs_err
    real max_rel_err
    real rel_err, abs_err

    real, dimension(:, :), allocatable :: data_orig
    allocate(data_orig    (ims:ime, jms:jme))

    open(unit = 666, ACCESS="STREAM", file=file_name, convert="big_endian", action="read")
    read (666) data_orig
    close (666)

    max_abs_err = 0.0
    max_rel_err = 0.0

    do i = its, ite
      do j = jts, jte

        if( abs(data(i,j)) .ne. 0.0 .and. abs(data_orig(i,j)) .ne. 0.0 ) then
          rel_err = (abs(data(i,j) - data_orig(i,j)) ) / MAX(abs(data(i,j)), abs(data_orig(i,j)) ) ;
        else 
          rel_err = MAX( abs(data(i,j)), abs(data_orig(i,j)) )
        end if
        if(rel_err .gt. max_rel_err) then
          max_rel_err = rel_err
        end if

        abs_err = abs(data(i,j) - data_orig(i,j))
        if(abs_err .gt. max_abs_err) then
          max_abs_err = abs_err
        end if

      end do
    end do

    print *
    print *, file_name
    print *, 'max relative error: ', max_rel_err
    print *, 'max absolute error: ', max_abs_err

    deallocate(data_orig)
  end subroutine compare_2d_integer


  subroutine compare_2d(data, file_name, ims, ime, jms, jme)
    IMPLICIT NONE

    INTEGER,  INTENT(IN) :: ims, ime, jms, jme
    real, DIMENSION( ims:ime, jms:jme ), INTENT(IN) :: data
    CHARACTER *(*), INTENT(IN) :: file_name

    INTEGER i, j
    real max_abs_err
    real max_rel_err
    real rel_err, abs_err

    real, dimension(:, :), allocatable :: data_orig
    allocate(data_orig    (ims:ime, jms:jme))

    open(unit = 666, ACCESS="STREAM", file=file_name, convert="big_endian", action="read")
    read (666) data_orig
    close (666)

    max_abs_err = 0.0
    max_rel_err = 0.0

    do i = ims, ime
      do j = jms, jme

        if( abs(data(i,j)) .ne. 0.0 .and. abs(data_orig(i,j)) .ne. 0.0 ) then
          rel_err = (abs(data(i,j) - data_orig(i,j)) ) / MAX(abs(data(i,j)), abs(data_orig(i,j)) ) ;
        else 
          rel_err = MAX( abs(data(i,j)), abs(data_orig(i,j)) )
        end if
        if(rel_err .gt. max_rel_err) then
          max_rel_err = rel_err
        end if

        abs_err = abs(data(i,j) - data_orig(i,j))
        if(abs_err .gt. max_abs_err) then
          max_abs_err = abs_err
        end if

      end do
    end do

    print *
    print *, file_name
    print *, 'max relative error: ', max_rel_err
    print *, 'max absolute error: ', max_abs_err

    deallocate(data_orig)
  end subroutine compare_2d

  subroutine compare(data, file_name, ims, ime, jms, jme, kms, kme)
    IMPLICIT NONE

    INTEGER,  INTENT(IN) :: ims, ime, kms, kme, jms, jme
    real, DIMENSION( ims:ime, kms:kme, jms:jme ), INTENT(IN) :: data
    CHARACTER *(*), INTENT(IN) :: file_name

    INTEGER i, k, j
    real max_abs_err
    real max_rel_err
    real rel_err, abs_err

    real, dimension(:, :, :), allocatable :: data_orig
    allocate(data_orig    (ims:ime, kms:kme, jms:jme))

    open(unit = 666, ACCESS="STREAM", file=file_name, convert="big_endian", action="read")
    read (666) data_orig
    close (666)

    max_abs_err = 0.0
    max_rel_err = 0.0

    do i = ims, ime
      do k = kms, kme
        do j = jms, jme

          if( abs(data(i,k,j)) .ne. 0.0 .and. abs(data_orig(i,k,j)) .ne. 0.0 ) then
            rel_err = (abs(data(i,k,j) - data_orig(i,k,j)) ) / MAX(abs(data(i,k,j)), abs(data_orig(i,k,j)) ) ;
          else 
            rel_err = MAX( abs(data(i,k,j)), abs(data_orig(i,k,j)) )
          end if
          if(rel_err .gt. max_rel_err) then
            max_rel_err = rel_err
          end if

          abs_err = abs(data(i,k,j) - data_orig(i,k,j))
          if(abs_err .gt. max_abs_err) then
            max_abs_err = abs_err
          end if

        end do
      end do
    end do

    print *
    print *, file_name
    print *, 'max relative error: ', max_rel_err
    print *, 'max absolute error: ', max_abs_err

    deallocate(data_orig)
  end subroutine compare


