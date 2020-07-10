# 1 "stencil2d-mpi.F90"
! ******************************************************
!     Program: stencil2d
!      Author: Oliver Fuhrer
!       Email: oliverf@vulcan.com
!        Date: 20.05.2020
! Description: Simple stencil example (4th-order diffusion)
! ******************************************************

! Driver for apply_diffusion() that sets up fields and does timings
# 10
program main
    use mpi, only: MPI_COMM_WORLD
    use m_utils, only: timer_init, timer_start, timer_end, timer_get, is_master, num_rank, write_field_to_file
    use m_partitioner, only: Partitioner
    implicit none

    ! constants
    integer, parameter :: wp = 4

    ! local
    integer :: global_nx, global_ny, global_nz, num_iter
    logical :: scan

    integer :: num_halo = 2
    real (kind=wp) :: alpha = 1.0_wp / 32.0_wp

    real (kind=wp), allocatable :: in_field(:, :, :), f_in(:, :, :)
    real (kind=wp), allocatable :: out_field(:, :, :), f_out(:, :, :)

    integer :: timer_work
    real (kind=8) :: runtime

    integer :: cur_setup, num_setups = 1
    integer :: nx_setups(7) = (/ 16, 32, 48, 64, 96, 128, 192 /)
    integer :: ny_setups(7) = (/ 16, 32, 48, 64, 96, 128, 192 /)

    type(Partitioner) :: p



# 1 "/opt/cray/pe/perftools/7.1.1/include/pat_apif.h" 1
!
!       (C) COPYRIGHT CRAY INC.
!       UNPUBLISHED PROPRIETARY INFORMATION.
!       ALL RIGHTS RESERVED.
!
! -----------------------------------------------------------------
!
! PAT_API_OK or PAT_API_FAIL is returned by the API call to indicate
! success or failure.
!
# 11
       integer*4,parameter :: PAT_API_FAIL      =  0
       integer*4,parameter :: PAT_API_OK        =  1
!
! PAT_STATE_ON and PAT_STATE_OFF are states of data recording.
! They should be used for:
! 1.  the argument for PAT_record
!
! PAT_STATE_QUERY can be used as the argument for PAT_record
! to obtain the current state in the second argument.

       integer*4,parameter :: PAT_STATE_OFF     =  0
       integer*4,parameter :: PAT_STATE_ON      =  1
       integer*4,parameter :: PAT_STATE_QUERY   =  2

! These are the various counter categories upon which performance counters
! exist:
! o processor (traditional)
! o network router (e.g, Aries)
! o hardware accelerators (e.g., Nvidia GPUs)
! o Intel Running Average Power Level
! o Intel Uncore

       integer*4,parameter :: PAT_CTRS_CPU      =  0
       integer*4,parameter :: PAT_CTRS_NETWORK  =  1
       integer*4,parameter :: PAT_CTRS_ACCEL    =  2
       integer*4,parameter :: PAT_CTRS_RAPL     =  4
       integer*4,parameter :: PAT_CTRS_PM       =  5
       integer*4,parameter :: PAT_CTRS_UNCORE   =  6

! -----------------------------------------------------------------
!
!	Example of use:
!
! include "pat_apif.h"
!
! call pat_record(PAT_STATE_OFF, istat)
!
! call initialize()
!
! call pat_record(PAT_STATE_ON, istat)
!
! call pat_region_begin(1, "work", istat)
!
! call work1()
! call work2()
!
! call pat_region_end(1, istat)
!
! call finalize()
!
! end
!
! -----------------------------------------------------------------

      interface PAT_record
        subroutine PAT_record_i4 (cmd, istat)
          integer*4 cmd
          integer*4 istat
        end subroutine PAT_record_i4
        subroutine PAT_record_i8 (cmd, istat)
          integer*4 cmd
          integer*8 istat
        end subroutine PAT_record_i8
        subroutine PAT_record_ns (cmd)
          integer*4 cmd
        end subroutine PAT_record_ns
      end interface PAT_record

!   name	the name of the procedure that is the subject of this call
!		It must have been specified to pat_build to be
!		instrumented for tracing.
!   cmd		must be either PAT_STATE_ON or PAT_STATE_OFF.
!   istat	an integer variable that will contain a non-zero value
!		if the procedure could be enabled/disabled for tracing,
!		and otherwise zero.

      interface PAT_flush_buffer
        subroutine PAT_flush_buffer_i4 (nbytes, istat)
          integer*8 nbytes
          integer*4 istat
        end subroutine PAT_flush_buffer_i4
        subroutine PAT_flush_buffer_i8 (nbytes, istat)
          integer*8 nbytes
          integer*8 istat
        end subroutine PAT_flush_buffer_i8
        subroutine PAT_flush_buffer_ns (nbytes)
          integer*8 nbytes
        end subroutine PAT_flush_buffer_ns
      end interface PAT_flush_buffer

!   nbytes	an integer output variable for the number of bytes written
!   istat	an integer variable that will contain a non-zero value
!		if the data buffer is flushed and otherwise zero.

      interface PAT_region_begin
        subroutine PAT_region_begin_i4 (id, label, istat)
          integer*4 id
          character label
          integer*4 istat
        end subroutine PAT_region_begin_i4
        subroutine PAT_region_begin_i8 (id, label, istat)
          integer*8 id
          character label
          integer*8 istat
        end subroutine PAT_region_begin_i8
        subroutine PAT_region_begin_i4ns (id, label)
          integer*4 id
          character label
        end subroutine PAT_region_begin_i4ns
        subroutine PAT_region_begin_i8ns (id, label)
          integer*8 id
          character label
        end subroutine PAT_region_begin_i8ns
      end interface PAT_region_begin

!   id		an integer value that uniquely identifies this region
!   label	a character string used to label data from the region
!   istat	an integer variable that will contain a non-zero value
!		if data is recorded, and otherwise zero.

      interface PAT_region_end
        subroutine PAT_region_end_i4 (id, istat)
          integer*4 id
          integer*4 istat
        end subroutine PAT_region_end_i4
        subroutine PAT_region_end_i8 (id, istat)
          integer*8 id
          integer*8 istat
        end subroutine PAT_region_end_i8
        subroutine PAT_region_end_i4ns (id)
          integer*4 id
        end subroutine PAT_region_end_i4ns
        subroutine PAT_region_end_i8ns (id)
          integer*8 id
        end subroutine PAT_region_end_i8ns
      end interface PAT_region_end

!   id		the same value used in the matching call to PAT_region_begin
!   istat	an integer variable that will contain a non-zero value
!		if data is recorded, and otherwise zero.

      interface PAT_heap_stats
        subroutine PAT_heap_stats_i4 (istat)
          integer*4 istat
        end subroutine PAT_heap_stats_i4
        subroutine PAT_heap_stats_i8 (istat)
          integer*8 istat
        end subroutine PAT_heap_stats_i8
        subroutine PAT_heap_stats_ns ()
        end subroutine PAT_heap_stats_ns
      end interface PAT_heap_stats

!   istat	an integer variable that will contain a non-zero value
!		if data is recorded, and otherwise zero.

      interface PAT_counters
        subroutine PAT_counters_i4 (category, names, nevents, values,      &
     &                                   istat)
          integer*4 category
          character names (*)
          integer*8 values (*)
          integer*4 nevents
          integer*4 istat
        end subroutine PAT_counters_i4
        subroutine PAT_counters_i8 (category, names, nevents, values,      &
     &                                   istat)
          integer*4 category
          character names (*)
          integer*8 values (*)
          integer*8 nevents
          integer*8 istat
        end subroutine PAT_counters_i8
        subroutine PAT_counters_i4ns (category, names, nevents, values)
          integer*4 category
          character names (*)
          integer*8 values (*)
          integer*4 nevents
        end subroutine PAT_counters_i4ns
        subroutine PAT_counters_i8ns (category, names, nevents, values)
          integer*4 category
          character names (*)
          integer*8 values (*)
          integer*8 nevents
        end subroutine PAT_counters_i8ns
      end interface PAT_counters

!   category	HW category of counters
!   names	character strings of event names (returned)
!   values	array of event counts (returned)
!   nevents	number of event names (returned)
!   istat	an integer variable that will contain a non-zero value
!		if counter names/values are acquired, and otherwise zero.

!   The following routines can be used to instrument OpenMP constructs
!   for compilers that do not support automatic instrumentation.
!   These functions are optional but any uses of them must satisfy
!   the following requirements:
!
!   + If one member of a _enter/_exit or _begin/_end pair is called,
!     then the other must also be called.
!   + Calls to pairs of functions that end in _enter/_exit must
!     immediately precede and follow the relevant construct.
!   + Calls to pairs of functions that end in _begin/_end must
!     immediately precede and follow the code within the relevant
!     construct.
!   + For a given parallel region, all or none of the four
!     functions with prefix PAT_omp_parallel must be called.
!   + For a given 'sections' construct, all or none of the four
!     functions with prefix PAT_omp_section must be called, with
!     calls to the _begin/_end functions for each section.
!   + A 'single' construct should be treated as if it were a
!     'sections' construct with one section.

      interface

        subroutine PAT_omp_parallel_enter()
        end subroutine PAT_omp_parallel_enter

        subroutine PAT_omp_parallel_exit()
        end subroutine PAT_omp_parallel_exit

        subroutine PAT_omp_parallel_begin()
        end subroutine PAT_omp_parallel_begin

        subroutine PAT_omp_parallel_end()
        end subroutine PAT_omp_parallel_end

        subroutine PAT_omp_loop_enter()
        end subroutine PAT_omp_loop_enter

        subroutine PAT_omp_loop_exit()
        end subroutine PAT_omp_loop_exit

        subroutine PAT_omp_sections_enter()
        end subroutine PAT_omp_sections_enter

        subroutine PAT_omp_sections_exit()
        end subroutine PAT_omp_sections_exit

        subroutine PAT_omp_section_begin()
        end subroutine PAT_omp_section_begin

        subroutine PAT_omp_section_end()
        end subroutine PAT_omp_section_end

        subroutine PAT_omp_workshare_enter()
        end subroutine PAT_omp_workshare_enter

        subroutine PAT_omp_workshare_exit()
        end subroutine PAT_omp_workshare_exit

        subroutine PAT_omp_task_enter()
        end subroutine PAT_omp_task_enter

        subroutine PAT_omp_task_exit()
        end subroutine PAT_omp_task_exit

        subroutine PAT_omp_task_begin()
        end subroutine PAT_omp_task_begin

        subroutine PAT_omp_task_end()
        end subroutine PAT_omp_task_end

      end interface

! -----------------------------------------------------------------
# 40 "stencil2d-mpi.F90" 2
# 40 "stencil2d-mpi.F90"
    integer :: istat
    call PAT_record( PAT_STATE_OFF, istat )


# 44
    call init()

    if ( is_master() ) then
        write(*, '(a)') '# ranks nx ny ny nz num_iter time'
        write(*, '(a)') 'data = np.array( [ \'
    end if

    if ( scan ) num_setups = size(nx_setups) * size(ny_setups)
    do cur_setup = 0, num_setups - 1

        call timer_init()

        if ( scan ) then
            global_nx = nx_setups( modulo(cur_setup, size(ny_setups) ) + 1 )
            global_ny = ny_setups( cur_setup / size(ny_setups) + 1 )
        end if

        if ( is_master() ) &
            call setup()

        if ( .not. scan .and. is_master() ) &
            call write_field_to_file( in_field, num_halo, "in_field_mpi.dat" )

        p = Partitioner(MPI_COMM_WORLD, (/global_nx, global_ny, global_nz/), num_halo, periodic=(/.true., .true./))

        f_in = p%scatter(in_field, root=0)
        allocate(f_out, source=f_in)

        ! warmup caches
        call apply_diffusion( f_in, f_out, alpha, num_iter=1, p=p )

        ! time the actual work

# 77
        call PAT_record( PAT_STATE_ON, istat )

# 79
        timer_work = -999
        call timer_start('work', timer_work)

        call apply_diffusion( f_in, f_out, alpha, num_iter=num_iter, p=p )

        call timer_end( timer_work )

# 86
        call PAT_record( PAT_STATE_OFF, istat )


# 89
        call update_halo( f_out, p )

        out_field = p%gather(f_out, root=0)

        if ( .not. scan .and. is_master() ) &
            call write_field_to_file( out_field, num_halo, "out_field_mpi.dat" )

        if ( is_master() ) &
            call cleanup()

        runtime = timer_get( timer_work )
        if ( is_master() ) &
            write(*, '(a, i5, a, i5, a, i5, a, i5, a, i8, a, e15.7, a)') &
                '[', num_rank(), ',', global_nx, ',', global_ny, ',', global_nz, &
                ',', num_iter, ',', runtime, '], \'

    end do

    if ( is_master() ) then
        write(*, '(a)') '] )'
    end if

    call finalize()

contains


    ! Integrate 4th-order diffusion equation by a certain number of iterations.
    !
    !  in_field          -- input field (nx x ny x nz with halo in x- and y-direction)
    !  out_field         -- result (must be same size as in_field)
    !  alpha             -- diffusion coefficient (dimensionless)
    !  num_iter          -- number of iterations to execute
    !
    subroutine apply_diffusion( in_field, out_field, alpha, num_iter, p )
        implicit none

        ! arguments
        real (kind=wp), intent(inout) :: in_field(:, :, :)
        real (kind=wp), intent(inout) :: out_field(:, :, :)
        real (kind=wp), intent(in) :: alpha
        integer, intent(in) :: num_iter
        type(Partitioner), intent(in) :: p

        ! local
        real (kind=wp), save, allocatable :: tmp1_field(:, :, :)
        real (kind=wp), save, allocatable :: tmp2_field(:, :, :)
        integer :: iter, i, j, k
        integer :: dims(3), nx, ny, nz

        dims = p%shape()
        nx = dims(1) - 2 * p%num_halo()
        ny = dims(2) - 2 * p%num_halo()
        nz = dims(3)

        ! this is only done the first time this subroutine is called (warmup)
        ! or when the dimensions of the fields change
        if ( allocated(tmp1_field) .and. &
            any( shape(tmp1_field) /= (/nx + 2 * num_halo, ny + 2 * num_halo, nz /) ) ) then
            deallocate( tmp1_field, tmp2_field )
        end if
        if ( .not. allocated(tmp1_field) ) then
            allocate( tmp1_field(nx + 2 * num_halo, ny + 2 * num_halo, nz) )
            allocate( tmp2_field(nx + 2 * num_halo, ny + 2 * num_halo, nz) )
            tmp1_field = 0.0_wp
            tmp2_field = 0.0_wp
        end if

        do iter = 1, num_iter

            call update_halo( in_field, p )

            call laplacian( in_field, tmp1_field, num_halo, extend=1 )
            call laplacian( tmp1_field, tmp2_field, num_halo, extend=0 )

            ! do forward in time step
            do k = lbound(out_field,3), ubound(out_field,3)
            do j = lbound(out_field,2) + num_halo, ubound(out_field,2) - num_halo
            do i = lbound(out_field,1) + num_halo, ubound(out_field,1) - num_halo
                out_field(i, j, k) = in_field(i, j, k) - alpha * tmp2_field(i, j, k)
            end do
            end do
            end do

            ! copy out to in in caes this is not the last iteration
            if ( iter /= num_iter ) then
                do k = lbound(in_field,3), ubound(in_field,3)
                do j = lbound(in_field,2) + num_halo, ubound(in_field,2) - num_halo
                do i = lbound(in_field,1) + num_halo, ubound(in_field,1) - num_halo
                    in_field(i, j, k) = out_field(i, j, k)
                end do
                end do
                end do
            end if

        end do

    end subroutine apply_diffusion


    ! Compute Laplacian using 2nd-order centered differences.
    !
    !  in_field          -- input field (nx x ny x nz with halo in x- and y-direction)
    !  lap_field         -- result (must be same size as in_field)
    !  num_halo          -- number of halo points
    !  extend            -- extend computation into halo-zone by this number of points
    !
    subroutine laplacian( field, lap, num_halo, extend )
        implicit none

        ! argument
        real (kind=wp), intent(in) :: field(:, :, :)
        real (kind=wp), intent(inout) :: lap(:, :, :)
        integer, intent(in) :: num_halo, extend

        ! local
        integer :: i, j, k

        do k = lbound(field,3), ubound(field,3)
        do j = lbound(field,2) + num_halo - extend, ubound(field,2) - num_halo + extend
        do i = lbound(field,1) + num_halo - extend, ubound(field,1) - num_halo + extend
            lap(i, j, k) = -4._wp * field(i, j, k)      &
                + field(i - 1, j, k) + field(i + 1, j, k)  &
                + field(i, j - 1, k) + field(i, j + 1, k)
        end do
        end do
        end do

    end subroutine laplacian


    ! Update the halo-zone using an up/down and left/right strategy.
    !
    !  field             -- input/output field (nz x ny x nx with halo in x- and y-direction)
    !
    !  Note: corners are updated in the left/right phase of the halo-update
    !
    subroutine update_halo( field, p )
        use mpi !, only : MPI_FLOAT, MPI_DOUBLE, MPI_SUCCESS, MPI_STATUS_SIZE, &
                       !MPI_Irecv, MPI_Isend, MPI_Waitall
        use m_utils, only : error
        implicit none

        ! argument
        real (kind=wp), intent(inout) :: field(:, :, :)
        type(Partitioner), intent(in) :: p

        ! local
        integer :: i, j, k
        integer :: dims(3), nx, ny, nz
        integer :: lr_size, tb_size, dtype
        integer :: tb_req(4), lr_req(4)
        integer :: ierror, status(MPI_STATUS_SIZE, 4), icount
        real (kind=wp), save, allocatable :: sndbuf_l(:), sndbuf_r(:), sndbuf_t(:), sndbuf_b(:)
        real (kind=wp), save, allocatable :: rcvbuf_l(:), rcvbuf_r(:), rcvbuf_t(:), rcvbuf_b(:)

        ! set datatype
        if (wp == 4) then
            dtype = MPI_FLOAT
        else
            dtype = MPI_DOUBLE
        end if

        ! get dimensions
        dims = p%shape()
        nx = dims(1) - 2 * p%num_halo()
        ny = dims(2) - 2 * p%num_halo()
        nz = dims(3)

        ! compute sizes of buffers
        tb_size = nz * num_halo * nx
        lr_size = nz * num_halo * (ny + 2 * num_halo)

        ! this is only done the first time this subroutine is called (warmup)
        ! or when the dimensions of the fields change
        if ( allocated(sndbuf_l) .and. &
            ( ( size(sndbuf_l) /= lr_size ) .or. ( size(sndbuf_t) /= tb_size ) ) ) then
            deallocate( sndbuf_l, sndbuf_r, sndbuf_t, sndbuf_b )
            deallocate( rcvbuf_l, rcvbuf_r, rcvbuf_t, rcvbuf_b )
        end if
        if ( .not. allocated(sndbuf_l) ) then
            allocate( sndbuf_l(lr_size), sndbuf_r(lr_size), sndbuf_t(tb_size), sndbuf_b(tb_size) )
            allocate( rcvbuf_l(lr_size), rcvbuf_r(lr_size), rcvbuf_t(tb_size), rcvbuf_b(tb_size) )
            sndbuf_l = 0.0_wp; sndbuf_r = 0.0_wp; sndbuf_t = 0.0_wp; sndbuf_b = 0.0_wp
            rcvbuf_l = 0.0_wp; rcvbuf_r = 0.0_wp; rcvbuf_t = 0.0_wp; rcvbuf_b = 0.0_wp
        end if

        ! pre-post the receives
        call MPI_Irecv(rcvbuf_b, tb_size, dtype, p%bottom(), 1000, p%comm(), tb_req(1), ierror)
        call error(ierror /= MPI_SUCCESS, 'Problem with MPI_Irecv(bottom)', code=ierror)
        call MPI_Irecv(rcvbuf_t, tb_size, dtype, p%top(), 1001, p%comm(), tb_req(2), ierror)
        call error(ierror /= MPI_SUCCESS, 'Problem with MPI_Irecv(top)', code=ierror)
        call MPI_Irecv(rcvbuf_l, lr_size, dtype, p%left(), 1002, p%comm(), lr_req(1), ierror)
        call error(ierror /= MPI_SUCCESS, 'Problem with MPI_Irecv(left)', code=ierror)
        call MPI_Irecv(rcvbuf_r, lr_size, dtype, p%right(), 1003, p%comm(), lr_req(2), ierror)
        call error(ierror /= MPI_SUCCESS, 'Problem with MPI_Irecv(right)', code=ierror)

        ! pack the tb-buffers (without corners)
        icount = 0
        do k = 1, nz
        do j = 1, num_halo
        do i = 1 + num_halo, nx + num_halo
            icount = icount + 1
            sndbuf_t(icount) = field(i, j + ny, k)
            sndbuf_b(icount) = field(i, j + num_halo, k)
        end do
        end do
        end do

        ! send lr-buffers
        call MPI_Isend(sndbuf_t, tb_size, dtype, p%top(), 1000, p%comm(), tb_req(3), ierror)
        call error(ierror /= MPI_SUCCESS, 'Problem with MPI_Isend(top)', code=ierror)
        call MPI_Isend(sndbuf_b, tb_size, dtype, p%bottom(), 1001, p%comm(), tb_req(4), ierror)
        call error(ierror /= MPI_SUCCESS, 'Problem with MPI_Isend(bottom)', code=ierror)

        ! wait for lr-comm to finish
        call MPI_Waitall(4, tb_req, status, ierror)
        call error(ierror /= MPI_SUCCESS, 'Problem with MPI_Waitall(tb)', code=ierror)

        ! pack the lr-buffers (including corners)
        icount = 0
        do k = 1, nz
        do j = 1, ny + 2 * num_halo
        do i = 1, num_halo
            icount = icount + 1
            sndbuf_r(icount) = field(i + nx, j, k)
            sndbuf_l(icount) = field(i + num_halo, j, k)
        end do
        end do
        end do

        call MPI_Isend(sndbuf_r, lr_size, dtype, p%right(), 1002, p%comm(), lr_req(3), ierror)
        call error(ierror /= MPI_SUCCESS, 'Problem with MPI_Isend(right)', code=ierror)
        call MPI_Isend(sndbuf_l, lr_size, dtype, p%left(), 1003, p%comm(), lr_req(4), ierror)
        call error(ierror /= MPI_SUCCESS, 'Problem with MPI_Isend(left)', code=ierror)

        ! unpack the tb-buffers (without corners)
        icount = 0
        do k = 1, nz
        do j = 1, num_halo
        do i = 1 + num_halo, nx + num_halo
            icount = icount + 1
            field(i, j, k) = rcvbuf_b(icount)
            field(i, j + num_halo + ny, k) = rcvbuf_t(icount)
        end do
        end do
        end do

        ! wait for tb-comm to finish
        call MPI_Waitall(4, lr_req, status, ierror)
        call error(ierror /= MPI_SUCCESS, 'Problem with MPI_Waitall(lr)', code=ierror)

        ! unpack the lr-buffers (with corners)
        icount = 0
        do k = 1, nz
        do j = 1, ny + 2 * num_halo
        do i = 1, num_halo
            icount = icount + 1
            field(i, j, k) = rcvbuf_l(icount)
            field(i + num_halo + nx, j, k) = rcvbuf_r(icount)
        end do
        end do
        end do

    end subroutine update_halo


    ! initialize at program start
    ! (init MPI, read command line arguments)
    subroutine init()
        use mpi, only : MPI_INIT
        use m_utils, only : error
        implicit none

        ! local
        integer :: ierror

        ! initialize MPI environment
        call MPI_INIT(ierror)
        call error(ierror /= 0, 'Problem with MPI_INIT', code=ierror)

        call read_cmd_line_arguments()

    end subroutine init


    ! setup everything before work
    ! (init timers, allocate memory, initialize fields)
    subroutine setup()
        implicit none

        ! local
        integer :: i, j, k

        allocate( in_field(global_nx + 2 * num_halo, global_ny + 2 * num_halo, global_nz) )
        in_field = 0.0_wp
        do k = 1 + global_nz / 4, 3 * global_nz / 4
        do j = 1 + num_halo + global_ny / 4, num_halo + 3 * global_ny / 4
        do i = 1 + num_halo + global_nx / 4, num_halo + 3 * global_nx / 4
            in_field(i, j, k) = 1.0_wp
        end do
        end do
        end do

        allocate( out_field(global_nx + 2 * num_halo, global_ny + 2 * num_halo, global_nz) )
        out_field = in_field

    end subroutine setup


    ! read and parse the command line arguments
    ! (read values, convert type, ensure all required arguments are present,
    !  ensure values are reasonable)
    subroutine read_cmd_line_arguments()
        use m_utils, only : error
        implicit none

        ! local
        integer iarg, num_arg
        character(len=256) :: arg, arg_val

        ! setup defaults
        global_nx = -1
        global_ny = -1
        global_nz = -1
        num_iter = -1
        scan = .false.

        num_arg = command_argument_count()
        iarg = 1
        do while ( iarg <= num_arg )
            call get_command_argument(iarg, arg)
            select case (arg)
            case ("--nx")
                call error(iarg + 1 > num_arg, "Missing value for --nx argument")
                call get_command_argument(iarg + 1, arg_val)
                call error(arg_val(1:1) == "-", "Missing value for --nx argument")
                read(arg_val, *) global_nx
                iarg = iarg + 1
            case ("--ny")
                call error(iarg + 1 > num_arg, "Missing value for --ny argument")
                call get_command_argument(iarg + 1, arg_val)
                call error(arg_val(1:1) == "-", "Missing value for --ny argument")
                read(arg_val, *) global_ny
                iarg = iarg + 1
            case ("--nz")
                call error(iarg + 1 > num_arg, "Missing value for --nz argument")
                call get_command_argument(iarg + 1, arg_val)
                call error(arg_val(1:1) == "-", "Missing value for --nz argument")
                read(arg_val, *) global_nz
                iarg = iarg + 1
            case ("--num_iter")
                call error(iarg + 1 > num_arg, "Missing value for --num_iter argument")
                call get_command_argument(iarg + 1, arg_val)
                call error(arg_val(1:1) == "-", "Missing value for --num_iter argument")
                read(arg_val, *) num_iter
                iarg = iarg + 1
            case ("--scan")
                scan = .true.
            case default
                call error(.true., "Unknown command line argument encountered: " // trim(arg))
            end select
            iarg = iarg + 1
        end do

        ! make sure everything is set
        if (.not. scan) then
            call error(global_nx == -1, 'You have to specify nx')
            call error(global_ny == -1, 'You have to specify ny')
        end if
        call error(global_nz == -1, 'You have to specify nz')
        call error(num_iter == -1, 'You have to specify num_iter')

        ! check consistency of values
        if (.not. scan) then
            call error(global_nx < 0 .or. global_nx > 1024*1024, "Please provide a reasonable value of nx")
            call error(global_ny < 0 .or. global_ny > 1024*1024, "Please provide a reasonable value of ny")
        end if
        call error(global_nz < 0 .or. global_nz > 1024, "Please provide a reasonable value of nz")
        call error(num_iter < 1 .or. num_iter > 1024*1024, "Please provide a reasonable value of num_iter")

    end subroutine read_cmd_line_arguments


    ! cleanup at end of work
    ! (report timers, free memory)
    subroutine cleanup()
        implicit none

        deallocate(in_field, out_field)

    end subroutine cleanup


    ! finalize at end of program
    ! (finalize MPI)
    subroutine finalize()
        use mpi, only : MPI_FINALIZE
        use m_utils, only : error
        implicit none

        integer :: ierror

        call MPI_FINALIZE(ierror)
        call error(ierror /= 0, 'Problem with MPI_FINALIZE', code=ierror)

    end subroutine finalize


end program main
