module davidson_orthogonalization_selftest_m
  use iso_c_binding, only: c_double, c_int
  use precision, only: dp
  use tdhf_lib, only: rpanewb
  use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan, ieee_positive_inf, ieee_is_finite
  implicit none
  private
  public :: davidson_orthogonalization_selftest
contains
  subroutine davidson_orthogonalization_selftest(error, checks, failures) bind(C)
    real(c_double), intent(out) :: error
    integer(c_int), intent(out) :: checks, failures
    real(dp), parameter :: scales(4) = [1.0e-300_dp, 1.0e-12_dp, 1.0_dp, 1.0e300_dp]
    real(dp) :: basis(6,6), candidates(6,4), expected(6,6), gram(6,6), local_error
    integer :: mode, scale_index, kind, nvec, novec, status, wanted, j
    logical :: tda, accepted

    error=0; checks=0; failures=0
    expected=0
    expected(1,1)=1
    expected(2:3,2)=[2.0_dp,3.0_dp]/sqrt(13.0_dp)
    expected(2:4,3)=[3.0_dp,-2.0_dp,5.0_dp]/sqrt(38.0_dp)
    expected(5,4)=1; expected(6,5)=1
    do mode=1,2
      tda=mode==1
      wanted=merge(3,5,tda)
      do scale_index=1,size(scales)
        basis=0; basis(1,1)=1
        candidates=0
        candidates(1:3,1)=[1.0_dp,2.0_dp,3.0_dp]
        candidates(1:4,2)=[-4.0_dp,3.0_dp,-2.0_dp,5.0_dp]
        candidates(1,3)=7; candidates(5,3)=2
        candidates(1,4)=-3; candidates(6,4)=4
        candidates=candidates*scales(scale_index)
        nvec=1; novec=-1
        call rpanewb(2,basis,candidates,novec,nvec,status,tda)
        checks=checks+1
        if (status/=0 .or. nvec/=wanted .or. novec/=1) failures=failures+1
        if (.not.all(ieee_is_finite(basis))) then
          failures=failures+1
        else
          local_error=maxval(abs(basis(:,1:wanted)-expected(:,1:wanted)))
          gram=matmul(transpose(basis),basis)
          do j=1,wanted
            gram(j,j)=gram(j,j)-1
          end do
          local_error=max(local_error,maxval(abs(gram)))
          error=max(error,local_error)
          if (local_error>1.0e-12_dp) failures=failures+1
        end if
      end do
      do scale_index=1,size(scales)
        do kind=0,5
          basis=0; basis(1,1)=1; candidates=0
          select case(kind)
          case(0) ! Exactly zero update.
          case(1)
            candidates(2,1)=ieee_value(0.0_dp,ieee_quiet_nan)
          case(2)
            candidates(2,1)=ieee_value(0.0_dp,ieee_positive_inf)
          case(3) ! Exactly dependent, at every amplitude.
            candidates(1,1)=scales(scale_index)
          case(4) ! Relative remainder below the dependence cutoff.
            candidates(1:2,1)=scales(scale_index)*[1.0_dp,1.0e-12_dp]
          case(5) ! Small but resolvable independent direction.
            candidates(1:2,1)=scales(scale_index)*[1.0_dp,1.0e-6_dp]
          end select
          candidates(:,2)=candidates(:,1)
          nvec=1; novec=-1
          call rpanewb(1,basis,candidates,novec,nvec,status,tda)
          accepted=kind==5
          checks=checks+1
          if (nvec/=merge(2,1,accepted) .or. status/=merge(0,3,accepted) .or. novec/=1) &
            failures=failures+1
          if (.not.all(ieee_is_finite(basis))) then
            failures=failures+1
          else
            ! Original basis is always retained; accepted near-dependent
            ! candidates have the unique orthogonal direction +e2.
            local_error=abs(basis(1,1)-1)+sum(abs(basis(2:,1)))
            if (accepted) then
              local_error=max(local_error,abs(basis(2,2)-1),abs(basis(1,2)),maxval(abs(basis(3:,2))))
            end if
            error=max(error,local_error)
            if (local_error>1.0e-12_dp) failures=failures+1
          end if
        end do
      end do
    end do
  end subroutine davidson_orthogonalization_selftest
end module davidson_orthogonalization_selftest_m
