SHELL=/bin/sh
CLASS=S
SFILE=config/suite.def
GET_GPU_ARCH=../config/get_gpu_arch.sh

default: header
	@ $(SHELL) sys/print_instructions

	       
BT: bt
bt: header
	cd BT; $(GET_GPU_ARCH); $(MAKE) CLASS=$(CLASS)
		       
SP: sp		       
sp: header	       
	cd SP; $(GET_GPU_ARCH); $(MAKE) CLASS=$(CLASS)
		       
LU: lu		       
lu: header	       
	cd LU; $(GET_GPU_ARCH); $(MAKE) CLASS=$(CLASS)
		       
MG: mg		       
mg: header	       
	cd MG; $(GET_GPU_ARCH); $(MAKE) CLASS=$(CLASS)
		       
FT: ft		       
ft: header	       
	cd FT; $(GET_GPU_ARCH); $(MAKE) CLASS=$(CLASS)
		       
IS: is		       
is: header	       
	cd IS; $(GET_GPU_ARCH); $(MAKE) CLASS=$(CLASS)
		       
CG: cg		       
cg: header	       
	cd CG; $(GET_GPU_ARCH); $(MAKE) CLASS=$(CLASS)
		       
EP: ep		       
ep: header	       
	cd EP; $(GET_GPU_ARCH); $(MAKE) CLASS=$(CLASS)


# Awk script courtesy cmg@cray.com
suite:
	@ awk '{ if ($$1 !~ /^#/ &&  NF > 0)                              \
	printf "make %s CLASS=%s\n", $$1, $$2 }' $(SFILE)  \
	| $(SHELL)


# It would be nice to make clean in each subdirectory (the targets
# are defined) but on a really clean system this will won't work
# because those makefiles need config/make.def
clean:
	- rm -f core 
	- rm -f *~ */core */*~ */*.o */npbparams.hpp */*.obj */*.exe
	- rm -f sys/setparams sys/makesuite sys/setparams.hpp

cleanall: clean
	- rm -r bin/*

veryclean: clean
	- rm config/make.def config/suite.def Part*
	- rm bin/sp.* bin/lu.* bin/mg.* bin/ft.* bin/bt.* bin/is.* bin/ep.* bin/cg.*

header:
	@ $(SHELL) sys/print_header

kit: 
	- makekit -s100k -k30 * */* */*/*


