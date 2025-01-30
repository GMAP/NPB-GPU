SHELL=/bin/sh
BENCHMARK=bt
BENCHMARKU=BT

include ../config/make.def

OBJS =	bt.o \
	${COMMON}/c_print_results.o ${COMMON}/c_timers.o ${COMMON}/c_wtime.o

include ../sys/make.common

${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${C_LIB}

bt.o:             bt.cpp npbparams.hpp
	${CCOMPILE} bt.cpp
#	${CCOMPILE} --offload-arch=gfx1101 -mllvm --amdgpu-mode-register=1 bt.cpp
clean:
	- rm -f *.o *~ mputil*
	- rm -f  npbparams.hpp core
