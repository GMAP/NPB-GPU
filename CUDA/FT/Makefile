SHELL=/bin/sh
BENCHMARK=ft
BENCHMARKU=FT

include ../config/make.def

OBJS = ft.o ${COMMON}/c_${RAND}.o ${COMMON}/c_print_results.o \
       ${COMMON}/c_timers.o ${COMMON}/c_wtime.o

include ../sys/make.common

${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -o ${PROGRAM} ${OBJS} ${C_LIB}

ft.o:             ft.cu npbparams.hpp
	${CCOMPILE} ft.cu

clean:
	- rm -f *.o *~ mputil*
	- rm -f ft npbparams.hpp core
