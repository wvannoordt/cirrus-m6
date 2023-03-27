spade   := ${SPADE}
target := vdf.x

ifndef MPI_ENABLE
MPI_ENABLE := 1
endif

ifeq (${MPI_ENABLE},1)
cc := $(shell which mpicxx)
else
cc := $(shell which g++)
endif

compflags :=
compflags += -DMPI_ENABLE=${MPI_ENABLE}

flags :=
flags += -fconcepts-diagnostics-depth=3
ifeq (${sanny},1)
flags += -fsanitize=undefined,address -fstack-protector-all
endif

HYWALL_PATH := ${HYWALL}
ifneq (${HYWALL_10},)
HYWALL_PATH := ${HYWALL_10}
endif

main:
	${MAKE} -C ${HYWALL_PATH} -f makefile
	${cc} --version
	${cc} -std=c++20 -g -O3 ${flags} ${compflags} -I${spade}/src -I${PTL}/include -I${HYWALL_PATH}/include main.cc -o ${target} -L${PTL}/lib -lPTL -L${HYWALL_PATH}/lib -lHyWall

run: main
	./${target}

clean:
	${MAKE} -C ${HYWALL_PATH} -f makefile clean
	rm -f ${target}
