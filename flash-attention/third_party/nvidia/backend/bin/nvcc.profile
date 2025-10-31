
TOP              = $(_HERE_)/..

CICC_PATH        = $(TOP)/nvvm/bin
NVVMIR_LIBRARY_DIR = $(TOP)/nvvm/libdevice

LD_LIBRARY_PATH += $(TOP)/lib:
PATH            += $(CICC_PATH):$(_HERE_):

INCLUDES        +=  "-I$(TOP)/$(_TARGET_DIR_)/include" $(_SPACE_)

LIBRARIES        =+ $(_SPACE_) "-L$(TOP)/$(_TARGET_DIR_)/lib$(_TARGET_SIZE_)/stubs" "-L$(TOP)/$(_TARGET_DIR_)/lib$(_TARGET_SIZE_)"

CUDAFE_FLAGS    +=
PTXAS_FLAGS     +=
