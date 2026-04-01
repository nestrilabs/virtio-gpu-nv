# Guest kernel module Makefile
#
# Usage:
#   make KDIR=/lib/modules/$(uname -r)/build
#
# Or, when cross-compiling for the guest kernel:
#   make KDIR=/path/to/guest-kernel ARCH=x86_64 CROSS_COMPILE=x86_64-linux-gnu-

KDIR ?= /lib/modules/$(shell uname -r)/build

# Extra CFLAGS for better diagnostics.
ccflags-y := -Wall -Wno-unused-function

.PHONY: all clean

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
