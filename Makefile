KERNEL_VERSION = linux-6.19.11
KERNEL_REMOTE = https://cdn.kernel.org/pub/linux/kernel/v6.x/$(KERNEL_VERSION).tar.xz
KERNEL_TARBALL = tarballs/$(KERNEL_VERSION).tar.xz
KERNEL_SOURCES = $(KERNEL_VERSION)
KERNEL_C_BUNDLE = kernel.c

NESTRI_DRIVERS_SRC = $(shell find nvgpu -type f 2>/dev/null)

ABI_VERSION = 5
FULL_VERSION = 5.3.0
TIMESTAMP = "Tue Mar 10 13:28:56 CET 2026"

KERNEL_FLAGS = KBUILD_BUILD_TIMESTAMP=$(TIMESTAMP)
KERNEL_FLAGS += KBUILD_BUILD_USER=root
KERNEL_FLAGS += KBUILD_BUILD_HOST=libkrunfw

HOSTARCH = $(shell uname -m)
OS = $(shell uname -s)
ifeq ($(ARCH),)
	GUESTARCH := $(HOSTARCH)
	STRIP := strip
else ifeq ($(ARCH),arm64)
	GUESTARCH := aarch64
	CC := $(CROSS_COMPILE)gcc
	STRIP := $(CROSS_COMPILE)strip
else ifeq ($(ARCH),riscv)
	GUESTARCH := riscv64
	CC := $(CROSS_COMPILE)gcc
	STRIP := $(CROSS_COMPILE)strip
else
	GUESTARCH := $(ARCH)
	CC := $(CROSS_COMPILE)gcc
	STRIP := $(CROSS_COMPILE)strip
endif

KBUNDLE_TYPE_x86_64 = vmlinux
KBUNDLE_TYPE_aarch64 = Image
KBUNDLE_TYPE_riscv64 = Image

KERNEL_BINARY_x86_64 = $(KERNEL_SOURCES)/vmlinux
KERNEL_BINARY_aarch64 = $(KERNEL_SOURCES)/arch/arm64/boot/Image
KERNEL_BINARY_riscv64 = $(KERNEL_SOURCES)/arch/riscv/boot/Image

KRUNFW_BINARY_Linux = libkrunfw$(VARIANT).so.$(FULL_VERSION)
KRUNFW_SONAME_Linux = libkrunfw$(VARIANT).so.$(ABI_VERSION)
KRUNFW_BASE_Linux = libkrunfw$(VARIANT).so
SONAME_Linux = -Wl,-soname,$(KRUNFW_SONAME_Linux)

KRUNFW_BINARY_Darwin = libkrunfw.$(ABI_VERSION).dylib
KRUNFW_SONAME_Darwin = libkrunfw.$(ABI_VERSION).dylib
KRUNFW_BASE_Darwin = libkrunfw.dylib
SONAME_Darwin =

LIBDIR_Linux = lib64
LIBDIR_Darwin = lib

ifeq ($(PREFIX),)
    PREFIX := /usr/local
endif

.PHONY: all install clean

all: $(KRUNFW_BINARY_$(OS))

$(KERNEL_TARBALL):
	@mkdir -p tarballs
	curl $(KERNEL_REMOTE) -o $(KERNEL_TARBALL)

$(KERNEL_SOURCES): $(KERNEL_TARBALL)
	tar xf $(KERNEL_TARBALL)
	# --- Nestri driver ---
	cp -r nvgpu $(KERNEL_SOURCES)/drivers/virtio/nvgpu
	@echo 'source "drivers/virtio/nvgpu/Kconfig"' >> $(KERNEL_SOURCES)/drivers/virtio/Kconfig
	@echo 'obj-y += nvgpu/' >> $(KERNEL_SOURCES)/drivers/virtio/Makefile
	# --- EOF Nestri driver ---
	cp config-libkrunfw$(VARIANT)_$(GUESTARCH) $(KERNEL_SOURCES)/.config
	cd $(KERNEL_SOURCES) ; $(MAKE) olddefconfig

$(KERNEL_BINARY_$(GUESTARCH)): $(KERNEL_SOURCES) $(NESTRI_DRIVERS_SRC)
	@echo "Syncing updated drivers into the kernel tree..."
	@rsync -a --delete nvgpu/ $(KERNEL_SOURCES)/drivers/virtio/nvgpu/
	cd $(KERNEL_SOURCES) ; rm -f .version ; $(MAKE) $(MAKEFLAGS) $(KERNEL_FLAGS)

ifeq ($(OS),Darwin)
$(KERNEL_C_BUNDLE):
	@echo "Building on macOS, using ./build_on_krunvm.sh"
	./build_on_krunvm.sh
else
$(KERNEL_C_BUNDLE): $(KERNEL_BINARY_$(GUESTARCH))
	@echo "Generating $(KERNEL_C_BUNDLE) from $(KERNEL_BINARY_$(GUESTARCH))..."
	@python3 bin2cbundle.py -t $(KBUNDLE_TYPE_$(GUESTARCH)) $(KERNEL_BINARY_$(GUESTARCH)) kernel.c
endif

$(KRUNFW_BINARY_$(OS)): $(KERNEL_C_BUNDLE) $(QBOOT_C_BUNDLE) $(INITRD_C_BUNDLE)
	$(CC) -fPIC -DABI_VERSION=$(ABI_VERSION) -shared $(SONAME_$(OS)) -o $@ $(KERNEL_C_BUNDLE) $(QBOOT_C_BUNDLE) $(INITRD_C_BUNDLE)
ifeq ($(OS),Linux)
	$(STRIP) $(KRUNFW_BINARY_$(OS))
endif

install:
	install -d $(DESTDIR)$(PREFIX)/$(LIBDIR_$(OS))/
	install -m 755 $(KRUNFW_BINARY_$(OS)) $(DESTDIR)$(PREFIX)/$(LIBDIR_$(OS))/
ifeq ($(OS),Darwin)
	cd $(DESTDIR)$(PREFIX)/$(LIBDIR_$(OS))/ ; ln -sf $(KRUNFW_BINARY_$(OS)) $(KRUNFW_BASE_$(OS))
else
	cd $(DESTDIR)$(PREFIX)/$(LIBDIR_$(OS))/ ; ln -sf $(KRUNFW_BINARY_$(OS)) $(KRUNFW_SONAME_$(OS)) ; ln -sf $(KRUNFW_SONAME_$(OS)) $(KRUNFW_BASE_$(OS))
endif

clean:
	rm -fr $(KERNEL_SOURCES) $(KERNEL_C_BUNDLE) $(QBOOT_C_BUNDLE) $(INITRD_C_BUNDLE) $(KRUNFW_BINARY_$(OS))
