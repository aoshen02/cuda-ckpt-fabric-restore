CFLAGS ?= -O2 -Wall -Wextra

libfabric_restore_shim.so: fabric_restore_shim.c
	$(CC) $(CFLAGS) -shared -fPIC -o $@ $< -ldl -lpthread

clean:
	rm -f libfabric_restore_shim.so

.PHONY: clean
