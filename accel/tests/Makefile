CC = g++
CFLAGS = -Wall -Wextra -O2
VERILATOR_FLAGS = --cc --trace

UNIT_TESTS = unit/test_mac8.cpp unit/test_pe.cpp unit/test_csr.cpp
INTEGRATION_TESTS = integration/test_systolic_array.cpp

all: unit_tests integration_tests

unit_tests: $(UNIT_TESTS)
	$(CC) $(CFLAGS) $(VERILATOR_FLAGS) -o unit_tests $(UNIT_TESTS)

integration_tests: $(INTEGRATION_TESTS)
	$(CC) $(CFLAGS) $(VERILATOR_FLAGS) -o integration_tests $(INTEGRATION_TESTS)

run: all
	./unit_tests
	./integration_tests

clean:
	rm -f unit_tests integration_tests