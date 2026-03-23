CXX ?= g++
CXXFLAGS ?= -std=c++20 -O3 -DNDEBUG -march=native -flto -Wall -Wextra -Wshadow -Wconversion -pedantic
LDFLAGS ?= -flto

all: aquachess

aquachess: src/main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f aquachess

.PHONY: all clean
