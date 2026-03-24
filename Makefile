CXX ?= g++
CXXFLAGS ?= -std=c++20 -O3 -DNDEBUG -march=native -flto -Wall -Wextra -Wshadow -Wconversion -pedantic
LDFLAGS ?= -flto
INCLUDES ?= -Iinclude

SRC := src/main.cpp src/engine.cpp

all: aquachess

aquachess: $(SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $(SRC) $(LDFLAGS)

clean:
	rm -f aquachess

.PHONY: all clean
