CC = mpicc
CXX = mpicxx
CXXFLAGS = -std=c++20 -O3 -lm -pthread 
CFLAGS = -std=c++20 -O3 -lm -pthread

hw4: Worker.cc Scheduler.cc hw4.cc
	$(CXX) $(CXXFLAGS) -o $@ hw4.cc Worker.cc Scheduler.cc

.PHONY: clean
clean:
	-rm -f hw4
	-rm -r ./output ./inter
	mkdir ./output ./inter