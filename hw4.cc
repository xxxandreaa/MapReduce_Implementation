#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>
#include <vector>
#include <mpi.h>
#include "Scheduler.h"
#include "Worker.h"

int main(int argc, char *argv[]) {
    int provided_level ;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided_level) ;
    // args
    std::string job_name = std::string(argv[1]);
    int num_reducer = std::stoi(argv[2]);
    int delay = std::stoi(argv[3]);
    std::string input_filename = std::string(argv[4]);
    int chunk_size = std::stoi(argv[5]);
    std::string locality_config_filename = std::string(argv[6]);
    std::string output_dir = std::string(argv[7]);
    // rank, size
    int rank, size ;
    MPI_Comm_size(MPI_COMM_WORLD, &size) ;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank) ;

    if (rank == 0) { // job tracker
        Scheduler scheduler(size, job_name, num_reducer, delay, input_filename, chunk_size, locality_config_filename, output_dir) ;
        scheduler.Map_phase() ;
        scheduler.Reduce_phase() ;
        MPI_Barrier(MPI_COMM_WORLD) ;
    }
    else { // worker
        Worker worker(size, rank, job_name, num_reducer, delay, input_filename, chunk_size, output_dir) ;
        worker.Map_phase() ;
        worker.Reduce_phase() ;
        MPI_Barrier(MPI_COMM_WORLD) ;
    }

    MPI_Finalize() ;

    return 0 ;
}
