#include "Scheduler.h"
#include <iostream>
#include <mpi.h>

typedef struct arg
{
    Scheduler *scheduler;
    int rank;
} Arg;


void *dispatch_map_thread_func(void *args) {
    Arg *arg = (Arg*)args;
    int rank = arg->rank;
    Scheduler *scheduler = arg->scheduler;

    int buf;
    int task_chunkIdx;

    while(true) {
        MPI_Recv(&buf, 1, MPI_INT, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        task_chunkIdx = scheduler->Dispatch_mapper(rank);
        MPI_Send(&task_chunkIdx, 1, MPI_INT, rank, 0, MPI_COMM_WORLD);
        if (task_chunkIdx == 0) { // done
            break ;
        }
    }

    pthread_exit(NULL);
}

void *check_map_thread_func(void *args) {
    Arg *arg = (Arg*)args;
    int rank = arg->rank;
    Scheduler *scheduler = arg->scheduler;

    int task_chunkIdx;
    int done = 0;

    while (!done) {
        MPI_Recv(&task_chunkIdx, 1, MPI_INT, rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        done = scheduler->Map_task_complete(rank, task_chunkIdx);
    }

    pthread_exit(NULL);
}

void *dispatch_reduce_thread_func(void *args) {
    Arg *arg = (Arg*)args;
    int rank = arg->rank;
    Scheduler *scheduler = arg->scheduler;

    int reducer_task, buf;

    while (true) {
        MPI_Recv(&buf, 1, MPI_INT, rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        reducer_task = scheduler->Dispatch_reducer(rank) ;
        MPI_Send(&reducer_task, 1, MPI_INT, rank, 2, MPI_COMM_WORLD) ;
        if (reducer_task < 0) {
            break;
        }
        MPI_Recv(&reducer_task, 1, MPI_INT, rank, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        scheduler->Reduce_task_complete(reducer_task);
    }

    pthread_exit(NULL);
}

Scheduler::Scheduler(int size, std::string job_name, int num_reducer, int delay, std::string input_filename, int chunk_size, std::string locality_config_filename, std::string output_dir)
:size(size), num_reducer(num_reducer), job_name(job_name), locality_config_filename(locality_config_filename), output_dir(output_dir)
{
    // init
    worker_num = size-1;
    reducer_to_dispatch = 0;
    log_filename = output_dir + job_name + "-log.out";

    cpu_set_t cpuset;
    sched_getaffinity(0, sizeof(cpuset), &cpuset);
    int ncpus = CPU_COUNT(&cpuset);
    log_file.open(log_filename);
    log_file << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    log_file << ",Start_Job," << job_name << "," << size << "," << ncpus << "," << num_reducer << "," << delay << ","
             << input_filename << "," << chunk_size << "," << locality_config_filename << "," << output_dir << "\n";
    start_job_time = std::chrono::steady_clock::now();

    // read locality file
    std::ifstream locality_file(locality_config_filename);
    std::string line ;
    
    while (getline(locality_file, line)) {
        int chunk = std::stoi(line.substr(0, line.find(" ", 0)));
        int node = std::stoi(line.substr(line.find(" ", 0), line.npos));
        if (node > worker_num) {
            node %= worker_num;
        }

        std::pair<int, int> loc(chunk, node);
        locality.push_back(loc);
    }
    locality_file.close();
}

Scheduler::~Scheduler(){}

void Scheduler::Map_phase() {
    // threads init
    pthread_mutex_init(&lock, NULL);

    dispatch_threads = new pthread_t[worker_num];
    dispatch_task_num = new int[size]{0} ;
    task_dispatch_time = new std::chrono::steady_clock::time_point[locality.size()+1];

    args = (void*)new Arg[worker_num];
    for (int i=0; i<worker_num; i++) {
        ((Arg*)args)[i].rank = i+1;
        ((Arg*)args)[i].scheduler = this;

        pthread_create(&dispatch_threads[i], NULL, dispatch_map_thread_func, (void*)&(((Arg*)args)[i]));
    }
    check_threads = new pthread_t[worker_num];
    for (int i=0; i<worker_num; i++){
        pthread_create(&check_threads[i], NULL, check_map_thread_func, (void*)&(((Arg*)args)[i]));
    }

    // wait all dispatch/check threads
    for (int i=0; i<worker_num; i++) {
        pthread_join(dispatch_threads[i], NULL);
    }
    for (int i=0; i<worker_num; i++) {
        pthread_join(check_threads[i], NULL);
    }

    // shuffle
    int total_words_num;
    int dummy = 0;
    MPI_Reduce(&dummy, &total_words_num, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD) ;
    std::chrono::time_point t_start_shuffle = std::chrono::steady_clock::now();
    log_file << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    log_file << ",Start_Shuffle," << total_words_num << "\n" ;

    MPI_Barrier(MPI_COMM_WORLD);
    int shuffle_time = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - t_start_shuffle).count();
    log_file << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    log_file << ",Finish_Shuffle," << shuffle_time << "\n";

    pthread_mutex_destroy(&lock);
    delete [] dispatch_threads;
    delete [] check_threads;
    delete [] dispatch_task_num;
    delete [] task_dispatch_time;
}

int Scheduler::Dispatch_mapper(int target_rank) {
    pthread_mutex_lock(&lock);
    // all dispatched
    if (locality.size() == 0) {
        pthread_mutex_unlock(&lock);
        return 0 ;
    }

    int task_chunkIdx;
    // find locality
    for (auto i=locality.begin(); i!=locality.end(); i++) {
        if ((*i).second == target_rank) {
            task_chunkIdx = (*i).first;
            locality.erase(i);

            log_file << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() ;
            log_file << ",Dispatch_MapTask," << task_chunkIdx << "," << target_rank << "\n";
            task_dispatch_time[task_chunkIdx] = std::chrono::steady_clock::now();
            dispatch_task_num[target_rank]++;

            pthread_mutex_unlock(&lock);
            return task_chunkIdx;
        }
    }
    // no locality exists
    task_chunkIdx = locality[0].first;
    locality.erase(locality.begin());
    
    log_file << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    log_file << ",Dispatch_MapTask," << task_chunkIdx << "," << target_rank << "\n" ;
    task_dispatch_time[task_chunkIdx] = std::chrono::steady_clock::now();
    dispatch_task_num[target_rank]++;

    pthread_mutex_unlock(&lock);
    return -task_chunkIdx;
}

int Scheduler::Map_task_complete(int rank, int task_chunkIdx) {
    int retval;
    pthread_mutex_lock(&lock);
    if (task_chunkIdx != 0) {
        dispatch_task_num[rank]--;
        int exe_time = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - task_dispatch_time[task_chunkIdx]).count();

        log_file << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        log_file << ",Complete_MapTask," << task_chunkIdx << "," << exe_time << "\n";
    }

    if (dispatch_task_num[rank] == 0 && locality.size() == 0) { // all task recv
        retval = 1;
    }
    else retval = 0;

    pthread_mutex_unlock(&lock);

    return retval;
}

void Scheduler::Reduce_phase() {
    // threads init
    pthread_mutex_init(&lock, NULL);

    dispatch_threads = new pthread_t[worker_num];
    task_dispatch_time = new std::chrono::steady_clock::time_point[num_reducer];

    for (int i=0; i<worker_num; i++) {
        pthread_create(&dispatch_threads[i], NULL, dispatch_reduce_thread_func, (void*)&(((Arg*)args)[i]));
    }

    for (int i=0; i<worker_num; i++) {
        pthread_join(dispatch_threads[i], NULL);
    }

    pthread_mutex_destroy(&lock);
    delete [] dispatch_threads;
    delete [] task_dispatch_time;
    delete [] ((Arg*)args);

    int exe_time = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_job_time).count();
    log_file << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    log_file << ",FinishJob," << exe_time << "\n";
    log_file.close();
}

int Scheduler::Dispatch_reducer(int target_rank) {
    int reducer_task;
    pthread_mutex_lock(&lock);
    // all dispatched
    if (reducer_to_dispatch >= num_reducer) {
        pthread_mutex_unlock(&lock);
        return -1;
    }
    
    reducer_task = reducer_to_dispatch;
    reducer_to_dispatch++;
    
    log_file << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() ;
    log_file << ",Dispatch_ReduceTask," << reducer_task+1 << "," << target_rank << "\n";
    task_dispatch_time[reducer_task] = std::chrono::steady_clock::now() ;
    pthread_mutex_unlock(&lock);
    return reducer_task;
}

void Scheduler::Reduce_task_complete(int reducer_task) {
    pthread_mutex_lock(&lock);

    int exe_time = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - task_dispatch_time[reducer_task]).count();

    log_file << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() ;
    log_file << ",Complete_ReduceTask," << reducer_task+1 << "," << exe_time << "\n";

    pthread_mutex_unlock(&lock);
}
