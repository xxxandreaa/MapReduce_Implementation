#include "Worker.h"
#include <iostream>
#include <fstream>
#include <mpi.h>
#include <unistd.h>
#include <algorithm>

void *worker_thread_func(void *args) {
    Worker *pool = (Worker *)args ;
    int task_chunkIdx ;

    for (; ; ) {
        pthread_mutex_lock(&(pool->work_lock)) ;

        while (pool->task[1] == 1 && !pool->done) {
            pool->waiting_threads++ ;
            pthread_cond_wait(&(pool->cond), &(pool->work_lock)) ;
        }

        if (pool->done) {
            break ;
        }

        task_chunkIdx = pool->task[0] ;
        pool->task[1] = 1 ;
        pool->waiting_threads-- ;
        pthread_mutex_unlock(&(pool->work_lock)) ;
        
        pool->Map_functions(task_chunkIdx) ;
    }

    pthread_mutex_unlock(&(pool->work_lock)) ;
    pthread_exit(NULL) ;
}

Worker::Worker(int size, int rank, std::string job_name, int num_reducer, int delay, std::string input_filename, int chunk_size, std::string output_dir)
:size(size), rank(rank), job_name(job_name), num_reducer(num_reducer), delay(delay), input_filename(input_filename), chunk_size(chunk_size), output_dir(output_dir)
{
    cpu_set_t cpuset;
    sched_getaffinity(0, sizeof(cpuset), &cpuset);
    int ncpus = CPU_COUNT(&cpuset) ;

    thread_num = ncpus-1 ;
    task = new int[2] ;
    task[1] = 1 ;
    done = false ;
    threads = new pthread_t[thread_num] ;
    waiting_threads = 0 ;
    inter_results = new std::vector<std::pair<std::string, int>>[num_reducer] ;
}

Worker::~Worker(){}

void Worker::Map_phase() {
    pthread_mutex_init(&work_lock, NULL) ;
    pthread_mutex_init(&write_lock, NULL) ;
    pthread_cond_init(&cond, NULL) ;

    for (int i=0; i<thread_num; i++) {
        pthread_create(&threads[i], NULL, worker_thread_func, (void*)this);
    }

    while (true) {
        pthread_mutex_lock(&work_lock) ;
        if (task[1] == 1 && waiting_threads > 0) {
            int task_chunkIdx ;
            // send request
            MPI_Send(&rank, 1, MPI_INT, 0, 0, MPI_COMM_WORLD) ;
            // recv
            MPI_Recv(&task_chunkIdx, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE) ;

            // if done
            if (task_chunkIdx == 0) {
                done = true ;
                MPI_Send(&task_chunkIdx, 1, MPI_INT, 0, 1, MPI_COMM_WORLD) ;

                pthread_cond_broadcast(&cond) ;
                pthread_mutex_unlock(&work_lock) ;
                break ;
            }
            else {
                task[0] = task_chunkIdx ;
                task[1] = 0 ;
                pthread_cond_signal(&cond) ;
            }
        }
        pthread_mutex_unlock(&work_lock) ;
    }

    for (int i=0; i<thread_num; i++) {
        pthread_join(threads[i], NULL) ;
    }

    int total_words_num = 0 ;
    int dummy ;
    for (int i=0; i<num_reducer; i++) {
        total_words_num += inter_results[i].size() ;
    }
    MPI_Reduce(&total_words_num, &dummy, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD) ;
    Shuffle() ;
    MPI_Barrier(MPI_COMM_WORLD) ;

    pthread_mutex_destroy(&work_lock) ;
    pthread_mutex_destroy(&write_lock) ;
    pthread_cond_destroy(&cond) ;
    delete [] task ;
    delete [] threads ;
    delete [] inter_results ;
}

void Worker::Map_functions(int task_chunkIdx) {
    if (task_chunkIdx < 0) {
        task_chunkIdx = -task_chunkIdx;
        sleep(delay);
    }
    std::vector<std::pair<int, std::string>> record = Input_split(task_chunkIdx);
    std::vector<std::pair<std::string, int>> out_record = Map(record);
    Partition(out_record);
    MPI_Send(&task_chunkIdx, 1, MPI_INT, 0, 1, MPI_COMM_WORLD) ;
}

// map phase functions
std::vector<std::pair<int, std::string>> Worker::Input_split(int task_chunkIdx) {
    std::ifstream input_file ;
    input_file.open(input_filename) ;

    std::string line ;
    for (int i=0; i<(task_chunkIdx-1)*chunk_size; i++) {
        getline(input_file, line) ;
    }

    std::vector<std::pair<int, std::string>> record ;
    for (int i=0; i<chunk_size; i++) {
        getline(input_file, line) ;

        record.emplace_back((task_chunkIdx-1)*chunk_size+i+1, line) ;
    }
    input_file.close() ;

    return record ;
}

std::vector<std::pair<std::string, int>> Worker::Map(std::vector<std::pair<int, std::string>> record) {
    std::vector<std::pair<std::string, int>> out_record ;
    size_t pos ;
    std::string line, word ;

    for (auto r : record) {
        line = r.second ;
        while ((pos = line.find(" ")) != std::string::npos) {
            word = line.substr(0, pos);
            out_record.emplace_back(word, 1);

            line.erase(0, pos + 1);
        }
        if (!line.empty()) {
            out_record.emplace_back(line, 1);
        }
    }

    return out_record ;
}

void Worker::Partition(std::vector<std::pair<std::string, int>> out_record) {
    pthread_mutex_lock(&write_lock) ;
    for (auto r : out_record) {
        int hash = (r.first[0] - 'A') % num_reducer ;
        inter_results[hash].emplace_back(r) ;
    }

    pthread_mutex_unlock(&write_lock) ;
}

void Worker::Shuffle() {
    std::ofstream inter_file ;
    for (int i=0; i<num_reducer; i++) {
        inter_file.open("./inter/" + std::to_string(rank) + "-" + std::to_string(i) + ".txt") ;
        for (auto r : inter_results[i]) {
            inter_file << r.first << " " << r.second << std::endl ;
        }
        inter_file.close() ;
    }
}

void Worker::Reduce_phase() {
    int reducer_task ;
    while (true) {
        // send request
        MPI_Send(&rank, 1, MPI_INT, 0, 2, MPI_COMM_WORLD) ;
        // recv
        MPI_Recv(&reducer_task, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE) ;

        // if done
        if (reducer_task < 0) {
            break ;
        }
        else {
            Reduce_functions(reducer_task) ;
            MPI_Send(&reducer_task, 1, MPI_INT, 0, 3, MPI_COMM_WORLD);
        }
    }
}

void Worker::Reduce_functions(int reducer_task) {
    std::vector<std::pair<std::string, int>> records = Sort(reducer_task) ;
    std::map<std::string, std::vector<int>> group_records = Group(records) ;
    std::vector<std::pair<std::string, int>> results = Reduce(group_records) ;
    Output(results, reducer_task) ;
}

//CHANGE SORT FUNCTION HERE
bool compare_func(std::pair<std::string, int> a, std::pair<std::string, int> b) {
    return a.first < b.first ;
}

std::vector<std::pair<std::string, int>> Worker::Sort(int reducer_task) {
    std::ifstream inter_file ;
    std::string word ;
    int count ;
    std::vector<std::pair<std::string, int>> records;

    for (int i=1; i<=size; i++) {
        inter_file.open("./inter/" + std::to_string(i) + "-" + std::to_string(reducer_task) + ".txt") ;
        while (inter_file >> word >> count) {
            records.emplace_back(word, count) ;
        }
        inter_file.close() ;
    }
    std::sort(records.begin(), records.end(), compare_func) ;

    return records ;
}

//CHANGE GROUP FUNCTION HERE
std::map<std::string, std::vector<int>> Worker::Group(std::vector<std::pair<std::string, int>> records) {
    std::map<std::string, std::vector<int>> group_records ;

    for (auto r : records) {
        std::string key = r.first ;
        //std::string key = r.first.substr(0,1);
        if (group_records.count(key) == 0){
            std::vector<int> temp ;
            temp.emplace_back(r.second) ;
            group_records.emplace(key, temp) ;
        }
        else {
            group_records[key].emplace_back(r.second) ;
        }
    }

    return group_records ;
}

std::vector<std::pair<std::string, int>> Worker::Reduce(std::map<std::string, std::vector<int>> group_records) {
    std::vector<std::pair<std::string, int>> results;
    for (auto r : group_records) {
        int counts = 0 ;
        for (auto c : r.second) {
            counts += c ;
        }
        results.emplace_back(r.first, counts) ;
    }
    return results ;
}

//CHANGE OUTPUT FUNCTION HERE
void Worker::Output(std::vector<std::pair<std::string, int>> results, int reducer_task) {
    std::ofstream output_file ;
    output_file.open(output_dir + job_name + "-" + std::to_string(reducer_task+1) + ".out") ;
    for (auto r : results) {
        output_file << r.first << " " << r.second << "\n" ;
    }
    output_file.close() ;
}
