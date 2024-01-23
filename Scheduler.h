#include <string>
#include <vector>
#include <pthread.h>
#include <fstream>
#include <chrono>

class Scheduler {
private:
    int size, num_reducer;
    std::string job_name, locality_config_filename, output_dir, log_filename;
    std::ofstream log_file;
    std::vector<std::pair<int, int>> locality;
    pthread_t *dispatch_threads;
    int *dispatch_task_num;
    std::chrono::steady_clock::time_point *task_dispatch_time;
    pthread_t *check_threads;
    int worker_num;
    void *args;
    int reducer_to_dispatch;
    std::chrono::steady_clock::time_point start_job_time;
    
public:
    pthread_mutex_t lock;
    Scheduler(int size, std::string job_name, int num_reducer, int delay, std::string input_filename, int chunk_size, std::string locality_config_filename, std::string output_dir);
    ~Scheduler();
    void Map_phase();
    int Dispatch_mapper(int target_rank);
    int Map_task_complete(int rank, int task_chunkIdx);
    void Reduce_phase();
    int Dispatch_reducer(int target_rank);
    void Reduce_task_complete(int reducer_task);
};
