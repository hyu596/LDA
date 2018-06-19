#include <string>
#include <vector>
// #include <thread>
#include <cassert>
// #include <memory>
#include <algorithm>
#include <atomic>
#include <map>
#include <iomanip>
#include <chrono>

#include "LDADataset.h"
#include "InputReader.h"
#include "LDAModel.h"

using namespace cirrus;

std::unique_ptr<LDAModel> model;
void model_sample(int thread_idx){
  for(int i=0; i<model->nslices; i++){
    auto updates = model->sample(thread_idx, i);
    // model->update_slice(i, updates.first);
    // model->update_nt(updates.second);
  }
}


int main(){

  InputReader input;
  LDADataset dataset = input.read_lda_input(
    "nytimes.txt",
    "nytimes_vocab.txt",
    ","); //,

  int K = 20;
  int nworkers = 5;

  std::cout << "Finished reading documents\n";

  auto start = std::chrono::system_clock::now();
  model.reset(new LDAModel(dataset, K, nworkers));

  std::cout << "Init model\n";
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> diff = end-start;
  std::cout << "Init takes: " << diff.count() << std::endl;

  std::cout <<  model->loglikelihood() << " " << model->nworkers_ << " end"  << std::endl;

  start = std::chrono::system_clock::now();
  int p = 1, cur = 0, cur_slice = 0;
  for(int j=0; j<10000; j++){

    auto start_iter = std::chrono::system_clock::now();
    // std::cout << "Start sampling\n";
    std::vector<std::shared_ptr<std::thread>> threads;
    for(int i = 0; i < p; i++){
      threads.push_back(std::make_shared<std::thread>(
        model_sample, cur
      ));
      cur = (cur + 1) % model->nworkers_;
    }

    for(auto& t : threads){
      t->join();
    }
    // model_sample(cur);
    // cur = (cur + 1) % model->nworkers_;

    end = std::chrono::system_clock::now();
    diff = end-start_iter;
    // if(j%100 == 0)
      std::cout << j << " : sampling " << diff.count() << " " << model->loglikelihood() << std::endl;

    // for(int i=0; i<model->K_; i++){
    //   std::cout << model->global_nvt[0][4][i] << " ";
    // }
    // std::cout << std::endl;
  }
  // model->most_frequent_words_all_topics();

  std::cout << "Finished.\n";


}
