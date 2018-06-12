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
void sample_one_thread(int idx){
  model->sample(idx);
}

void model_sample(int idx){
  model->sample(idx);
}

void model_sync(int idx){
  model->sync(idx);
}

void model_update(int idx){
  model->update(idx);
}

int main(){

  InputReader input;
  LDADataset dataset = input.read_lda_input(
    "nyt_data.txt",
    "vocabs_nytimes.txt",
    ","); //,

  int K = 20;
  int nworkers = 10;

  auto start = std::chrono::system_clock::now();
  model.reset(new LDAModel(dataset, K, nworkers));
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> diff = end-start;
  std::cout << "Init takes: " << diff.count() << std::endl;

  std::cout <<  model->loglikelihood() << " " << model->nworkers_ << " end"  << std::endl;

  start = std::chrono::system_clock::now();
  int p = 1, cur = 0;

  for(int j=0; j<1000; j++){

    // auto start_iter = std::chrono::system_clock::now();

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

    // end = std::chrono::system_clock::now();
    // diff = end-start_iter;
    // std::cout << j << " : sampling " << diff.count() << " ";

    // start_iter = std::chrono::system_clock::now();

    threads.clear();
    for(int i = 0; i < model->nworkers_; i++){
      // std::cout << "aa" << std::endl;
      threads.push_back(std::make_shared<std::thread>(
        model_sync, i
      ));
    }
    for(auto& t : threads){
      t->join();
    }

    // end = std::chrono::system_clock::now();
    // diff = end-start_iter;
    // std::cout << " syncing " << diff.count() << " ";

    // start_iter = std::chrono::system_clock::now();
    threads.clear();
    for(int i = 0; i < model->nworkers_; i++){
      threads.push_back(std::make_shared<std::thread>(
        model_update, i
      ));
    }
    for(auto& t : threads){
      t->join();
    }

    end = std::chrono::system_clock::now();
    // diff = end-start_iter;
    // std::cout << " updating " << diff.count() << std::endl;

    diff = end - start;
    std::cout << j << " : " << model->loglikelihood() << " " << diff.count() <<std::endl;
  }

  std::cout << "Finished.\n";


}
