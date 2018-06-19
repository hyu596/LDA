#define DEBUG

#include <algorithm>
#include <chrono>
#include <random>
#include <map>
#include <ctime>
#include <math.h>
// #include <boost>

#include "gamma.h"
// #include <LDAModel.h>

namespace cirrus{

  LDAModel::LDAModel(LDADataset dataset, int K, int nworkers){

    // Setting default values
    K_ = K;
    alpha = .1, eta = .01;
    nworkers_ = nworkers;

    // Store vocabularies list and separate into slices
    vocabs = dataset.vocabs_;
    V_ = vocabs.size();
    nslices = 5; // set to constant 5 for now; should be determined by the data size
    slices.resize(nslices);
    int slice_size = ceil(vocabs.size() / float(nslices));
    int j=0, end;
    for(int i=0; i<vocabs.size(); i+=slice_size){
      if(i + slice_size < vocabs.size()){
        end = i + slice_size;
        // slices[j] = std::vector<int>(boost::counting_iterator<int>(i),
        //                                 boost::counting_iterator<int>(i+slice
        //                                 ));
      }
      else{
        end = vocabs.size();
        // slices[j] = std::vector<int>(boost::counting_iterator<int>(i),
        //                              boost::counting_iterator<int>(vocabs.size()-1));
      }

      for(int t = i; t < end; t++){
        // j = rand() % nslices;
        slices[j].push_back(t);
      }

      ++ j;
    }

    // initialize global variables
    global_nvt.resize(nslices);
    for(int i=0; i<nslices; ++i){
      global_nvt[i].resize(slices[i].size());
      for(int j=0; j<slices[i].size(); ++j){
        global_nvt[i][j].resize(K_);
      }
    }
    global_nt.resize(K_);

    // initialize local variables
    ts.resize(nworkers_);
    ds.resize(nworkers_);
    ws.resize(nworkers_);
    ndts.resize(nworkers_);
    nts.resize(nworkers_);
    for(int i=0; i<nworkers_; ++i){
      nts[i].resize(K_);
    }

    TopicsGenerator generator(K_);
    std::mutex dataset_lock, generator_lock;
    std::vector<std::shared_ptr<std::thread>> threads;
    for(int i = 0; i < nworkers_; ++i){
      threads.push_back(
        std::make_shared<std::thread>(
          std::bind(&LDAModel::prepare_thread, this,
          std::placeholders::_1, std::placeholders::_2,
          std::placeholders::_3, std::placeholders::_4,
          std::placeholders::_5, std::placeholders::_6,
          std::placeholders::_7),
          std::ref(dataset), std::ref(dataset_lock),
          std::ref(ts[i]), std::ref(ds[i]), std::ref(ws[i]),
          std::ref(ndts[i]), std::ref(nts[i]))
      );
    }
    for(auto& th : threads){
      th->join();
    }

    // int i=0;
    // prepare_thread(std::ref(dataset), std::ref(dataset_lock),
    //                 std::ref(ts[i]), std::ref(ds[i]), std::ref(ws[i]),
    //                 std::ref(ndts[i]));

  }

  void LDAModel::prepare_thread(LDADataset& dataset,
                std::mutex& dataset_lock,
                std::vector<int>& t,
                std::vector<int>& d,
                std::vector<int>& w,
                std::vector<std::vector<int>>& ndt,
                std::vector<int>& nt){

    std::vector<std::vector<std::pair<int, int>>> docs;
    while(1){

      std::cout << dataset.num_docs() << std::endl;

      if(dataset.num_docs() == 0)
        break;

      dataset_lock.lock();
      dataset.get_some_docs(docs);
      dataset_lock.unlock();

      for(int i=0; i<nslices; ++i){

        // TODO: switch to vector might be faster
        int* change_nvt = new int[slices[i].size() * K_];
        int* change_nt = new int[K_];

        for(int j=0; j<slices[i].size() * K_; j++){
          change_nvt[j] = 0;
        }
        for(int j=0; j<K_; j++){
          change_nt[j] = 0;
        }

        prepare_partial_docs(docs, std::ref(t), std::ref(d),
                  std::ref(w), std::ref(ndt), std::ref(nt), change_nvt, change_nt,
                  slices[i][0], slices[i][slices[i].size()-1]);
        update_slice(i, change_nvt);
        update_nt(change_nt);
        // delete[] change_nvt;
        // delete[] change_nt;
      }
    }
  }

  void LDAModel::prepare_partial_docs(const std::vector<std::vector<std::pair<int, int>>>& docs,
                  std::vector<int>& t,
                  std::vector<int>& d,
                  std::vector<int>& w,
                  std::vector<std::vector<int>>& ndt,
                  std::vector<int>& nt,
                  int* change_nvt,
                  int* change_nt,
                  int lower, int upper){

    // TopicsGenerator generator(K_);

    std::vector<int>::iterator it;

    for(const auto& doc: docs){

      std::vector<int> ndt_row(K_);

      for(const auto& feat: doc){

        int gindex = feat.first, count = feat.second;

        // inclusive
        if(gindex >= lower && gindex <= upper){
          int nvt_idx_initial = (gindex - lower) * K_;
          // int nvt_idx_initial = it - slice.begin();
          // nvt_idx_initial *= K_;

          for(int i=0; i<count; ++i){

            // draw topic
            // generator_lock.lock();
            // int top = generator.get_topic();
            // generator_lock.unlock();

            int top = rand()%K_ + 1;

            // fill lcoal variables
            t.push_back(top-1);
            d.push_back(ndt.size());
            w.push_back(gindex);
            ++ ndt_row[top-1];
            ++ nt[top-1];

            // fill the updates
            change_nvt[nvt_idx_initial + top - 1] += 1;
            change_nt[top - 1] += 1;

          }
        }
      }
      ndt.push_back(ndt_row);
    }
  }

  // server side
  void LDAModel::update_slice(int slice_idx, int* change_nvt){
    int V = slices[slice_idx].size();
    for(int i=0; i<V; ++i){
      for(int j=0; j<K_; ++j){
        global_nvt[slice_idx][i][j] += change_nvt[i*K_ + j];
        // std::cout << global_nvt[slice_idx][i][j] << " ";
      }
      // std::cout << std::endl;
    }
    // std::cout << "-----------\n";
    delete[] change_nvt;
  }
  void LDAModel::update_nt(int* change_nt){
    for(int i=0; i<K_; ++i){
      global_nt[i] += change_nt[i];
    }
    delete[] change_nt;
  }
  //
  std::pair<int*, int*> LDAModel::sample(int thread_idx, int slice_idx){
    if(thread_idx < 0 || thread_idx >= nworkers_){
      throw std::runtime_error("Invalid worker idx");
    }
    else if(slice_idx < 0 || slice_idx >= nslices){
      throw std::runtime_error("Invalid slice idx");
    }
    // void sample_thread(std::vector<int>& t,
    //                 const std::vector<int>& d,
    //                 const std::vector<int>& w,
    //                 const std::vector<int>& nt,
    //                 const std::vector<std::vector<int>>& nvt,
    //                 std::vector<std::vector<int>>& ndt,
    //                 const int vocab_idx_lower,
    //                 const int vocab_idx_upper
    //               );

    auto nvt = global_nvt[slice_idx];

    return sample_thread(std::ref(ts[thread_idx]), std::ref(ds[thread_idx]),
                      std::ref(ws[thread_idx]), std::ref(nts[thread_idx]), std::ref(nvt),
                      std::ref(ndts[thread_idx]), slices[slice_idx][0], slices[slice_idx][slices[slice_idx].size()-1]);
  }

  std::pair<int*, int*> LDAModel::sample_thread(std::vector<int>& t,
                              std::vector<int>& d,
                              std::vector<int>& w,
                              std::vector<int>& nt,
                              std::vector<std::vector<int>>& nvt,
                              std::vector<std::vector<int>>& ndt,
                              int lower, int upper
                            ){

    double* rate = new double[K_];
    double r, rate_cum, linear;
    int top, new_top, doc, gindex, j;//, lindex, j;
    std::vector<int>::iterator it;

    std::vector<int>::iterator it_ndt;
    std::vector<int>::iterator it_nvt;
    std::vector<int>::iterator it_nt;

    int* change_nvt = new int[nvt.size() * K_];
    int* change_nt = new int[K_];
    for(int j=0; j<nvt.size() * K_; j++){
      change_nvt[j] = 0;
    }
    for(int j=0; j<K_; j++){
      change_nt[j] = 0;
    }

    std::vector<std::vector<int> > change_ndt(ndt.size());
    for(int i=0; i<ndt.size(); i++){
      change_ndt[i].resize(K_);
    }
    for(int i=0; i<t.size(); i++){

      // std::cout << i << "/" << t.size() << std::endl;
      top = t[i], doc = d[i], gindex = w[i];
      // lindex = l2g[gindex];

      // it = std::find(slice.begin(), slice.end(), gindex);
      if(gindex < lower || gindex > upper)
        continue;

      global_nvt[0][gindex - lower][top] -= 1;
      global_nt[top] -= 1;

      nvt[gindex - lower][top] -= 1;
      ndt[doc][top] -= 1;
      nt[top] -= 1;

      // it_ndt = ndt[doc].begin();
      // it_nvt = nvt[lindex].begin();
      // it_nt = nt.begin();

      rate_cum = 0.0;
      // rate.clear();
      // for(j=0, it_ndt = ndt[doc].begin(), it_nvt = nvt[gindex - vocab_idx_lower].begin(), it_nt = nt.begin(); j<K_; ++j, ++it_ndt, ++it_nvt, ++it_nt){
      for(int j=0; j<K_; ++j){
        // r = (alpha + *it_ndt) * (eta + *it_nvt) / (V_ * eta + *it_nt);
        r = (alpha + ndt[doc][j]) * (eta + nvt[gindex - lower][j]) / (V_ * eta + nt[j]);
        // if(r>.8){
        //   std::cout << r << std::endl;
        //   std::cout << ndt[doc][j] << " " << nvt[gindex - vocab_idx_lower][j] << " " << nt[j] << std::endl;
        // }
        if(r>0)
          rate_cum += r;

        rate[j] = rate_cum;
      }
      // std::cout << std::endl;
      // double tttt = rand() / RAND_MAX;
      // std::cout << tttt << std::endl;
      linear = rand()*rate_cum/RAND_MAX;
      // std::cout << linear / rate_cum << std::endl;
      new_top = (std::lower_bound(rate, rate+K_, linear)) - rate;
      // std::cout << new_top << std::endl;

      t[i] = new_top;
      nvt[gindex - lower][new_top] += 1;
      ndt[doc][new_top] += 1;
      nt[new_top] += 1;

      global_nvt[0][gindex - lower][new_top] += 1;
      global_nt[new_top] += 1;

      // change_nvt[(gindex - lower) * K_ + top] -= 1;
      // change_nvt[(gindex - lower) * K_ + new_top] += 1;
      // change_nt[top] -= 1;
      // change_nt[new_top] += 1;

      // change_ndt[doc][top] -= 1;
      // change_ndt[doc][new_top] += 1;

      // key_dec = std::make_pair(gindex, top);
      // key_inc = std::make_pair(gindex, new_top);
      //
      // change_vt.push_back(std::make_pair(key_dec, -1));
      // change_vt.push_back(std::make_pair(key_inc, 1));
      //
      // change_nt[top] -= 1;
      // change_nt[new_top] += 1;

    }

    // for(int i=0; i<ndt.size(); i++){
    //   for(int j=0; j<K_; ++j){
    //     ndt[i][j] += change_ndt[i][j];
    //   }
    // }

    // std::cout << "change_vt size: " << change_vt.size() << std::endl;
    // std::cout << "change_nt size: " << change_nt.size() << std::endl;
    delete[] rate;
    return std::make_pair(change_nvt, change_nt);
  }


  double LDAModel::loglikelihood(){

    double lgamma_eta = lda_lgamma(eta), lgamma_alpha = lda_lgamma(alpha);
    double ll = 0;
    ll += K_ * lda_lgamma(eta * V_);

    for(int i=0; i<nworkers_; i++){
      for(int j=0; j<ndts[i].size(); j++){
        int ndj = 0;
        for(int k=0; k<K_; k++){
          ndj += ndts[i][j][k];
          if(ndts[i][j][k] > 0)
            ll += lda_lgamma(alpha + ndts[i][j][k]) - lgamma_alpha;
          // else
          //   std::cout << "Nope";
        }
        ll += lda_lgamma(alpha * K_) - lda_lgamma(alpha * K_ + ndj);
      }
    }

    for(int i=0; i<K_; i++){
      ll -= lda_lgamma(eta * V_ + global_nt[i]);
      for(int j=0; j<nslices; ++j){
        for(int v=0; v<global_nvt[j].size(); ++v){
          if(global_nvt[j][v][i] > 0)
            ll += lda_lgamma(eta + global_nvt[j][v][i]) - lgamma_eta;
          // else
          //   std::cout << "Nope";
        }
      }
    }

    return ll;
  }

  void LDAModel::most_frequent_words_all_topics(){
    std::vector<double> beta(V_);
    for(int i=0; i<K_; ++i){
      int idx = 0;
      for(int j=0; j<nslices; ++j){
        for(int t=0; t<global_nvt[j].size(); ++t){
          beta[idx] = (eta + global_nvt[j][t][i]) / (V_ * eta + global_nt[i]);
          idx ++;
        }
      }
      for(int j=0; j<10; ++j){
        idx = std::distance(beta.begin(), std::max_element(beta.begin(), beta.end()));
        std::cout << vocabs[idx] << " ";
        beta[idx] = 0;
      }
      std::cout << std::endl;
    }
  }

}
