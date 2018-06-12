#define DEBUG

#include <algorithm>
#include <chrono>
#include <random>
#include <map>
#include <ctime>

#include "gamma.h"
// #include <LDAModel.h>

namespace cirrus{

  LDAModel::LDAModel(LDADataset dataset, int K, int nworkers){

    K_ = K;
    alpha = .1, eta = .01;
    nworkers_ = nworkers;
    vocabs = dataset.vocabs_;
    V_ = vocabs.size();

    global_nvt.resize(V_);
    for(int i=0; i<V_; i++){
      global_nvt[i].resize(K_);
    }

    ts.resize(nworkers_);
    ds.resize(nworkers_);
    ws.resize(nworkers_);
    nts.resize(nworkers_);
    l2gs.resize(nworkers_);
    nvts.resize(nworkers_);
    ndts.resize(nworkers_);
    change_vts.resize(nworkers_);

    change_nts.resize(nworkers_);
    for(int i=0; i<nworkers_; i++){
      change_nts[i].resize(K_);
    }
    global_nt.resize(K_);

    TopicsGenerator generator(K_);
    std::mutex dataset_lock, generator_lock, nvt_lock, nt_lock;
    std::vector<std::shared_ptr<std::thread>> threads;
    for(int i = 0; i < nworkers_; ++i){
      nts[i].resize(K_);
      threads.push_back(
        std::make_shared<std::thread>(
          std::bind(&LDAModel::prepare_thread, this,
          std::placeholders::_1, std::placeholders::_2,
          std::placeholders::_3, std::placeholders::_4,
          std::placeholders::_5, std::placeholders::_6,
          std::placeholders::_7, std::placeholders::_8,
          std::placeholders::_9, std::placeholders::_10,
          std::placeholders::_11, std::placeholders::_12,
          std::placeholders::_13, std::placeholders::_14),
          std::ref(dataset), std::ref(dataset_lock),
          std::ref(ts[i]), std::ref(ds[i]), std::ref(ws[i]),
          std::ref(nts[i]), std::ref(l2gs[i]),
          std::ref(nvts[i]), std::ref(ndts[i]), std::ref(change_vts[i]),
          std::ref(generator), std::ref(generator_lock), std::ref(nvt_lock), std::ref(nt_lock))
      );
    }

    for(auto& th : threads){
      th->join();
    }

    for(int i=0; i<nworkers_; i++){
      update(i);
    }
  }

  void LDAModel::prepare_thread(LDADataset& dataset,
                std::mutex& dataset_lock,
                std::vector<int>& t,
                std::vector<int>& d,
                std::vector<int>& w,
                std::vector<int>& nt,
                std::map<int, int>& l2g,
                std::vector<std::vector<int>>& nvt,
                std::vector<std::vector<int>>& ndt,
                std::vector<std::pair<std::pair<int, int>, int> >& change_vt,
                TopicsGenerator& generator,
                std::mutex& generator_lock,
                std::mutex& nvt_lock,
                std::mutex& nt_lock
              ){

    std::vector<std::vector<std::pair<int, int>>> docs;
    while(1){

      if(dataset.num_docs() == 0)
        break;

      dataset_lock.lock();
      dataset.get_some_docs(docs);
      dataset_lock.unlock();

      // std::cout << "--\n";
      prepare_partial_docs(docs, std::ref(t), std::ref(d),
                std::ref(w), std::ref(nt), std::ref(l2g),
                std::ref(nvt), std::ref(ndt), std::ref(change_vt),
                std::ref(generator), std::ref(generator_lock), std::ref(nvt_lock), std::ref(nt_lock));

    }
  }

  void LDAModel::prepare_partial_docs(const std::vector<std::vector<std::pair<int, int>>>& docs,
                std::vector<int>& t,
                std::vector<int>& d,
                std::vector<int>& w,
                std::vector<int>& nt,
                std::map<int, int>& l2g,
                std::vector<std::vector<int>>& nvt,
                std::vector<std::vector<int>>& ndt,
                std::vector<std::pair<std::pair<int, int>, int> >& change_vt,
                // std::map<std::pair<int, int>, int>& change_vt,
                TopicsGenerator& generator,
                std::mutex& generator_lock,
                std::mutex& nvt_lock,
                std::mutex& nt_lock
              ){
    int temp;
    for(const auto& doc: docs){
      std::vector<int> ndt_row(K_);
      for(const auto& feat: doc){
        int gindex = feat.first, count = feat.second;
        // find_iter = std::find(l2g.begin(), l2g.end(), gindex);
        if(l2g.count(gindex) == 0){
          temp =l2g.size();
          l2g[gindex] = temp;
          nvt.push_back(std::vector<int>(K_));
        }

        for(int i=0; i<count; i++){
          generator_lock.lock();
          int top = generator.get_topic();
          generator_lock.unlock();

          t.push_back(top-1);
          d.push_back(ndt.size());
          w.push_back(gindex);

          nt[top-1] ++;
          ndt_row[top-1] ++;
          nvt[l2g[gindex]][top-1] ++;

          nvt_lock.lock();
          global_nvt[gindex][top-1] ++;
          nvt_lock.unlock();

          nt_lock.lock();
          global_nt[top-1] ++;
          nt_lock.unlock();
        }
      }
      ndt.push_back(ndt_row);
    }
  }

  void LDAModel::sample(int thread_idx){
    if(thread_idx < 0 || thread_idx >= nworkers_){
      throw std::runtime_error("Invalid worker idx");
    }

    // change_vts[thread_idx] = std::map<std::pair<int, int>, int>();
    sample_thread(std::ref(ts[thread_idx]), std::ref(ds[thread_idx]), std::ref(ws[thread_idx]),
                  std::ref(l2gs[thread_idx]), std::ref(nts[thread_idx]), std::ref(nvts[thread_idx]),
                  std::ref(ndts[thread_idx]), std::ref(change_vts[thread_idx]), std::ref(change_nts[thread_idx]));
  }

  void LDAModel::sample_thread(std::vector<int>& t,
                  std::vector<int>& d,
                  std::vector<int>& w,
                  std::map<int, int>& l2g,
                  std::vector<int>& nt,
                  std::vector<std::vector<int>>& nvt,
                  std::vector<std::vector<int>>& ndt,
                  std::vector<std::pair<std::pair<int, int>, int> >& change_vt,
                  std::vector<int>& change_nt){
                  // std::map<std::pair<int, int>, int>& change_vt){

    double* rate = new double[K_];
    double r, rate_cum, linear;
    int top, new_top, doc, gindex, lindex;
    std::pair<int, int> key_dec, key_inc;


    std::vector<int>::iterator it_ndt;
    std::vector<int>::iterator it_nvt;
    std::vector<int>::iterator it_nt;

    for(int i=0; i<t.size(); i++){

      top = t[i], doc = d[i], gindex = w[i];
      lindex = l2g[gindex];

      nvt[lindex][top] -= 1;
      ndt[doc][top] -= 1;
      nt[top] -= 1;

      it_ndt = ndt[doc].begin();
      it_nvt = nvt[lindex].begin();
      it_nt = nt.begin();

      rate_cum = 0.0;
      // rate.clear();
      for(int j=0; j<K_; j++){
        r = (alpha + *it_ndt) * (eta + *it_nvt) / (V_ * eta + *it_nt);
        // double r = (alpha + ndt[doc][j]) * (eta + nvt[lindex][j]) / (V_ * eta + nt[j]);
        if(r>0)
          rate_cum += r;

        rate[j] = rate_cum;

        it_ndt ++;
        it_nvt ++;
        it_nt ++;
      }

      linear = rand()*rate_cum/RAND_MAX;
      new_top = (std::lower_bound(rate, rate+K_, linear)) - rate;

      t[i] = new_top;
      nvt[lindex][new_top] += 1;
      ndt[doc][new_top] += 1;
      nt[new_top] += 1;

      key_dec = std::make_pair(gindex, top);
      key_inc = std::make_pair(gindex, new_top);

      change_vt.push_back(std::make_pair(key_dec, -1));
      change_vt.push_back(std::make_pair(key_inc, 1));

      change_nt[top] -= 1;
      change_nt[new_top] += 1;

    }
    delete[] rate;
  }

  void LDAModel::sync(int thread_idx){
    while(!change_vts[thread_idx].empty()){
      auto c = change_vts[thread_idx].back();
      global_nvt[c.first.first][c.first.second] += c.second;
      change_vts[thread_idx].pop_back();
    }
    for(int i=0; i<K_; i++){
      global_nt[i] += change_nts[thread_idx][i];
      change_nts[thread_idx][i] = 0;
    }
  }

  void LDAModel::update(int thread_idx){
    std::map<int, int>::iterator it = l2gs[thread_idx].begin();
    while(it != l2gs[thread_idx].end()){
      for(int j=0; j<K_; j++){
        nvts[thread_idx][it->second][j] = global_nvt[it->first][j];
      }
      it ++;
    }
    for(int i=0; i<K_; i++){
      nts[thread_idx][i] = global_nt[i];
    }
  }

  double LDAModel::loglikelihood(){

    double lgamma_eta = lda_lgamma(eta), lgamma_alpha = lda_lgamma(alpha);
    double ll = 0;
    ll += K_ * lda_lgamma(eta * V_);

    // std::vector<int> nt_l(K_);
    for(int i=0; i<nworkers_; i++){
      for(int j=0; j<ndts[i].size(); j++){
        int ndj = 0;
        for(int k=0; k<K_; k++){
          // nt_l[k] += ndts[i][j][k];
          ndj += ndts[i][j][k];
          if(ndts[i][j][k] > 0)
            ll += lda_lgamma(alpha + ndts[i][j][k]) - lgamma_alpha;
        }
        ll += lda_lgamma(alpha * K_) - lda_lgamma(alpha * K_ + ndj);
      }
    }

    for(int i=0; i<K_; i++){
      double ni = 0.0;
      // ll -= lda_lgamma(eta * V_ + nt_l[i]);
      ll -= lda_lgamma(eta * V_ + global_nt[i]);
      for(int j=0; j<V_; j++){
        // double nij_sum = 0, times = 0;
        // for(int m=0; m<nworkers_; m++){
        //   if(l2gs[m].count(j) == 1){
        //     nij_sum += nvts[m][l2gs[m][j]][i];
        //     times += 1;
        //   }
        // }
        // nij_sum /= times;
        if(global_nvt[j][i] > 0)
          ll += lda_lgamma(eta + global_nvt[j][i]) - lgamma_eta;
      }
    }

    return ll;
  }

}
