#ifndef _LDAMODEL_H_
#define _LDAMODEL_H_

#include <vector>
#include <map>
// #include <utility>
// #include <Model.h>
// #include <SparseDataset.h>
#include "LDADataset.h"
// #include "TopicsGenerator.h"
//#include <ModelGradient.h>
// #include <Configuration.h>
// #include <map>

namespace cirrus{

    /**
      * Latent Dirichlet Allocation model
      */
    class LDAModel{
      public:
        /**
          * LDA Model constructor from dataset
          * @param LDA Dataset
          * @param K Number of latent topics; hyper-parameter
          * @param nworkers Number of groups we separate whole dataset into
          */
        LDAModel(LDADataset input, int K, int nworkers);
        /**
          * LDA sampling method
          * @param thread_idx
          * @param slice_idx
          *
          */
        std::pair<int*, int*> sample(int thread_idx, int slice_idx);

        /**
          * LDA loglikelihood method
          *
          * Compute loglikelihood based on
          * 1) global statistic
          * 2) local ndts of all the groups
          *
          */
        double loglikelihood();

        void most_frequent_words_all_topics();

      // private:

      /**
        * K_: number of topics
        * V_: number of words
        * D_: number of documents
        */
        int K_, V_, D_;

        // nslices should be determined by the data size but not just use constant 10
        int nworkers_, nslices;
        double alpha, eta;

        std::vector<std::string> vocabs;
        std::vector<std::vector<int> > slices;

        std::vector<std::vector<int>> ts, ds, ws;
        std::vector<std::vector<std::vector<int>>> ndts;
        std::vector<std::vector<int> > nts;

        std::vector<std::vector<std::pair<std::pair<int, int>, int> > > change_vts;
        std::vector<std::vector<int>> change_nts;

        /**
          *
          * the global word-topic-count statistic
          *
          */
        std::vector<std::vector<std::vector<int> > > global_nvt;
        std::vector<int> global_nt;

        void prepare_thread(LDADataset& dataset,
                      std::mutex& dataset_lock,
                      std::vector<int>& t,
                      std::vector<int>& d,
                      std::vector<int>& w,
                      std::vector<std::vector<int>>& ndt,
                      std::vector<int>& nt);

        void prepare_partial_docs(const std::vector<std::vector<std::pair<int, int>>>& docs,
                      std::vector<int>& t,
                      std::vector<int>& d,
                      std::vector<int>& w,
                      std::vector<std::vector<int>>& ndt,
                      std::vector<int>& nt,
                      int* change_nvt,
                      int* change_nt,
                      int lower, int upper
                    );

        /**
          * update the slice_idx^th global_nvt based on change_nvt
          */
        void update_slice(int slice_idx, int* change_nvt);
        /**
          * update global_nt based on change_nt
          */
        void update_nt(int* change_nt);

        std::pair<int*, int*> sample_thread(std::vector<int>& t,
                                    std::vector<int>& d,
                                    std::vector<int>& w,
                                    std::vector<int>& nt,
                                    std::vector<std::vector<int>>& nvt,
                                    std::vector<std::vector<int>>& ndt,
                                    int lower, int upper
                                    );


    };
}

#include "LDAModel.cpp"
#endif
