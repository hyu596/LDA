#ifndef _LDAMODEL_H_
#define _LDAMODEL_H_

#include <vector>
#include <map>
// #include <utility>
// #include <Model.h>
// #include <SparseDataset.h>
#include "LDADataset.h"
#include "TopicsGenerator.h"
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
          *
          * Given a group index, only allow the selected group to do sampling.
          * During sampling, changes are applied to the local statistics only
          * and are saved to change_vt of the selected group.
          *
          * Note that the global statistic is not updated yet.
          */
        void sample(int thread_idx);
        /**
          * LDA sync method
          * @param thread_idx
          *
          * Given a group index, update the global statistic based on
          * change_vt of the selected group.
          *
          * Applied updates would be removed from change_vt.
          */
        void sync(int thread_idx);
        /**
          * LDA update method
          * @param thread_idx
          *
          * Given a group index, update the statistic of the selected group
          * based on the global statistic.
          *
          */
        void update(int thread_idx);
        double loglikelihood();

      // private:

        int K_, V_, D_;
        int nworkers_;
        double alpha, eta;

        std::vector<std::string> vocabs;

        // Each entry of the following
        std::vector<std::vector<int>> ts, ds, ws;
        std::vector<std::vector<std::vector<int>>> nvts, ndts;
        std::vector<std::vector<int>> nts;
        std::vector<std::map<int, int>> l2gs;

        std::vector<std::vector<int>> global_nvt;

        // std::vector<std::map<std::pair<int, int>, int>> change_vts;
        std::vector<std::vector<std::pair<std::pair<int, int>, int> > > change_vts;

        void prepare_thread(LDADataset& dataset,
                      std::mutex& dataset_lock,
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
                      std::mutex& nvt_lock);

        void prepare_partial_docs(const std::vector<std::vector<std::pair<int, int>>>& docs,
                      std::vector<int>& t,
                      std::vector<int>& nt,
                      std::vector<int>& d,
                      std::vector<int>& w,
                      std::map<int, int>& l2g,
                      std::vector<std::vector<int>>& nvt,
                      std::vector<std::vector<int>>& ndt,
                      std::vector<std::pair<std::pair<int, int>, int> >& change_vt,
                      // std::map<std::pair<int, int>, int>& change_vt,
                      TopicsGenerator& generator,
                      std::mutex& generator_lock,
                      std::mutex& nvt_lock
                    );

        void sample_thread(std::vector<int>& t,
                        std::vector<int>& d,
                        std::vector<int>& w,
                        std::map<int, int>& l2g,
                        std::vector<int>& nt,
                        std::vector<std::vector<int>>& nvt,
                        std::vector<std::vector<int>>& ndt,
                        std::vector<std::pair<std::pair<int, int>, int> >& change_vt
                        // std::map<std::pair<int, int>, int>& change_vt
                      );
    };
}

#include "LDAModel.cpp"
#endif
