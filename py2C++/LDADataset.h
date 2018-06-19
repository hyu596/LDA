#ifndef _LDADATASET_H_
#define _LDADATASET_H_

#include <vector>
#include <string>

namespace cirrus{
  /**
    * This class is used to hold a LDA dataset
    */
  class LDADataset{
  public:
    LDADataset();
    LDADataset(std::vector<std::vector<std::pair<int, int> > > docs,
               std::vector<std::string> vocabs);
    //TODO
    void get_some_docs(std::vector<std::vector<std::pair<int, int> > > & docs);
    int empty();

    uint64_t num_docs() const;
    uint64_t num_vocabs() const;
    void check() const;

    std::vector<std::vector<std::pair<int, int> > > docs_;
    std::vector<std::string> vocabs_;
    int sample_size;
  };
}

#include "LDADataset.cpp"

#endif
