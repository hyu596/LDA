// #include <LDADataset.h>

#include <iostream>

namespace cirrus{
  LDADataset::LDADataset(){

  }

  LDADataset::LDADataset(std::vector<std::vector<std::pair<int, int> > > docs,
               std::vector<std::string> vocabs){
                docs_ = docs;
                vocabs_ = vocabs;
               }

  uint64_t LDADataset::num_docs() const{
    return docs_.size();
  }

  uint64_t LDADataset::num_vocabs() const{
    return vocabs_.size();
  }

  void LDADataset::check() const{
    for(const auto& w: docs_){
      for(const auto v: w){
        if(v.first < 0 || v.second <= 0)
          throw std::runtime_error("Input error");
      }
    }
    std::cout << "Dataset has been checked.\n";
  }


  void LDADataset::get_some_docs(std::vector<std::vector<std::pair<int, int> > >& docs){
    if(docs_.size() > 100)
      docs.resize(100);
    else
      docs.resize(docs_.size());
    std::copy(docs_.begin(), docs_.size() > 100 ? docs_.begin() + 100 : docs_.end(), docs.begin());
    docs_.erase(docs_.begin(), docs_.size() > 100 ? docs_.begin() + 100 : docs_.end());
  }

}
