#ifndef _INPUT_H_
#define _INPUT_H_

// #include <Dataset.h>
// #include <SparseDataset.h>
// #include <Configuration.h>
#include <string>
#include <vector>
#include <queue>
#include <mutex>
#include <map>
#include <iostream>
// #include <config.h>
// #include <MurmurHash3.h>
#include "LDADataset.h"
// #include "config.h"


namespace cirrus {

class InputReader {
  // TODO: double check if const Configuration& needs to be added to fun

  public:
    LDADataset read_lda_input(
        const std::string& input_doc_file,
        const std::string& input_vocab_file,
        std::string delimiter);//,
        // const Configuration&);

  private:
    void parse_read_lda_input_thread(std::ifstream& fin, std::mutex& fin_lock,
      std::mutex& out_lock,
      const std::string& delimiter,
      std::vector<std::vector<std::pair<int,int> > >& samples_res,
      uint64_t limit_lines, std::atomic<unsigned int>&,
      std::function<void(const std::string&, const std::string&,
        std::vector<std::pair<int, int> >&)> fun);

    void parse_read_lda_input_line(
      const std::string& line, const std::string& delimiter,
      std::vector<std::pair<int, int> >& features);//,
      // const Configuration&);

    void read_lda_vocab_input(std::ifstream& fin, std::vector<std::string>& vocabs);
};

} // namespace cirrus

#include "InputReader.cpp"

#endif  // _INPUT_H_
