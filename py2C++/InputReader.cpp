#include "InputReader.h"

#include <string.h>
#include <vector>
#include <thread>
#include <cassert>
#include <memory>
#include <algorithm>
#include <atomic>
#include <map>
#include <iomanip>
#include <fstream>

#define DEBUG

namespace cirrus{

static const int STR_SIZE = 100000;        // max size for dataset line


LDADataset  InputReader::read_lda_input(
  const std::string& input_doc_file,
  const std::string& input_vocab_file,
  std::string delimiter){//,
  // const Configuration& config){

std::cout << "Reading input file: " << input_doc_file << std::endl;
std::cout << "Reading Vocab file: " << input_vocab_file << std::endl;
// std::cout << "Limit_line: " << config.get_limit_samples() << std::endl;

std::ifstream fin(input_doc_file, std::ifstream::in);
std::ifstream fin_vocab(input_vocab_file, std::ifstream::in);
if (!fin || !fin_vocab) {
  throw std::runtime_error("Error opening input file");
}
std::mutex fin_lock, out_lock; // one for reading fin, one for writing to samples
std::atomic<unsigned int> lines_count(0);

std::vector<std::vector<std::pair<int, int>>> samples;  // final result

uint64_t nthreads = 10;
std::vector<std::string> vocabs;
std::vector<std::shared_ptr<std::thread>> threads;

for (uint64_t i = 0; i < nthreads - 1; ++i) {
  threads.push_back(
      std::make_shared<std::thread>(
        std::bind(&InputReader::parse_read_lda_input_thread, this,
          std::placeholders::_1, std::placeholders::_2,
          std::placeholders::_3, std::placeholders::_4,
          std::placeholders::_5, std::placeholders::_6,
          std::placeholders::_7, std::placeholders::_8),
        std::ref(fin), std::ref(fin_lock), std::ref(out_lock),
        std::ref(delimiter), std::ref(samples),
        300000, std::ref(lines_count), // config.get_limit_samples(), std::ref(lines_count),
        std::bind(&InputReader::parse_read_lda_input_line, this,
          std::placeholders::_1,
          std::placeholders::_2,
          std::placeholders::_3)//,
          // config)
        ));
}
threads.push_back(
  std::make_shared<std::thread>(
    std::bind(&InputReader::read_lda_vocab_input, this,
            std::placeholders::_1, std::placeholders::_2),
    std::ref(fin_vocab), std::ref(vocabs)
  )
);

for (auto& t : threads) {
  t->join();
}

// parse_read_lda_input_thread(std::ref(fin), std::ref(fin_lock), std::ref(out_lock),
//         std::ref(delimiter), std::ref(samples),
//         10000, std::ref(lines_count), // config.get_limit_samples(), std::ref(lines_count),
//         std::bind(&InputReader::parse_read_lda_input_line, this,
//           std::placeholders::_1,
//           std::placeholders::_2,
//           std::placeholders::_3));
// read_lda_vocab_input(std::ref(fin_vocab), std::ref(vocabs));

// process each line
std::cout << "Read a total of " << samples.size() << " samples" << std::endl;
return LDADataset(std::move(samples), vocabs);
}

void InputReader::parse_read_lda_input_thread(std::ifstream& fin,
  std::mutex& fin_lock,
  std::mutex& out_lock,
  const std::string& delimiter,
  std::vector<std::vector<std::pair<int,int>>>& samples_res,
  uint64_t limit_lines, std::atomic<unsigned int>& lines_count,
  std::function<void(const std::string&, const std::string&,
    std::vector<std::pair<int, int>>&)> fun){

  std::vector<std::vector<std::pair<int, int>>> samples;
  std::string line;
  uint64_t lines_count_thread = 0;

  while(1){

      fin_lock.lock();
      getline(fin, line);
      fin_lock.unlock();

      if(fin.eof()){
        // std::cout << "aa\n";
        break;
      }

      if (lines_count && lines_count >= limit_lines){
        // std::cout << "bb\n";
        break;
      }

      std::vector<std::pair<int, int>> words;
      fun(line, delimiter, words);
      samples.push_back(words);

      // if (lines_count % 100000 == 0) {
      //   std::cout << "Read: " << lines_count << "/" << lines_count_thread << " lines." << std::endl;
      // }
      ++lines_count;
      lines_count_thread++;
  }
  out_lock.lock();
  for(const auto& s: samples){
    samples_res.push_back(s);
  }
  out_lock.unlock();
}

void InputReader::parse_read_lda_input_line(
  const std::string& line, const std::string& delimiter,
  std::vector<std::pair<int, int>>& output_features){//,
  // const Configuration& config){

  if (line.size() > STR_SIZE){
    throw std::runtime_error(
          "Criteo input line is too big: " + std::to_string(line.size()) + " " + std::to_string(STR_SIZE)) ;
  }

  char str[STR_SIZE];
  strncpy(str, line.c_str(), STR_SIZE - 1);
  char* s = str;

  uint64_t col = 0;
  std::string delim2 = ":";

  // on each line,
  // the assumed format is vocab1:count1, vocab2:count2 ...
  while(char* l = strsep(&s, delimiter.c_str())){
    int d = atoi(strsep(&l, delim2.c_str())); // d is vocab1
    int c = atoi(l); // c is count
    output_features.push_back(std::make_pair(d, c));
  }

}

void InputReader::read_lda_vocab_input(std::ifstream& fin,
          std::vector<std::string>& vocabs){
    std::string line;
    int i = 0;
    while(1){
      getline(fin, line);
      vocabs.push_back(line);

      if(fin.eof()){
        break;
      }
      i ++;
    }
}

}
