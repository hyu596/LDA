#define DEBUG

// #include <TopicsGenerator.h>

namespace cirrus{
    TopicsGenerator::TopicsGenerator(const int max){
      max_ = max;
      cur_ = 0;
    }

    int TopicsGenerator::get_topic(){
      if(++cur_ > max_)
          cur_ = 1;
      return cur_;
    }
}
