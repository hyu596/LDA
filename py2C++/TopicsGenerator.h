#ifndef _TOPICS_GENERATOR_H_
#define _TOPICS_GENERATOR_H_

namespace cirrus{
    class TopicsGenerator{
      public:
        TopicsGenerator(const int max);
        int get_topic();

      private:
        int max_, cur_;
    };
}

#include "TopicsGenerator.cpp"
#endif
