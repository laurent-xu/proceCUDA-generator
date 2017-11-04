#include <iostream>
#include <list>
#include <unordered_map>
#include <assert.h>

using namespace std;

template <class KEY_T, class VAL_T>
class LRUCache
{
    private:
        list<pair<KEY_T, VAL_T>> item_list;
        unordered_map<KEY_T, decltype(item_list.begin())> item_map;
        size_t cache_size;
    private:
        void clean(void)
        {
            while(item_map.size() > cache_size)
            {
                auto last_it = item_list.end();
                --last_it;
                item_map.erase(last_it->first);
                item_list.pop_back();
            }
        };
    public:
        LRUCache(size_t cache_size_) : cache_size(cache_size_) {};
		size_t size() {return item_map.size();};
		size_t max_size() {return cache_size;};
        bool exist(const KEY_T &key) {return item_map.count(key) > 0;};
        void put(const KEY_T &key, const VAL_T &val);
        VAL_T get(const KEY_T &key);
};

#include "lru.cc"
