#include <iostream>
#include <list>
#include <unordered_map>
#include <assert.h>

using namespace std;

template <class KEY_T, class VAL_T, class HASH_T>
class LRUCache
{
    private:
        list<pair<KEY_T, VAL_T>> item_list;
        unordered_map<KEY_T, decltype(item_list.begin()), HASH_T> item_map;
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
        void add(const KEY_T &key, const VAL_T &val);
        VAL_T get(const KEY_T &key);
        bool contains(const KEY_T &key) {return item_map.count(key) > 0;};
        size_t size() const {return cache_size;}
};

#include "lru.cc"
