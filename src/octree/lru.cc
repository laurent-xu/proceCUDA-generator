template <class KEY_T, class VAL_T, class HASH_T>
void LRUCache<KEY_T, VAL_T, HASH_T>::add(const KEY_T &key, const VAL_T &val)
{
    auto it = item_map.find(key);
    if(it != item_map.end())
    {
        item_list.erase(it->second);
        item_map.erase(it);
    }
    item_list.push_front(make_pair(key, val));
    item_map.insert(make_pair(key, item_list.begin()));
    clean();
};

template <class KEY_T, class VAL_T, class HASH_T>
VAL_T LRUCache<KEY_T, class VAL_T, class HASH_T>::get(const KEY_T &key)
{
    auto it = item_map.find(key);
    item_list.splice(item_list.begin(), item_list, it->second);
    return it->second->second;
};
