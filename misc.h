#pragma once

static int ext_nt(const std::string& filename, int n_expc)
{
    std::vector<std::string> nums;
    std::string cur;
    std::string ints = "0123456789";
    bool was_int = false;
    for (const auto c: filename)
    {
        bool is_int = ints.find(c) != std::string::npos;
        if (is_int)
        {
            cur += c;
        }
        else if (was_int)
        {
            nums.push_back(cur);
            cur = "";
        }
        if (is_int) was_int = true;
        else was_int = false;
    }
    if (cur.length() > 0) nums.push_back(cur);
    for (auto ii: nums)
    {
        if (ii.length() == n_expc)
        {
            std::istringstream iss(ii);
            int out;
            iss >> out;
            return out;
        }
    }
    return 0;
}

static inline bool ends_with(std::string const & value, std::string const & ending)
{
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

static int ids_impl(int ct, const std::string& v)
{
    throw std::invalid_argument( "received invlid value" );
    return -1;
}
template <typename str_t, typename... strs_t> static int ids_impl(int ct, const std::string& v, const str_t& k, const strs_t&... ks)
{
    if (k==v) return ct;
    return ids_impl(ct+1, v, ks...);
}
template <typename... strs_t> static int id_search(const std::string& v, const strs_t&... ks)
{
    return ids_impl(0, v, ks...);
}