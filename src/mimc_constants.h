
#ifndef DF_EXPLORER_MIMC_CONSTANTS_H
#define DF_EXPLORER_MIMC_CONSTANTS_H

#include <gmpxx.h>

class MimcConstants
{
public:
    static const int32_t rounds = 220;

public:
    static MimcConstants &getInstance()
    {
        static MimcConstants instance;
        return instance;
    }

    static const mpz_class& get_p() { return getInstance().m_p; }
    static const mpz_class& c_at(size_t index) { return getInstance().m_c[index]; }

    MimcConstants(const MimcConstants &) = delete;
    MimcConstants &operator=(const MimcConstants &) = delete;

private:
    mpz_class m_p;
    mpz_class m_c[rounds];

    MimcConstants();
    ~MimcConstants() = default;
};

#endif //DF_EXPLORER_MIMC_CONSTANTS_H