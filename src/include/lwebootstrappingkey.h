#ifndef LweBOOTSTRAPPINGKEY_H
#define LweBOOTSTRAPPINGKEY_H

///@file
///@brief This file contains the declaration of bootstrapping key structures

#include "tfhe_core.h"


struct LweBootstrappingKey{
    const LweParams* in_out_params; ///< paramètre de l'input et de l'output. key: s
    const TGswParams* bk_params; ///< params of the Gsw elems in bk. key: s"
    const TLweParams* accum_params; ///< params of the accum variable key: s"
    const LweParams* extract_params; ///< params after extraction: key: s'
    TGswSample* bk; ///< the bootstrapping key (s->s")
    LweKeySwitchKey* ks; ///< the keyswitch key (s'->s)
    const uint32_t window_size;


#ifdef __cplusplus
   LweBootstrappingKey(const LweParams* in_out_params,
    const TGswParams* bk_params,
    const TLweParams* accum_params,
    const LweParams* extract_params,
    TGswSample* bk,
    LweKeySwitchKey* ks,
    const uint32_t window_size);
    ~LweBootstrappingKey();
    LweBootstrappingKey(const LweBootstrappingKey&) = delete;
    void operator=(const LweBootstrappingKey&) = delete;

#endif


};


struct LweBootstrappingKeyFFT {
    const LweParams* in_out_params; ///< paramètre de l'input et de l'output. key: s
    const TGswParams* bk_params; ///< params of the Gsw elems in bk. key: s"
    const TLweParams* accum_params; ///< params of the accum variable key: s"
    const LweParams* extract_params; ///< params after extraction: key: s'
    const TGswSampleFFT* bkFFT; ///< the bootstrapping key (s->s")
    const LweKeySwitchKey* ks; ///< the keyswitch key (s'->s)
    const uint32_t window_size;


#ifdef __cplusplus
   LweBootstrappingKeyFFT(const LweParams* in_out_params,
    const TGswParams* bk_params,
    const TLweParams* accum_params,
    const LweParams* extract_params,
    const TGswSampleFFT* bkFFT,
    const LweKeySwitchKey* ks,
    const uint32_t window_size);
    ~LweBootstrappingKeyFFT();
    LweBootstrappingKeyFFT(const LweBootstrappingKeyFFT&) = delete;
    void operator=(const LweBootstrappingKeyFFT&) = delete;

#endif


};


//allocate memory space for a LweBootstrappingKey
EXPORT LweBootstrappingKey* alloc_LweBootstrappingKey();
EXPORT LweBootstrappingKey* alloc_LweBootstrappingKey_array(int nbelts);

//free memory space for a LweBootstrappingKey
EXPORT void free_LweBootstrappingKey(LweBootstrappingKey* ptr);
EXPORT void free_LweBootstrappingKey_array(int nbelts, LweBootstrappingKey* ptr);

//initialize the LweBootstrappingKey structure
//(equivalent of the C++ constructor)
EXPORT void init_LweBootstrappingKey(LweBootstrappingKey* obj, int32_t ks_t, int32_t ks_basebit, const LweParams* in_out_params, const TGswParams* bk_params, const uint32_t window_size);
EXPORT void init_LweBootstrappingKey_array(int32_t nbelts, LweBootstrappingKey* obj, int32_t ks_t, int32_t ks_basebit, const LweParams* in_out_params, const TGswParams* bk_params, const uint32_t window_size);

//destroys the LweBootstrappingKey structure
//(equivalent of the C++ destructor)
EXPORT void destroy_LweBootstrappingKey(LweBootstrappingKey* obj, const uint32_t window_size);
EXPORT void destroy_LweBootstrappingKey_array(int32_t nbelts, LweBootstrappingKey* obj, const uint32_t window_size);

//allocates and initialize the LweBootstrappingKey structure
//(equivalent of the C++ new)
EXPORT LweBootstrappingKey* new_LweBootstrappingKey(const int32_t ks_t, const int32_t ks_basebit, const LweParams* in_out_params, const TGswParams* bk_params, const uint32_t window_size);
EXPORT LweBootstrappingKey* new_LweBootstrappingKey_array(int32_t nbelts, const int32_t ks_t, const int32_t ks_basebit, const LweParams* in_out_params, const TGswParams* bk_params, const uint32_t window_size);

//destroys and frees the LweBootstrappingKey structure
//(equivalent of the C++ delete)
EXPORT void delete_LweBootstrappingKey(LweBootstrappingKey* obj, const uint32_t window_size);
EXPORT void delete_LweBootstrappingKey_array(int32_t nbelts, LweBootstrappingKey* obj, const uint32_t window_size);
//allocate memory space for a LweBootstrappingKeyFFT
EXPORT LweBootstrappingKeyFFT* alloc_LweBootstrappingKeyFFT();
EXPORT LweBootstrappingKeyFFT* alloc_LweBootstrappingKeyFFT_array(int nbelts);

//free memory space for a LweBootstrappingKeyFFT
EXPORT void free_LweBootstrappingKeyFFT(LweBootstrappingKeyFFT* ptr);
EXPORT void free_LweBootstrappingKeyFFT_array(int nbelts, LweBootstrappingKeyFFT* ptr);

//initialize the LweBootstrappingKeyFFT structure
//(equivalent of the C++ constructor)
EXPORT void init_LweBootstrappingKeyFFT(LweBootstrappingKeyFFT* obj, const LweBootstrappingKey* bk, const uint32_t window_size);
EXPORT void init_LweBootstrappingKeyFFT_array(int32_t nbelts, LweBootstrappingKeyFFT* obj, const LweBootstrappingKey* bk, const uint32_t window_size);

//destroys the LweBootstrappingKeyFFT structure
//(equivalent of the C++ destructor)
EXPORT void destroy_LweBootstrappingKeyFFT(LweBootstrappingKeyFFT* obj, const uint32_t window_size);
EXPORT void destroy_LweBootstrappingKeyFFT_array(int32_t nbelts, LweBootstrappingKeyFFT* obj, const uint32_t window_size);

//allocates and initialize the LweBootstrappingKeyFFT structure
//(equivalent of the C++ new)
EXPORT LweBootstrappingKeyFFT* new_LweBootstrappingKeyFFT(const LweBootstrappingKey* bk, const uint32_t window_size);
EXPORT LweBootstrappingKeyFFT* new_LweBootstrappingKeyFFT_array(int32_t nbelts, const LweBootstrappingKey* bk, const uint32_t window_size);

//destroys and frees the LweBootstrappingKeyFFT structure
//(equivalent of the C++ delete)
EXPORT void delete_LweBootstrappingKeyFFT(LweBootstrappingKeyFFT* obj, const uint32_t window_size);
EXPORT void delete_LweBootstrappingKeyFFT_array(int32_t nbelts, LweBootstrappingKeyFFT* obj, const uint32_t window_size);
#endif
