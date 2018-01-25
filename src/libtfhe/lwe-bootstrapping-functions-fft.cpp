/*
 * Bootstrapping FFT functions
 */


#ifndef TFHE_TEST_ENVIRONMENT

#include <iostream>
#include <cassert>
#include "tfhe.h"

using namespace std;
#define INCLUDE_ALL
#else
#undef EXPORT
#define EXPORT
#endif


#if defined INCLUDE_ALL || defined INCLUDE_TFHE_INIT_LWEBOOTSTRAPPINGKEY_FFT
#undef INCLUDE_TFHE_INIT_LWEBOOTSTRAPPINGKEY_FFT
//(equivalent of the C++ constructor)
EXPORT void init_LweBootstrappingKeyFFT(LweBootstrappingKeyFFT *obj, const LweBootstrappingKey *bk, const int32_t window_size)
{

    const LweParams *in_out_params = bk->in_out_params;
    const TGswParams *bk_params = bk->bk_params;
    const TLweParams *accum_params = bk_params->tlwe_params;
    const LweParams *extract_params = &accum_params->extracted_lweparams;
    const int n = in_out_params->n;
    const int t = bk->ks->t;
    const int basebit = bk->ks->basebit;
    const int base = bk->ks->base;
    const int N = extract_params->n;

    LweKeySwitchKey *ks = new_LweKeySwitchKey(N, t, basebit, in_out_params);
    // Copy the KeySwitching key
    for (int32_t i = 0; i < N; i++)
    {
        for (int32_t j = 0; j < t; j++)
        {
            for (int32_t p = 0; p < base; p++)
            {
                lweCopy(&ks->ks[i][j][p], &bk->ks->ks[i][j][p], in_out_params);
            }
        }
    }

    // Bootstrapping Key FFT
    int32_t bk_size = (1 << window_size) - 1; // numerator: 2^w - 1
    int32_t ks_size = window_size;            // denominator: w
    int32_t nElem = (bk_size*n)/ks_size;  // cut off

    TGswSampleFFT *bkFFT = new_TGswSampleFFT_array( nElem, bk_params);
    for (int32_t i = 0; i < nElem; ++i)
    {
        tGswToFFTConvert(&bkFFT[i], &bk->bk[i], bk_params);
    }

    new(obj) LweBootstrappingKeyFFT(in_out_params, bk_params, accum_params, extract_params, bkFFT, ks, window_size);
}
#endif



//destroys the LweBootstrappingKeyFFT structure
//(equivalent of the C++ destructor)
EXPORT void destroy_LweBootstrappingKeyFFT(LweBootstrappingKeyFFT *obj, const int32_t window_size)
{
    delete_LweKeySwitchKey((LweKeySwitchKey *) obj->ks);

    int32_t bk_size = (1 << window_size) - 1; // numerator: 2^w - 1
    int32_t ks_size = window_size;            // denominator: w
    int32_t nElem = (bk_size*obj->in_out_params->n)/ks_size;  // cut off

    delete_TGswSampleFFT_array( nElem, (TGswSampleFFT *) obj->bkFFT);

    obj->~LweBootstrappingKeyFFT();
}


EXPORT void tfhe_MuxRotate_FFT(TLweSample *result, TGswSampleFFT *temp, const TLweSample *accum, const TGswSampleFFT* bk, const int32_t* bara, const TGswParams *bk_params, const int32_t n, const int32_t window_size)
{

    //int32_t bk_size = (1 << window_size) - 1; // numerator: 2^w - 1
    int32_t ks_size = window_size;            // denominator: w
    int32_t num_windows = n/ks_size;

    int32_t bk_pos = 0;

    for (int32_t i = 0; i < num_windows; i++)
    {
        int32_t w = bk_pos;
        bool zeroCheck = (bara[w] == 0);
        while(++w < ks_size)
            zeroCheck &= (bara[w] == 0);
        if(zeroCheck != 0) continue; //indeed, this is an easy case!

        // temp = (X^barai-1)*ACC
        tGswFFTClear(temp, bk_params);
        for (w = 0; w < ks_size ; w++)
            tGswFFTMulByXaiMinusOne(temp, bara[bk_pos++], bk + w, bk_params);
        //bk_pos += bk_size;
    }

    // result = temp * BKi
    tLweCopy(result, accum, bk_params->tlwe_params);
    tGswFFTExternMulToTLwe(result, temp, bk_params);
    // ACC += result
    tLweAddTo(result, accum, bk_params->tlwe_params);
    return;
    // ACC += BKi*[(X^barai-1)*ACC]
}


#if defined INCLUDE_ALL || defined INCLUDE_TFHE_BLIND_ROTATE_FFT
#undef INCLUDE_TFHE_BLIND_ROTATE_FFT
/**
 * multiply the accumulator by X^sum(bara_i.s_i)
 * @param accum the TLWE sample to multiply
 * @param bk An array of n TGSW FFT samples where bk_i encodes s_i
 * @param bara An array of n coefficients between 0 and 2N-1
 * @param bk_params The parameters of bk
 */
EXPORT void tfhe_blindRotate_FFT(TLweSample *accum,
                                 const TGswSampleFFT *bkFFT,
                                 const int32_t *bara,
                                 const int32_t n,
                                 const TGswParams *bk_params, const int32_t window_size)

{

    //TGswSampleFFT* temp = new_TGswSampleFFT(bk_params);
    TLweSample *temp = new_TLweSample(bk_params->tlwe_params);
    TLweSample *result = temp;
    TLweSample *temp_accum = accum;
    TGswSampleFFT *tmp = new_TGswSampleFFT(bk_params);

    // tfhe_MuxRotate_FFT(result, tmp, temp_accum, bkFFT + 3*i, bkFFT + 3*i+1, bkFFT + 3*i+2, barai, baraii, baraiii, bk_params);
    tfhe_MuxRotate_FFT(result, tmp, temp_accum, bkFFT, bara, bk_params, n, window_size);
    swap(result, temp_accum);

    if (temp_accum != accum)
    {
        tLweCopy(accum, temp_accum, bk_params->tlwe_params);
    }

    delete_TLweSample(temp);
    //delete_TGswSampleFFT(temp);
}
#endif


#if defined INCLUDE_ALL || defined INCLUDE_TFHE_BLIND_ROTATE_AND_EXTRACT_FFT
#undef INCLUDE_TFHE_BLIND_ROTATE_AND_EXTRACT_FFT
/**
 * result = LWE(v_p) where p=barb-sum(bara_i.s_i) mod 2N
 * @param result the output LWE sample
 * @param v a 2N-elt anticyclic function (represented by a TorusPolynomial)
 * @param bk An array of n TGSW FFT samples where bk_i encodes s_i
 * @param barb A coefficients between 0 and 2N-1
 * @param bara An array of n coefficients between 0 and 2N-1
 * @param bk_params The parameters of bk
 */
EXPORT void tfhe_blindRotateAndExtract_FFT(LweSample *result,
        const TorusPolynomial *v,
        const TGswSampleFFT *bk,
        const int32_t barb,
        const int32_t *bara,
        const int32_t n,
        const TGswParams *bk_params, const int32_t window_size)
{
    const TLweParams *accum_params = bk_params->tlwe_params;
    const LweParams *extract_params = &accum_params->extracted_lweparams;
    const int N = accum_params->N;
    const int _2N = 2 * N;

    // Test polynomial
    TorusPolynomial *testvectbis = new_TorusPolynomial(N);
    // Accumulator
    TLweSample *acc = new_TLweSample(accum_params);

    // testvector = X^{2N-barb}*v
    if (barb != 0) torusPolynomialMulByXai(testvectbis, _2N - barb, v);
    else torusPolynomialCopy(testvectbis, v);
    tLweNoiselessTrivial(acc, testvectbis, accum_params);
    // Blind rotation
    tfhe_blindRotate_FFT(acc, bk, bara, n, bk_params, window_size);
    // Extraction
    tLweExtractLweSample(result, acc, extract_params, accum_params);

    delete_TLweSample(acc);
    delete_TorusPolynomial(testvectbis);
}
#endif


#if defined INCLUDE_ALL || defined INCLUDE_TFHE_BOOTSTRAP_WO_KS_FFT
#undef INCLUDE_TFHE_BOOTSTRAP_WO_KS_FFT
/**
 * result = LWE(mu) iff phase(x)>0, LWE(-mu) iff phase(x)<0
 * @param result The resulting LweSample
 * @param bk The bootstrapping + keyswitch key
 * @param mu The output message (if phase(x)>0)
 * @param x The input sample
 */
EXPORT void tfhe_bootstrap_woKS_FFT(LweSample *result,
                                    const LweBootstrappingKeyFFT *bk,
                                    Torus32 mu,
                                    const LweSample *x,
                                    const int32_t window_size)
{
    const TGswParams *bk_params = bk->bk_params;
    const TLweParams *accum_params = bk->accum_params;
    const LweParams *in_params = bk->in_out_params;
    const int N = accum_params->N;
    const int Nx2 = 2 * N;
    const int n = in_params->n;

    TorusPolynomial *testvect = new_TorusPolynomial(N);
    // int32_t *bara = new int[(3*n)/2]; //TODO CHECK
    int32_t *bara = new int[N];

    // Modulus switching
    int32_t barb = modSwitchFromTorus32(x->b, Nx2);


    /* How bk is stored:
    // E.g for window_size = 1:
    for (int32_t i = 0; i < n; i++)
        tGswSymEncryptInt(&bk->bk[i], kin[i], alpha, rgsw_key);

    // E.g for window_size = 2:
    for (int32_t i = 0; i < n/2; i++) {
        tGswSymEncryptInt(&bk->bk[3*i  ], kin[2*i  ]*   kin[2*i+1] , alpha, rgsw_key);
        tGswSymEncryptInt(&bk->bk[3*i+1], kin[2*i+1]*(1-kin[2*i  ]), alpha, rgsw_key);
        tGswSymEncryptInt(&bk->bk[3*i+2], kin[2*i  ]*(1-kin[2*i+1]), alpha, rgsw_key);
    }
    */

    /* How bara is stored:
    // E.g for window_size = 1:
    for (int32_t i = 0; i < n; i++)
        modSwitchFromTorus32(x->a[i], Nx2);

    // E.g for window_size = 2:
    for (int32_t i = 0; i < n/2; i++) { //TODO , const int32_t window_size = 1
        bara[3*i  ] = modSwitchFromTorus32(x->a[2*i  ]+x->a[2*i+1], Nx2);
        bara[3*i+1] = modSwitchFromTorus32(x->a[2*i+1], Nx2);
        bara[3*i+2] = modSwitchFromTorus32(x->a[2*i], Nx2);
    }
    */

    int32_t bk_size = (1 << window_size) - 1; // numerator: 2^w - 1
    int32_t ks_size = window_size;            // denominator: w
    int32_t num_windows =n/ks_size;

    int32_t bara_pos = 0, a_pos = 0;
    int32_t message, b;

    for (int32_t window = 0; window < num_windows; window++) // Window
    {
        //std::cout << window << " ... current window index" << std::endl;
        for (int32_t W = 0; W < bk_size; W++) // enumerate all subsets of [1:window_size], in 2 loops, omit empty set.
        {
            message = 0; // To accumulate the sum: \sum_i ( s_i )
            int32_t subset = bk_size - W;
            //std::cout << subset << std::endl;
            for (int32_t w = 0; w < window_size ; w++)  // Loop through (little-endian) binary decomposition of W: window_size = log( bk_size )
            {
                b  =subset&1; // Extract last bit
                subset>>=1;   // Divide by 2
                //std::cout << w << b << std::endl;
                if (b)
                    message += x->a[a_pos + w];
            }
            //cout  << message << endl<< endl;
            bara[bara_pos + W] = modSwitchFromTorus32(message, Nx2);
            a_pos += ks_size;
        }
        bara_pos += bk_size;    // Compute start position of next window in arrays
    }
    //std::cout << message << bara_pos << a_pos << window_size << " ... window_size means " << num_windows << " windows have to be processed to cover all " << n << " coefficients.";
    //std::cout << " Arrays have to be of appropriate size, and the storage overhead is: bk_size / ks_size = " << bk_size  << " / " << ks_size << " = "<< (double) bk_size / ks_size << std::endl;


    // the initial testvec = [mu,mu,mu,...,mu]
    for (int i = 0; i < N; i++) testvect->coefsT[i] = mu;

    // Bootstrapping rotation and extraction
    tfhe_blindRotateAndExtract_FFT(result, testvect, bk->bkFFT, barb, bara, n, bk_params, window_size);


    delete[] bara;
    delete_TorusPolynomial(testvect);
}
#endif


#if defined INCLUDE_ALL || defined INCLUDE_TFHE_BOOTSTRAP_FFT
#undef INCLUDE_TFHE_BOOTSTRAP_FFT
/**
 * result = LWE(mu) iff phase(x)>0, LWE(-mu) iff phase(x)<0
 * @param result The resulting LweSample
 * @param bk The bootstrapping + keyswitch key
 * @param mu The output message (if phase(x)>0)
 * @param x The input sample
 */
EXPORT void tfhe_bootstrap_FFT(LweSample *result,
                               const LweBootstrappingKeyFFT *bk,
                               Torus32 mu,
                               const LweSample *x, const int32_t window_size)
{
    LweSample *u = new_LweSample(&bk->accum_params->extracted_lweparams);

    tfhe_bootstrap_woKS_FFT(u, bk, mu, x, window_size);
    // Key switching
    lweKeySwitch(result, bk->ks, u);

    delete_LweSample(u);
}
#endif


//allocate memory space for a LweBootstrappingKeyFFT

EXPORT LweBootstrappingKeyFFT *alloc_LweBootstrappingKeyFFT()
{
    return (LweBootstrappingKeyFFT *) malloc(sizeof(LweBootstrappingKeyFFT));
}
EXPORT LweBootstrappingKeyFFT *alloc_LweBootstrappingKeyFFT_array(int32_t nbelts)
{
    return (LweBootstrappingKeyFFT *) malloc(nbelts * sizeof(LweBootstrappingKeyFFT));
}

//free memory space for a LweKey
EXPORT void free_LweBootstrappingKeyFFT(LweBootstrappingKeyFFT *ptr)
{
    free(ptr);
}
EXPORT void free_LweBootstrappingKeyFFT_array(int32_t nbelts, LweBootstrappingKeyFFT *ptr)
{
    free(ptr);
}

//initialize the key structure

EXPORT void init_LweBootstrappingKeyFFT_array(int32_t nbelts, LweBootstrappingKeyFFT *obj, const LweBootstrappingKey *bk, const int32_t window_size)
{
    for (int32_t i = 0; i < nbelts; i++)
    {
        init_LweBootstrappingKeyFFT(obj + i, bk, window_size);
    }
}


EXPORT void destroy_LweBootstrappingKeyFFT_array(int32_t nbelts, LweBootstrappingKeyFFT *obj, const int32_t window_size)
{
    for (int32_t i = 0; i < nbelts; i++)
    {
        destroy_LweBootstrappingKeyFFT(obj + i, window_size);
    }
}

//allocates and initialize the LweBootstrappingKeyFFT structure
//(equivalent of the C++ new)
EXPORT LweBootstrappingKeyFFT *new_LweBootstrappingKeyFFT(const LweBootstrappingKey *bk, const int32_t window_size)
{
    LweBootstrappingKeyFFT *obj = alloc_LweBootstrappingKeyFFT();
    init_LweBootstrappingKeyFFT(obj, bk, window_size);
    return obj;
}
EXPORT LweBootstrappingKeyFFT *new_LweBootstrappingKeyFFT_array(int32_t nbelts, const LweBootstrappingKey *bk, const int32_t window_size)
{
    LweBootstrappingKeyFFT *obj = alloc_LweBootstrappingKeyFFT_array(nbelts);
    init_LweBootstrappingKeyFFT_array(nbelts, obj, bk, window_size);
    return obj;
}

//destroys and frees the LweBootstrappingKeyFFT structure
//(equivalent of the C++ delete)
EXPORT void delete_LweBootstrappingKeyFFT(LweBootstrappingKeyFFT *obj, const int32_t window_size)
{
    destroy_LweBootstrappingKeyFFT(obj, window_size);
    free_LweBootstrappingKeyFFT(obj);
}
EXPORT void delete_LweBootstrappingKeyFFT_array(int32_t nbelts, LweBootstrappingKeyFFT *obj, const int32_t window_size)
{
    destroy_LweBootstrappingKeyFFT_array(nbelts, obj, window_size);
    free_LweBootstrappingKeyFFT_array(nbelts, obj);
}






