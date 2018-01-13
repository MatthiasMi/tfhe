#ifndef TFHE_GATE_BOOTSTRAPPING_FUNCTIONS_H
#define TFHE_GATE_BOOTSTRAPPING_FUNCTIONS_H

///@file
///@brief tfhe gate bootstrapping api

#include "tfhe_gate_bootstrapping_structures.h"

//////////////////////////////////////////
// Gate bootstrapping public interface
//////////////////////////////////////////

/** generate default gate bootstrapping parameters */
EXPORT TFheGateBootstrappingParameterSet *new_default_gate_bootstrapping_parameters(int minimum_lambda);

/** generate a random gate bootstrapping secret key */
EXPORT TFheGateBootstrappingSecretKeySet *
new_random_gate_bootstrapping_secret_keyset(const TFheGateBootstrappingParameterSet *params, const int32_t window_size);

/** deletes gate bootstrapping parameters */
EXPORT void delete_gate_bootstrapping_parameters(TFheGateBootstrappingParameterSet *params);

/** deletes a gate bootstrapping secret key */
EXPORT void delete_gate_bootstrapping_secret_keyset(TFheGateBootstrappingSecretKeySet *keyset, const int32_t window_size);

/** deletes a gate bootstrapping secret key */
EXPORT void delete_gate_bootstrapping_cloud_keyset(TFheGateBootstrappingCloudKeySet *keyset, const int32_t window_size);

/** generate a new unititialized ciphertext */
EXPORT LweSample *new_gate_bootstrapping_ciphertext(const TFheGateBootstrappingParameterSet *params);

/** generate a new unititialized ciphertext array of length nbelems */
EXPORT LweSample *new_gate_bootstrapping_ciphertext_array(int nbelems, const TFheGateBootstrappingParameterSet *params);

/** deletes a ciphertext */
EXPORT void delete_gate_bootstrapping_ciphertext(LweSample *sample);

/** deletes a ciphertext array of length nbelems */
EXPORT void delete_gate_bootstrapping_ciphertext_array(int nbelems, LweSample *samples);

/** encrypts a boolean */
EXPORT void bootsSymEncrypt(LweSample *result, int message, const TFheGateBootstrappingSecretKeySet *params);

/** decrypts a boolean */
EXPORT int bootsSymDecrypt(const LweSample *sample, const TFheGateBootstrappingSecretKeySet *params);

/** bootstrapped Constant (true or false) trivial Gate */
EXPORT void bootsCONSTANT(LweSample *result, int32_t value, const TFheGateBootstrappingCloudKeySet *bk, const int32_t window_size);

/** bootstrapped Nand Gate */
EXPORT void
bootsNAND(LweSample *result, const LweSample *ca, const LweSample *cb, const TFheGateBootstrappingCloudKeySet *bk, const int32_t window_size);
/** bootstrapped Or Gate:  */
EXPORT void
bootsOR(LweSample *result, const LweSample *ca, const LweSample *cb, const TFheGateBootstrappingCloudKeySet *bk, const int32_t window_size);
/** bootstrapped And Gate: result = a and b */
EXPORT void
bootsAND(LweSample *result, const LweSample *ca, const LweSample *cb, const TFheGateBootstrappingCloudKeySet *bk, const int32_t window_size);
/** bootstrapped Xor Gate: result = a xor b */
EXPORT void
bootsXOR(LweSample *result, const LweSample *ca, const LweSample *cb, const TFheGateBootstrappingCloudKeySet *bk, const int32_t window_size);
/** bootstrapped Xnor Gate: result = (a==b) */
EXPORT void
bootsXNOR(LweSample *result, const LweSample *ca, const LweSample *cb, const TFheGateBootstrappingCloudKeySet *bk, const int32_t window_size);
/** bootstrapped Not Gate: result = not(a) */
EXPORT void bootsNOT(LweSample *result, const LweSample *ca, const TFheGateBootstrappingCloudKeySet *bk, const int32_t window_size);
/** bootstrapped Copy Gate: result = a */
EXPORT void bootsCOPY(LweSample *result, const LweSample *ca, const TFheGateBootstrappingCloudKeySet *bk, const int32_t window_size);
/** bootstrapped Nor Gate: result = not(a or b) */
EXPORT void
bootsNOR(LweSample *result, const LweSample *ca, const LweSample *cb, const TFheGateBootstrappingCloudKeySet *bk, const int32_t window_size);
/** bootstrapped AndNY Gate: not(a) and b */
EXPORT void
bootsANDNY(LweSample *result, const LweSample *ca, const LweSample *cb, const TFheGateBootstrappingCloudKeySet *bk, const int32_t window_size);
/** bootstrapped AndYN Gate: a and not(b) */
EXPORT void
bootsANDYN(LweSample *result, const LweSample *ca, const LweSample *cb, const TFheGateBootstrappingCloudKeySet *bk, const int32_t window_size);
/** bootstrapped OrNY Gate: not(a) or b */
EXPORT void
bootsORNY(LweSample *result, const LweSample *ca, const LweSample *cb, const TFheGateBootstrappingCloudKeySet *bk, const int32_t window_size);
/** bootstrapped OrYN Gate: a or not(b) */
EXPORT void
bootsORYN(LweSample *result, const LweSample *ca, const LweSample *cb, const TFheGateBootstrappingCloudKeySet *bk, const int32_t window_size);

/** bootstrapped Mux(a,b,c) = a?b:c */
EXPORT void bootsMUX(LweSample *result, const LweSample *a, const LweSample *b, const LweSample *c,
                     const TFheGateBootstrappingCloudKeySet *bk, const int32_t window_size);

#endif// TFHE_GATE_BOOTSTRAPPING_FUNCTIONS_H
