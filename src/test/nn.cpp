// ----------------------------------------------------------------------------|
// Title      : Fast Homomorphic Evaluation of Deep Discretized Neural Networks
// Project    : Showcase Fast Fully Homomorphic Evaluation of Encrypted Inputs
//              using Deep Discretized Neural Networks hence preserving Privacy
// ----------------------------------------------------------------------------|
// File       : nn.cpp
// Authors    : Florian Bourse      <Florian.Bourse@ens.fr>
//              Michele Minelli     <Michele.Minelli@ens.fr>
//              Matthias Minihold   <Matthias.Minihold@RUB.de>
//              Pascal Paillier     <Pascal.Paillier@cryptoexperts.com>
//
// Reference  : TFHE: Fast Fully Homomorphic Encryption Library over the Torus
//              https://github.com/tfhe
// ----------------------------------------------------------------------------|
// Description:
//     Showcases how to efficiently evaluate privacy-perserving neural networks.
// ----------------------------------------------------------------------------|
// Revisions  :
// Date        Version  Description
// 2017-11-16  0.3.0    Version for github, referenced by ePrint32_t paper
// ----------------------------------------------------------------------------|


// Includes
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <string>
#include <sys/time.h>
// tfhe-lib
#include "tfhe.h"
#include "tfhe_garbage_collector.h"
#include "tlwe.h"
#include "tlwe_functions.h"
#include "tgsw.h"
#include "lwesamples.h"
#include "lwekey.h"
#include "lweparams.h"
#include "polynomials.h"
// Multi-processing
#include <sys/wait.h>
#include <unistd.h>
// Handling
#include <sstream>
#include <ctime>
#include <signal.h>
volatile sig_atomic_t got_sigterm = 0;  /// Declared volatile to let (optimizing) compilers know it changes asynchronously.


// Defines
#define PROCESSES  4   /// Use multiple processes.
#define VERBOSE    true /// Some output.
#define DEBUG      true /// Some more output.
#define GENERATE   true /// If true: Generates fresh keys, else reuses saved ones.
#define MODE_STATS true /// If true: Writes a file with some statistics, else not.
#define MODE_LATEX true /// If true: Writes a file with some LaTeX formatted listing, else not.
#define MODE_BATCH true /// If true: Batching multiple inputs into one polynomial, else individual.
#define MODE_NOISE true /// If true: Adds Gaussian noise to samples, else trivially broken or for testing.


/*
# ePrint32_t and github release:
+ TODO rename functions, files,
+ TODO add a separate make for them,
+ TODO maybe use and test FFTW,
+ TODO answer&clear all TODOs, comments throughout the files,
+ TODO integrate the tfhelib sources correctly as submodule from github,
+ TODO add disclaimer, copyright/-left/licence notes like this?,
    Disclaimer
    This is an implementation for academic purposes, hence NOT to be used AS AS, not to be considered SECURED nor is the code well documented. Following the code-sharing idea, we publish it under license as a way to support  and to allow the community to verify easily both the correctness and the efficiency of this homomorphic evaluation.
    https://creativecommons.org/publicdomain/zero/1.0/
+ [DONE] clean nn.cpp
+ [DONE] annotate nn.cpp for future experiments/optimization,
+ [DONE] prepare nn.cpp for parallelization (processes, threads),
+ [DONE] adapt nn.cpp to handle multiple hidden layers as in the hom-mnist.py
+ [DONE] add codeblocks project for building,
+ [DONE] add howto use for manual building, e.g.,
            cd build (files are here)
            cmake ../src/ -DENABLE_TESTS=on -DENABLE_FFTW=off -DENABLE_NAYUKI=off -DCMAKE_BUILD_TYPE=optim
            make
            cd test
            ./nn-spqlios-fma (or ./nn-spqlios-avx)

// Security constants used for submission
#define SECLEVEL 80
#define SECNOISE true
#define SECALPHA pow(2., -30)
#define SEC_PARAMS_STDDEV    pow(2., -30)
#define SEC_PARAMS_n  450                   ///  LweParams
#define SEC_PARAMS_N 1024                   /// TLweParams
#define SEC_PARAMS_k    1                   /// TLweParams
#define SEC_PARAMS_BK_STDDEV pow(2., -38)   /// TLweParams
#define SEC_PARAMS_BK_BASEBITS  8           /// TGswParams
#define SEC_PARAMS_BK_LENGTH    3           /// TGswParams
#define SEC_PARAMS_KS_STDDEV pow(2., -20)   /// Key Switching Params
#define SEC_PARAMS_KS_BASEBITS  1           /// Key Switching Params
#define SEC_PARAMS_KS_LENGTH   18           /// Key Switching Params
*/

// Security constants used for testing
#define SEC_LEVEL            80
#define SEC_PARAMS_n         450            ///  LweParams
#define SEC_ALPHA            pow(2., -30)   ///  LweParams
#define SEC_PARAMS_STDDEV    pow(2., -30)   /// TLweParams and LweParams (max)
#define SEC_PARAMS_N         1024           /// TLweParams
#define SEC_PARAMS_k         1              /// TLweParams
#define SEC_PARAMS_BK_STDDEV pow(2., -36)   /// TLweParams
#define SEC_PARAMS_BK_BASEBITS  10          /// TGswParams
#define SEC_PARAMS_BK_LENGTH    3           /// TGswParams
// Best parameters evar!
#define SEC_PARAMS_KS_STDDEV pow(2., -17)   /// Key Switching Params
#define SEC_PARAMS_KS_BASEBITS  3           /// Key Switching Params
#define SEC_PARAMS_KS_LENGTH    5           /// Key Switching Params

// The expected topology of the provided neural network is 256:30:10
#define NUM_NEURONS_INPUT  256
#define NUM_NEURONS_HIDDEN 30
#define NUM_NEURONS_OUTPUT 10
#define NUM_NEURON_LAYERS 3

#define CARD_TESTSET 10000
#define MSG_SLOTS    700
#define TORUS_SLOTS  400

// Files are expected in the executable's directory
#define FILE_TEST           "tests.txt"
#define FILE_TXT_IMG        "txt_img_test.txt"
#define FILE_TXT_BIASES     "txt_biases.txt"
#define FILE_TXT_WEIGHTS    "txt_weights.txt"
#define FILE_TXT_LABELS     "txt_labels.txt"
#define FILE_STATISTICS     "results_stats.txt"
#define FILE_LATEX          "results_LaTeX.tex"
#define FILE_USER_KEY       "key_user_secret.bin"
#define FILE_CLOUD_KEY      "key_cloud.bin"

// Tweak trained neural network
#define THRESHOLD_WEIGHTS  9
#define THRESHOLD_SCORE -100


using namespace std;


// Internal Functions
/// deleteMatrix
void deleteMatrix(int32_t**  matrix, int32_t dim_mat);

/// deleteTensor
void deleteTensor(int32_t*** tensor, int32_t dim_mat, const int32_t* dim_vec);

/// Display/Write some statistics and/or LaTeX output
void writeStatistics();

/// Handle user-initiated end of computation i.e. SIGINT, when interrupted from keyboard (CRTL+C) and save progress!
void my_signal_handler(int32_t sig);

/// Generate bootstrapping parameters for FHE_NN
TFheGateBootstrappingParameterSet *our_default_bootstrapping_parameters(int32_t minimum_lambda);

/// (un)batched (Multi-Threaded) Processing of given Neural Net that writes/outputs some global statistics.
int32_t processNeuralNet();


// Global Counters, Timings, Variables
int32_t count_images;
int32_t count_errors;
int32_t count_errors_with_failed_bs;
int32_t count_disagreements;
int32_t count_disagreements_with_failed_bs;
int32_t count_disag_pro_clear;
int32_t count_disag_pro_hom;
int32_t count_wrong_bs;
double total_time_classifications, total_time_bootstrappings;

double avg_img;
double avg_img_bs;
double avg_time_per_classification;
double avg_time_per_bootstrapping;
double rel_error_percent;
double rel_time_bs_percent;


// Global Constants
const int32_t num_processes = PROCESSES;
const bool batched          = MODE_BATCH;

// Security
const int32_t minimum_lambda = SEC_LEVEL;
const double alpha = SEC_ALPHA;

static const int32_t n = SEC_PARAMS_n;
static const int32_t N = SEC_PARAMS_N;
static const int32_t k = SEC_PARAMS_k;
static const double max_stdev = SEC_PARAMS_STDDEV;   //mulBySqrtTwoOverPi(pow(2., -30));    //max standard deviation for a 1/4 msg space

static const int32_t bk_Bgbit = SEC_PARAMS_BK_BASEBITS;  //<-- ld, thus: 2^10
static const int32_t bk_l     = SEC_PARAMS_BK_LENGTH;
static const double bk_stdev  = SEC_PARAMS_BK_STDDEV; //mulBySqrtTwoOverPi(pow(2., -30));   //standard deviation

static const int32_t ks_basebit = SEC_PARAMS_KS_BASEBITS; //<-- ld, thus: 2^1
static const int32_t ks_length  = SEC_PARAMS_KS_LENGTH;
static const double ks_stdev    = SEC_PARAMS_KS_STDDEV; //mulBySqrtTwoOverPi(pow(2., -30));   //standard deviation

const bool noisyLWE      = MODE_NOISE;

// Input data
const int32_t num_images = CARD_TESTSET;

// Network specific
const int32_t num_neuron_layers = NUM_NEURON_LAYERS;
const int32_t num_wire_layers = NUM_NEURON_LAYERS - 1;
const int32_t num_neurons_in = NUM_NEURONS_INPUT;
const int32_t num_neurons_hidden = NUM_NEURONS_HIDDEN;
const int32_t num_neurons_out = NUM_NEURONS_OUTPUT;
const int32_t total_num_hidden_neurons = CARD_TESTSET * NUM_NEURONS_HIDDEN;  //TODO (sum all num_neurons_hidden)*n_images


// Derived quantitie (fsor statistics output)
const double clocks2seconds = 1. / CLOCKS_PER_SEC;
const double avg_bs = 1./NUM_NEURONS_HIDDEN;
const double avg_total_bs = 1./total_num_hidden_neurons;
const int32_t batch = (CARD_TESTSET+num_processes-1)/num_processes; // =ceil(1./PROCESSES*n_images) quick ceiling hack


// TODO µ Suggestion: make topology definition variable, passing on CMD line
// Vector of number of neurons in layer_in, layer_H1, layer_H2, ..., layer_Hd, layer_out;
const int32_t topology[num_neuron_layers] = {num_neurons_in, num_neurons_hidden, num_neurons_out};

const int32_t space_msg = MSG_SLOTS;
const int32_t space_after_bs = TORUS_SLOTS;

const bool clamp_biases  = false;
const bool clamp_weights = false;

const int32_t threshold_biases  = THRESHOLD_WEIGHTS;
const int32_t threshold_weights = THRESHOLD_WEIGHTS;
const int32_t threshold_scores  = THRESHOLD_SCORE;


TFheGateBootstrappingParameterSet *our_default_bootstrapping_parameters(int32_t minimum_lambda)
{
    if ( minimum_lambda != 80 ) // (minimum_lambda < 80)  || (128 < minimum_lambda) ) //TODO change hardcoded security level.
        cerr << "Sorry, for now, the parameters are only implemented for security-level of about 80 [bit].\n";

    LweParams  *params_in    = new_LweParams (n,    ks_stdev, max_stdev);
    TLweParams *params_accum = new_TLweParams(N, k, bk_stdev, max_stdev);
    TGswParams *params_bk    = new_TGswParams(bk_l, bk_Bgbit, params_accum);

    TfheGarbageCollector::register_param(params_in);
    TfheGarbageCollector::register_param(params_accum);
    TfheGarbageCollector::register_param(params_bk);
    TFheGateBootstrappingParameterSet *params;
    params = new TFheGateBootstrappingParameterSet(ks_length, ks_basebit, params_in, params_bk);
    return params;
}

void tfhe_ReLU_FFT(LweSample *result,
		   const LweBootstrappingKeyFFT *bk,
		   Torus32 mu,
		   int32_t slots,
		   const LweSample *u) {

    const TGswParams *bk_params = bk->bk_params;
    const TLweParams *accum_params = bk->accum_params;
    const LweParams *in_params = bk->in_out_params;
    const int32_t N = accum_params->N;
    const int32_t Nx2 = 2 * N;
    const int32_t n = in_params->n;

    LweSample *x = new_LweSample(in_params);
    lweKeySwitch(x, bk->ks, u);

    TorusPolynomial *testvect = new_TorusPolynomial(N);
    // int32_t *bara = new int[(3*n)/2];
    int32_t *bara = new int[N];

    // Modulus switching
    int32_t barb = modSwitchFromTorus32(x->b, Nx2);
    for (int32_t i = 0; i < n/2; i++) {
        bara[3*i] = modSwitchFromTorus32(x->a[2*i]+x->a[2*i+1], Nx2);
        bara[3*i+1] = modSwitchFromTorus32(x->a[2*i], Nx2);
        bara[3*i+2] = modSwitchFromTorus32(x->a[2*i+1], Nx2);
    }

    // the initial testvec = [mu,mu,mu,...,mu]
    for (int32_t i = 0; i < N/2; i++) testvect->coefsT[i] = ((i*slots)+slots/2)/Nx2 * mu;
    for (int32_t i = N/2; i < N; i++) testvect->coefsT[i] = 0;

    // Bootstrapping rotation and extraction
    tfhe_blindRotateAndExtract_FFT(result, testvect, bk->bkFFT, barb, bara, n, bk_params);

    delete[] bara;
    delete_TorusPolynomial(testvect);
    delete_LweSample(x);
}

//Inplace max: result = max(x,result)
void tfhe_max_FFT(LweSample *result, const LweSample *x,
		   const LweBootstrappingKeyFFT *bk,
		   Torus32 mu,
		   int32_t slots)
{
    const TLweParams *accum_params = bk->accum_params;
    const LweParams *final_params = &(accum_params->extracted_lweparams);
    lweSubTo(result, x, final_params);
    LweSample *u = new_LweSample(final_params);
    lweCopy(u, result, final_params);
    tfhe_ReLU_FFT(result, bk, mu, slots, u);
    lweAddTo(result, x, final_params);
}


int32_t processNeuralNet()
{

    // Some derived values of future interest
    // const int32_t torus_slices_upper_half[num_wire_layers] = {space_msg,       space_after_bs};
    // const int32_t torus_slices_all       [num_wire_layers] = {2*space_msg,   2*space_after_bs};
    // const int32_t torus_slices_allowed   [num_wire_layers] = {2*space_msg-1, 2*space_after_bs-1};


    // Huge arrays
    LagrangeHalfCPolynomial ***fft_weights = new LagrangeHalfCPolynomial**[num_wire_layers];
    int32_t*** weights = new int32_t**[num_wire_layers];  // allocate and fill matrices holding the weights
    int32_t ** biases  = new int32_t* [num_wire_layers];  // allocate and fill vectors holding the biases
    int32_t ** images  = new int32_t* [num_images];
    int32_t  * labels  = new int32_t  [num_images];


    // Temporary variables
    string line;
    int32_t el, l;
    int32_t num_neurons_current_layer_in, num_neurons_current_layer_out;

    if (VERBOSE) cout << "[GEN] Generate parameters for a security level (according to  CGGI16a) of at least " << minimum_lambda << " [bit]." << endl;
    TFheGateBootstrappingParameterSet *params = our_default_bootstrapping_parameters(minimum_lambda);
    //instead of: TFheGateBootstrappingParameterSet *params = new_default_gate_bootstrapping_parameters(minimum_lambda);

    TFheGateBootstrappingSecretKeySet *secret;

    if(GENERATE)
    {
        // Cloud sets up keys
        if (VERBOSE) cout << "[GEN] Generate the secret keyset." << endl;
        secret = new_random_gate_bootstrapping_secret_keyset(params, window_size);
        //const LweBootstrappingKey* const cloud_bk = secret->cloud.bk;

        // Save keys to file
        if (VERBOSE) cout << "[GEN] User's secret key is exported to file: " << FILE_USER_KEY << endl;
        FILE* sk = fopen(FILE_USER_KEY,"wb");
        export_tfheGateBootstrappingSecretKeySet_toFile(sk, secret);
        fclose(sk);

        // TODO Separate into user / cloud
        /*
        if (VERBOSE) cout << "[GEN] Cloud's key is exported to file: " << FILE_CLOUD_KEY << endl << endl;
        FILE* cloud_key = fopen(FILE_CLOUD_KEY,"wb");
        export_tfheGateBootstrappingCloudKeySet_toFile(cloud_key, &secret->cloud);
        fclose(cloud_key);
        */

    }
    else
    {
        // Read keys from file
        if (VERBOSE) cout << "[READ] User's secret key is read from file: " << FILE_USER_KEY << endl;
        FILE* sk = fopen(FILE_USER_KEY,"rb");
        secret = new_tfheGateBootstrappingSecretKeySet_fromFile(sk);
        fclose(sk);

        /*
        if (VERBOSE) cout << "[READ] Cloud's key is read from file: " << FILE_CLOUD_KEY << endl << endl ;
        FILE* cloud_key = fopen(FILE_CLOUD_KEY,"rb");
        TFheGateBootstrappingCloudKeySet *cloud_keyset = new_tfheGateBootstrappingCloudKeySet_fromFile(cloud_key);
        fclose(cloud_key);
        */
    }

    // Keys
    const LweBootstrappingKeyFFT *bs_key = secret->cloud.bkFFT;
    const TLweKey *tLwe_key = &((secret->tgsw_key)->tlwe_key);
    const TGswParams *tgsw_params = ((secret->cloud.bk)->bk_params);
    const TLweParams *tlwe_params = ((secret->cloud.bk)->accum_params);
    const LweParams *final_params = &(tlwe_params->extracted_lweparams);
    LweKey *final_key = new_LweKey(final_params);
    tLweExtractKey(final_key, tLwe_key);

    // asking for generation of lookup table for X^a-1
    TGswSampleFFT *tmp = new_TGswSampleFFT(tgsw_params);
    TGswSampleFFT *tmp2 = new_TGswSampleFFT(tgsw_params);
    tGswFFTMulByXaiMinusOne(tmp,0,tmp2,tgsw_params);
    delete_TGswSampleFFT(tmp);
    delete_TGswSampleFFT(tmp2);

    const LweKey* user_sk = secret->lwe_key;
    const LweParams* user_in_out_params = params->in_out_params;
    const LweParams* in_out_params = params->in_out_params;

    // Program the wheel to value(s) after Bootstrapping
    const Torus32 mu_boot = modSwitchToTorus32(1, space_after_bs);


    if (VERBOSE) cout << "Import pixels, weights, biases, and labels from provided files." << endl;
    try
    {

        if (VERBOSE) cout << "[READ] Reading images (regardless of dimension) from " << FILE_TXT_IMG << endl;
        ifstream file_images(FILE_TXT_IMG);

        for (int32_t img=0; img<num_images; ++img)
            images[img] = new int32_t[num_neurons_in];

        int32_t filling_image = 0;
        int32_t image_count = 0;
        while(getline(file_images, line))
        {
            images[filling_image][image_count++] = stoi(line);
            if (image_count == num_neurons_in)
            {
                image_count = 0;
                filling_image++;
            }
        }
        file_images.close();


        if (VERBOSE) cout << "[READ] Reading weights from " << FILE_TXT_WEIGHTS << endl;
        ifstream file_weights(FILE_TXT_WEIGHTS);
        /// A weight polynomial in Z[X]/X^N+1
        IntPolynomial *poly_weights = new_IntPolynomial(SEC_PARAMS_N);

        num_neurons_current_layer_out = topology[0];
        for (l=0; l<num_wire_layers; ++l)
        {
            num_neurons_current_layer_in = num_neurons_current_layer_out;
            num_neurons_current_layer_out = topology[l+1];

            weights[l] = new int32_t*[num_neurons_current_layer_in];
            fft_weights[l] = new LagrangeHalfCPolynomial*[num_neurons_current_layer_out];
            for (int32_t i = 0; i<num_neurons_current_layer_in; ++i)
            {
                weights[l][i] = new int32_t[num_neurons_current_layer_out];
                for (int32_t j=0; j<num_neurons_current_layer_out; ++j)
                {
                    getline(file_weights, line);
                    el = stoi(line);
                    if (clamp_weights)
                    {
                        if (el < -threshold_weights)
                            el = -threshold_weights;
                        else if (el > threshold_weights)
                            el = threshold_weights;
                        // else, nothing as it holds that: -threshold_weights < el < threshold_weights
                    }
                    weights[l][i][j] = el;
                }
            }
            for (int32_t j = 0; j < num_neurons_current_layer_out; ++j)
            {
                fft_weights[l][j] = new_LagrangeHalfCPolynomial(SEC_PARAMS_N);

                for (int32_t i = 0; i<num_neurons_current_layer_in; ++i)
                {
                    poly_weights->coefs[SEC_PARAMS_N-(i+1)] = - weights[l][i][j];
                }
                for (int32_t i = num_neurons_current_layer_in; i<SEC_PARAMS_N ; ++i)
                {
                    poly_weights->coefs[SEC_PARAMS_N-(i+1)] = 0;
                }
                IntPolynomial_ifft(fft_weights[l][j], poly_weights);
            }
        }
        file_weights.close();
        delete_IntPolynomial(poly_weights);


        if (VERBOSE) cout << "[READ] Reading biases from " << FILE_TXT_BIASES << endl;
        ifstream file_biases(FILE_TXT_BIASES);

        num_neurons_current_layer_out = topology[0];
        for (l=0; l<num_wire_layers; ++l)
        {
            num_neurons_current_layer_in = num_neurons_current_layer_out;
            num_neurons_current_layer_out = topology[l+1];

            biases [l] = new int32_t [num_neurons_current_layer_out];
            for (int32_t j=0; j<num_neurons_current_layer_out; ++j)
            {
                getline(file_biases, line);
                el = stoi(line);
                if (clamp_biases)
                {
                    if (el < -threshold_biases)
                        el = -threshold_biases;
                    else if (el > threshold_biases)
                        el = threshold_biases;
                    // else, nothing as it holds that: -threshold_biases < el < threshold_biases
                }
                biases[l][j] = el;
            }
        }
        file_biases.close();


        if (VERBOSE) cout << "[READ] Reading labels from " << FILE_TXT_LABELS << endl;
        ifstream file_labels(FILE_TXT_LABELS);
        for (int32_t img=0; img<num_images; ++img)
        {
            getline(file_labels, line);
            labels[img] = stoi(line);
        }
        file_labels.close();

        if (VERBOSE) cout << "Import done.\n" << endl;
    }
    catch (const exception& e)
    {
        // This base class rule executes if try-block throws std::logic_error
        if (VERBOSE) cout << " Import problem: Are all the required files in the correct place?\n" << endl;
    }catch (...)
    {
        if (VERBOSE) cout << "Catch everything." << endl;
    }

    // Temporary variables and Pointers to existing arrays for convenience (they are always written, before read)
    Torus32 mu, phase;
    int32_t** weight_layer;
    int32_t *bias, *image;
    int32_t pixel, label;
    int32_t x, w, w0;
    bool notSameSign;

    // TODO µ Suggestion: delete/new/overwrite one array of length max if more than one hidden layer
    LweSample     *enc_image_LWE, *multi_sum, *bootstrapped;
    TLweSample    *enc_image_TLWE;
    TLweSampleFFT *fft_enc_image;

    clock_t bs_begin, bs_end, net_begin, net_end;
    double time_for_classification, time_for_bootstrapping;
    int32_t multi_sum_clear[num_neurons_hidden];
    int32_t output_clear   [num_neurons_out];

    // Once per id_proc/batch re-initialize counters, timings, ...
    int32_t max_score,max_score_clear,class_enc,class_clear,score,score_clear;
    int32_t batch_accum_img, batch_accum_count_errors, batch_accum_count_disagreements, batch_accum_count_disag_pro_clear, batch_accum_count_disag_pro_hom, batch_accum_count_wrong_bs, batch_accum_count_errors_with_failed_bs, batch_accum_count_disagreements_with_failed_bs;
    double batch_accum_time_classification, batch_accum_time_bootstrappings;
    bool failed_bs;


    // TODO: µ Suggestion for the final loop order: img, layer, in, out for efficiency/canonical reading and accessing
    // Processing
    pid_t    pids[num_processes];
    int32_t pipes[num_processes][2];
    for (int32_t id_proc=0; id_proc < num_processes; ++id_proc)
    {
        if (pipe(pipes[id_proc]));   // before fork!
        pid_t pid = fork();
        if (pid != 0)
        {
            // In parent process
            pids [id_proc] = pid;
            close(pipes[id_proc][1]);
        }
        else
        {
            // In child process
            close(pipes[id_proc][0]);

            // Once per id_proc/batch re-initialize counters,timings,...
            failed_bs = false;
            batch_accum_img = 0;
            batch_accum_time_classification = 0.0;
            batch_accum_time_bootstrappings= 0.0;
            batch_accum_count_errors = 0;
            batch_accum_count_disagreements = 0;
            batch_accum_count_disag_pro_clear = 0;
            batch_accum_count_disag_pro_hom = 0;
            batch_accum_count_wrong_bs = 0;
            batch_accum_count_errors_with_failed_bs = 0;
            batch_accum_count_disagreements_with_failed_bs = 0;
            // Results
            max_score = 0;
            class_enc = 0;
            score = 0;
            // Evaluation
            max_score_clear = 0;
            class_clear = 0;
            score_clear = 0;

            int32_t img = id_proc*batch;

            bool run = !got_sigterm;
            while (run)  //for (int32_t img = id_proc*slice; img < ( (id_proc+1)*slice) && (img < n_images); /*img*/ )
            {
                image = images[img];
                label = labels[img];
                ++img; //if (VERBOSE) cout << endl << "### Sample " << img << endl;
                ++batch_accum_img;

		failed_bs = false;

                // Update running condition
                run =!got_sigterm && (batch_accum_img < batch) && (img < num_images);
                //run =!got_sigterm && (batch_accum_img <= batch) && (img <= num_images);   // 10004 images
                // if (DEBUG) cerr<< id_proc<<img<<run<< " | " <<batch_accum_img<<&batch_accum_img<< endl;

                /// User generates encryption of pixels (using user_sk and user_in_out_params) and sends them to the cloud.
                {
                    // USER domain
                    num_neurons_current_layer_in = topology[0];
                    if (batched)
                    {
                        /// User generates batched encryptions of pixels: {p_i} -> TLWE({p_i})
                        enc_image_TLWE = new_TLweSample(tlwe_params);
                        // TODO: Could be a function encryptImage(image, num_neurons_current_layer_in, tlwe_key, tlwe_params);
                        /// This represents a torus polynomial modulo X^N+1.
                        TorusPolynomial *poly_image = new_TorusPolynomial(SEC_PARAMS_N);
                        for (int32_t i = 0; i < num_neurons_current_layer_in; i++)
                        {
                            pixel = image[i];
                            mu = modSwitchToTorus32(pixel, space_msg);
                            poly_image->coefsT[i+1] = mu;
                        }
                        if (noisyLWE)
                        {
                            tLweSymEncrypt(enc_image_TLWE, poly_image, alpha, tLwe_key);
                        }
                        else
                        {
                            tLweNoiselessTrivial(enc_image_TLWE, poly_image, tlwe_params);
                        }

                        fft_enc_image = new_TLweSampleFFT(tlwe_params);
                        tLweToFFTConvert(fft_enc_image, enc_image_TLWE, tlwe_params);
                        delete_TLweSample(enc_image_TLWE);
                        delete_TorusPolynomial(poly_image);
                    }
                    else
                    {
                        /// User generates individual (unbatched) encryptions of pixels: p_i -> LWE(p_i)
                        enc_image_LWE = new_LweSample_array(num_neurons_current_layer_in, user_in_out_params);
                        // TODO: Could be a function encryptImage(image, num_neurons_current_layer_in, user_sk, user_in_out_params);
                        for (int32_t i = 0; i < num_neurons_current_layer_in; ++i)
                        {
                            pixel = image[i];
                            mu = modSwitchToTorus32(pixel, space_msg);
                            if (noisyLWE)
                            {
                                lweSymEncrypt(enc_image_LWE + i, mu, alpha, user_sk);
                            }
                            else
                            {
                                lweNoiselessTrivial(enc_image_LWE + i, mu, user_in_out_params);
                            }
                        }
                    }
                } // USER domain


                // ########  FIRST LAYER(S)  ########


                // To be generic...
                num_neurons_current_layer_out = topology[0];
                net_begin = clock();
                for (l=0; l<num_wire_layers - 1 ; ++l)     // Note: num_wire_layers - 1 iterations; last one is special. Access weights from level l to l+1.
                {
                    /// Step 1: Compute multi_sum
                    num_neurons_current_layer_in = num_neurons_current_layer_out;
                    // assertion ( l + 1 <= num_wire_layers - 1 )
                    num_neurons_current_layer_out= topology[l+1];
                    // cerr << "layer " << l;  // NOTE: Debugging with cerr is better than cout, as the buffer is immediately flushed!

                    // both
                    multi_sum = new_LweSample_array(num_neurons_current_layer_out, in_out_params); // TODO delete or overwrite after use?
                    bootstrapped = new_LweSample_array(num_neurons_current_layer_out, final_params); // TODO overwrite bootstrapped?
                    // temp
                    bias = biases[l];
                    weight_layer = weights[l];
                    for (int32_t j=0; j<num_neurons_current_layer_out; ++j)
                    {
                        w0 = bias[j];
                        multi_sum_clear[j] = w0;
                        // assertion ( 0 <= |w0| < space_msg )
                        mu = modSwitchToTorus32(w0, space_msg);

                        lweNoiselessTrivial(multi_sum + j, mu, in_out_params);  // Encode bias in the clear

                        if(batched)
                        {
                            TLweSampleFFT *fft_multisum = new_TLweSampleFFT(tlwe_params);
                            tLweFFTClear(fft_multisum, tlwe_params);
                            tLweFFTAddMulRTo(fft_multisum, fft_weights[l][j], fft_enc_image, tlwe_params);

                            TLweSample *poly_multisum = new_TLweSample(tlwe_params);
                            tLweFromFFTConvert(poly_multisum, fft_multisum, tlwe_params);

                            LweSample *multisum = new_LweSample(final_params);
                            tLweExtractLweSample(multisum, poly_multisum, final_params, tlwe_params);

                            LweSample *keyswitched_multisum = new_LweSample(in_out_params);
                            lweKeySwitch(keyswitched_multisum, secret->cloud.bk->ks, multisum);
                            lweAddTo(multi_sum + j, keyswitched_multisum, in_out_params);

                            delete_TLweSampleFFT(fft_multisum);
                            delete_TLweSample(poly_multisum);
                            delete_LweSample(keyswitched_multisum);
                        }
                        else
                        {
                            for (int32_t i=0; i<num_neurons_current_layer_in; ++i)
                            {
                                x = image [i];  // clear input
                                w = weight_layer[i][j];  // w^dagger
                                lweAddMulTo(multi_sum + j, w, enc_image_LWE + i, in_out_params); // process encrypted input
                            }
                        } // finished step 1

                        {
                            // USER domain
                            for (int32_t i=0; i<num_neurons_current_layer_in; ++i)
                            {
                                x = image [i];  // clear input
                                w = weight_layer[i][j];  // w^dagger
                                multi_sum_clear[j] += x * w; // process clear input
                            }
                        } // USER domain


                        /// Step 2: Bootstrap multi_sum
                        bs_begin = clock();
                        /**
                         *  Bootstrapping results in each coordinate 'bootstrapped[j]' to contain an LweSample
                         *  of low-noise (= fresh LweEncryption) of 'mu_boot*phase(multi_sum[j])' (= per output neuron).
                         */
                        LweSample *LWEout = bootstrapped + j;
                        LweSample *LWEin  = multi_sum + j;
                        tfhe_bootstrap_woKS_FFT(LWEout, bs_key, mu_boot, LWEin, window_size);
                        bs_end = clock();

                        time_for_bootstrapping = (bs_end - bs_begin)*clocks2seconds;
                        batch_accum_time_bootstrappings += time_for_bootstrapping;
                        // if (VERBOSE) printf("[TIME] %.8lf [sec/bootstrapping]\n", time_for_bootstrapping);
                    } // for j < neurons_current_layer_out


                    // Free memory
                    if(batched)
                    {
                        delete_TLweSampleFFT(fft_enc_image);
                    }
                    else
                    {
                        delete_LweSample_array(num_neurons_in,     enc_image_LWE);
                    }
                    delete_LweSample_array(num_neurons_current_layer_out, multi_sum);  // TODO delete or overwrite after use?


                    // Evaluation
                    for (int32_t j=0; j<num_neurons_current_layer_out; ++j)
                    {
                        phase = lwePhase(bootstrapped + j, final_key);
                        notSameSign = multi_sum_clear[j]*t32tod(phase) < 0; // TODO adapt for non-binary case
                        if (notSameSign)
                        {
                            if (DEBUG) cerr << "\nwrong_bs@" << img << "  " << multi_sum_clear[j] << " vs. " << space_after_bs*t32tod(phase) << endl;
                            batch_accum_count_wrong_bs++;
                            failed_bs = true;
                        }
                    }
                } // for layer l < layers - 1

                // ########  LAST (SECOND) LAYER  ########
                max_score = threshold_scores;
                max_score_clear = threshold_scores;

                bias = biases[l];
                weight_layer = weights[l];
                l++;
                // assertion (l == num_wire_layers - 1);
                num_neurons_current_layer_in = num_neurons_current_layer_out;
                num_neurons_current_layer_out= topology[l]; // l == L = 2
                multi_sum = new_LweSample_array(num_neurons_current_layer_out, final_params); // TODO possibly overwrite storage
                for (int32_t j=0; j<num_neurons_current_layer_out; ++j)
                {
                    w0 = bias[j];
                    output_clear[j] = w0;
                    mu = modSwitchToTorus32(w0, space_after_bs);

                    lweNoiselessTrivial(multi_sum + j, mu, final_params);

                    //weight = weight_layer[l]; // assertion: l == L = 2
                    for (int32_t i=0; i<num_neurons_current_layer_in; ++i)
                    {
                        w = weight_layer[i][j];
                        // process encrypted input
                        lweAddMulTo(multi_sum + j, w, bootstrapped + i, final_params);
                        // process clear input
                        if (multi_sum_clear[i] < 0) // TODO adapt for non-binary case
                            output_clear[j] -= w;
                        else
                            output_clear[j] += w;

                    }
                    score = lwePhase(multi_sum + j, final_key);
                    if (score > max_score)
                    {
                        max_score = score;
                        class_enc = j;  //arg max
                    }

                    // Evaluation
                    score_clear = output_clear[j];
                    if (score_clear > max_score_clear)
                    {
                        max_score_clear = score_clear;
                        class_clear = j;
                    }
                } // finished last step
                net_end = clock();
                if (VERBOSE) printf("[%5d] avg. time of %d: %.8lf [sec/bootstrapping]", img, num_neurons_current_layer_in, batch_accum_time_bootstrappings/num_neurons_current_layer_in);

                time_for_classification = (net_end - net_begin)*clocks2seconds;
                batch_accum_time_classification += time_for_classification;
                if (VERBOSE) printf(", %.8lf [sec/classification]\n", time_for_classification);


                // Evaluation
                if (class_enc != label)
                {
                    if (DEBUG) printf("Counted an error @ Image %d\n", img);
                    batch_accum_count_errors++;
                    if (failed_bs)
                        batch_accum_count_errors_with_failed_bs++;
                }

                if (class_clear != class_enc)
                {
                    batch_accum_count_disagreements++;
                    if (failed_bs)
                        batch_accum_count_disagreements_with_failed_bs++;

                    if (class_clear == label)
                        batch_accum_count_disag_pro_clear++;
                    else if (class_enc == label)
                        batch_accum_count_disag_pro_hom++;
                }

                delete_LweSample_array(num_neurons_out,    multi_sum);
                delete_LweSample_array(num_neurons_current_layer_out, bootstrapped);
            } // run
            // Batch statistics update
            FILE* stream = fdopen(pipes[id_proc][1], "w");
            fprintf(stream, "%d,%d,%d,%d,%d,%d,%d,%d,%lf,%lf\n", batch_accum_img, batch_accum_count_errors, batch_accum_count_disagreements,batch_accum_count_disag_pro_clear,
                    batch_accum_count_disag_pro_hom, batch_accum_count_wrong_bs, batch_accum_count_errors_with_failed_bs,
                    batch_accum_count_disagreements_with_failed_bs, batch_accum_time_classification, batch_accum_time_bootstrappings);
            /*
                fprintf(stream, "%d,%d,%d,%d,%d,%d,%d,%d,%lf,%lf\n", count_images, count_errors, count_disagreements, count_disag_pro_clear, count_disag_pro_hom, count_wrong_bs,
                    count_errors_with_failed_bs, count_disagreements_with_failed_bs, total_time_network, total_time_bootstrappings);
            fclose(stream);*/

            fclose(stream);
            exit(0);
        } // else ( assertion(pid == 0))
    } // for id_proc


    // Wait for all processes to finish before accumulating results
    for (auto pid : pids) waitpid(pid, 0, 0);


    // Accumulate counters & Timings of all batches
    for (int32_t id_proc=0; id_proc<num_processes; ++id_proc)
    {
        FILE* stream = fdopen(pipes[id_proc][0], "r");
        if (fscanf(stream, "%d,%d,%d,%d,%d,%d,%d,%d,%lf,%lf\n", &batch_accum_img, &batch_accum_count_errors, &batch_accum_count_disagreements,
               &batch_accum_count_disag_pro_clear, &batch_accum_count_disag_pro_hom, &batch_accum_count_wrong_bs, &batch_accum_count_errors_with_failed_bs,
		   &batch_accum_count_disagreements_with_failed_bs, &batch_accum_time_classification, &batch_accum_time_bootstrappings));
        fclose(stream);

        // Global statistics update
        count_images += batch_accum_img;
        count_errors += batch_accum_count_errors;
        count_disagreements += batch_accum_count_disagreements;
        count_disag_pro_clear += batch_accum_count_disag_pro_clear;
        count_disag_pro_hom += batch_accum_count_disag_pro_hom;
        count_wrong_bs += batch_accum_count_wrong_bs;
        count_errors_with_failed_bs += batch_accum_count_errors_with_failed_bs;
        count_disagreements_with_failed_bs += batch_accum_count_disagreements_with_failed_bs;
        total_time_classifications += batch_accum_time_classification;
        total_time_bootstrappings += batch_accum_time_bootstrappings;
    }


    // Compute Statistics
    avg_img = 1. / count_images;
    avg_img_bs = avg_img * avg_bs;
    avg_time_per_classification = total_time_classifications*avg_img;
    avg_time_per_bootstrapping  = total_time_bootstrappings*avg_img_bs;
    rel_error_percent = count_errors*avg_img*100;
    rel_time_bs_percent = total_time_bootstrappings/total_time_classifications*100;


    writeStatistics();


    // Free memory
    delete_gate_bootstrapping_secret_keyset(secret, window_size);
    delete_gate_bootstrapping_parameters(params);

    // TODO: delete fft_weights (Tensor(fft_weights,num_wire_layers, topology))
    deleteTensor(weights,num_wire_layers, topology);
    deleteMatrix(biases, num_wire_layers);
    deleteMatrix(images, num_images);
    delete[] labels;

    return 0;
}


void my_signal_handler(int32_t sig)
{
    cerr << "*** Received SIGint32_t *** (Stopping & finishing threads gracefully...)\n"; //  << sig
    got_sigterm = 1;
    return;
}


void writeStatistics()
{
    if (MODE_STATS)
    {
        ostringstream stats;
        stats << "\n[STAT] STATISTICS of inputs 1, 2, ..., " << count_images<< " of " << num_images << endl;
        stats << "Errors: " << count_errors << " / " << count_images << " (" << rel_error_percent << " %) 'Misclassifications'" << endl;
        stats << "Disagreements: " << count_disagreements;
        stats << " (pro-clear/pro-hom: " << count_disag_pro_clear << " / " << count_disag_pro_hom << ")" << endl;
        stats << "Wrong bootstrappings: " << count_wrong_bs << endl;
        stats << "Errors with failed bootstrapping: " << count_errors_with_failed_bs << endl;
        stats << "Disagreements with failed bootstrapping: " << count_disagreements_with_failed_bs << endl;
        stats << "Avg. time for the evaluation of the network (seconds): " << avg_time_per_classification << endl;
        stats << "Avg. time per bootstrapping (seconds): " << avg_time_per_bootstrapping << endl;
        stats << "[PROF] Bootstrapping took: " << rel_time_bs_percent << " % of the time!" << endl;

        // To Console
        string str = stats.str();
        cout << str;

        ofstream stat(FILE_STATISTICS);
        stat << str;
        stat.close();
        cout << "\n[SAVE] Wrote statistics to file:   " << FILE_STATISTICS << endl;
    }

    if (MODE_LATEX)
    {
        ostringstream latex;
        latex << "%\\input{"<<FILE_LATEX<<"}" << endl;

        latex << "% Experiments detailed" << endl;
        latex << "\\newcommand{\\EXPnumBS}{$"      <<total_num_hidden_neurons<<"$}" << endl;
        latex << "\\newcommand{\\EXPbsEXACT}{$"    <<avg_time_per_bootstrapping<<"$\\ [sec/bootstrapping]}" << endl;
        latex << "\\newcommand{\\EXPtimeEXACT}{$"  <<avg_time_per_classification<<"$\\ [sec/classification]}" << endl;

        latex << "\\newcommand{\\EXPnumERRabs}{$"  <<count_errors<<"$}" << endl;
        latex << "\\newcommand{\\EXPnumERRper}{$"  <<rel_error_percent<<"\\ \\%$}" << endl;
        latex << "\\newcommand{\\EXPwrongBSabs}{$" <<count_wrong_bs<<"$}" << endl;
        latex << "\\newcommand{\\EXPwrongDISabs}{$"<<count_disagreements_with_failed_bs<<"$}" << endl;
        latex << "\\newcommand{\\EXPdis}{$"        <<count_disagreements<<"$}" << endl;
        latex << "\\newcommand{\\EXPclear}{$"      <<count_disag_pro_clear<<"$}" << endl;
        latex << "\\newcommand{\\EXPhom}{$"        <<count_disag_pro_hom<<"$}" << endl << endl;

        latex << "\\begin{Verbatim}[frame=single,numbers=left,commandchars=+\\[\\]%" << endl;
        latex << "]" << endl;
        latex << "### Classified samples: +EXPtestset" << endl;
        latex << "Time per bootstrapping: +EXPbsEXACT" << endl;
        latex << "Errors: +EXPnumERRabs / +EXPtestset (+EXPnumERRper)" << endl;
        latex << "Disagreements: +EXPdis" << endl;
        latex << "(pro-clear/pro-hom: +EXPclear / +EXPhom)" << endl;
        latex << "Wrong bootstrappings: +EXPwrongBSabs" << endl;
        latex << "Disagreements with wrong bootstrapping: +EXPwrongDISabs" << endl;
        latex << "Avg. time for the evaluation of the network: +EXPtimeEXACT" << endl;
        latex << "\\end{Verbatim}" << endl;

        ofstream lat(FILE_LATEX);
        lat << latex.str();
        lat.close();
        cout << "[SAVE] Wrote LaTeX_result to file: " << FILE_LATEX << endl;
    }
}


void deleteMatrix(int32_t** matrix, int32_t dim_mat)
{
    for (int32_t i=0; i<dim_mat; ++i)
    {
        delete[] matrix[i];
    }
    delete[] matrix;
}


void deleteTensor(int32_t*** tensor, int32_t dim_tensor, const int32_t* dim_vec)
{
    int32_t** matrix;
    int32_t dim_mat;
    for (int32_t i=0; i<dim_tensor; ++i)
    {
        matrix =  tensor[i];
        dim_mat = dim_vec[i];
        deleteMatrix(matrix, dim_mat);
    }
    delete[] tensor;
}


/**
 * main
 * \brief gives argc, argv, runs fully homomorphic evaluation of Neural Net and writes results.
 */
int32_t main(int32_t argc, char *argv[])
{
    time_t timestamp = time(nullptr);

    ostringstream test;
    test << asctime(localtime(&timestamp));
    if (!batched) cout << "un";
        test << "batchedMultiProcessing Testrun with call: " << argv[0] << endl;

    //string output = test.str();

    // The signal handler needs to be registered in the main.
    if (signal((int) SIGINT, my_signal_handler) == SIG_ERR)
    {
        cerr << "Registering my_signal_handler failed!"<< endl;
    }


    if (VERBOSE)
    {
        cout << test.str();
        cout << "Starting experiment to classify " << num_images;
        if (!noisyLWE) cout << " noiseless";
        cout << " encrypted MNIST images." << endl << "(";
        cout << "Execution with parameters... alpha = " << alpha << ", number of processes: " << num_processes << endl;
        // if (!VERBOSE) cout << "Print32_t unused variables: "<< torus_slices_upper_half << torus_slices_all << torus_slices_allowed << endl;
        cout << "\n*** Graceful exit on CTRL+C, exit on  CTRL+Z. ***\n\n";
    }

    processNeuralNet();

    // Some output
    test << avg_time_per_bootstrapping  << " [sec/bootstrapping]  ... Avg. time per bootstrapping (seconds)\n";
    test << avg_time_per_classification << " [sec/classification] ... Avg. time for one classification (seconds)\n\n";

    ofstream testf(FILE_TEST, ios_base::app);
    testf << test.str();
    testf.close();
    cout << "[SAVE] Appended timestamped test results: " << FILE_TEST << endl;

    return 0;
}
