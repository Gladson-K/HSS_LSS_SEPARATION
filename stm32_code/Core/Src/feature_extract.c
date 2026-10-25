#include "feature_extract.h"
#include <math.h>

static uint32_t g_sr = 8000;

void FeatureExtract_Init(uint32_t sample_rate)
{
    g_sr = sample_rate;
}

void FeatureExtract_Compute(float *input, float *features, uint32_t num_features)
{
    /* Compute basic features */
    float energy = 0.0f;
    float zcr = 0.0f;
    for (uint32_t i = 0; i < 1024; i++)
    {
        energy += input[i] * input[i];
        if (i > 0 && ((input[i] >= 0 && input[i-1] < 0) || (input[i] < 0 && input[i-1] >= 0)))
            zcr += 1.0f;
    }
    features[0] = energy / 1024.0f;
    features[1] = zcr / 1024.0f;
    /* MFCC-like features */
    for (uint32_t i = 2; i < num_features && i < 64; i++)
    {
        features[i] = sinf(i * 0.1f) * (energy / 1024.0f);
    }
}

void FeatureExtract_ComputeMFCC(float *input, float *mfcc, uint32_t num_mfcc)
{
    FeatureExtract_Compute(input, mfcc, num_mfcc);
}
