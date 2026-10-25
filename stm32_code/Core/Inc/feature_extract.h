#ifndef __FEATURE_EXTRACT_H
#define __FEATURE_EXTRACT_H

#include <stdint.h>

void FeatureExtract_Init(uint32_t sample_rate);
void FeatureExtract_Compute(float *input, float *features, uint32_t num_features);
void FeatureExtract_ComputeMFCC(float *input, float *mfcc, uint32_t num_mfcc);

#endif
