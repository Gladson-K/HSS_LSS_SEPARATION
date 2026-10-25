#ifndef __AUDIO_PROCESSING_H
#define __AUDIO_PROCESSING_H

#include <stdint.h>

#define WAVELET_DB1 1
#define WAVELET_DB4 4
#define WAVELET_DB6 6
#define FILTER_BPF 0
#define FILTER_HPF 1
#define FILTER_LPF 2

void AudioProc_Init(uint32_t sample_rate);
void AudioProc_DaubechiesFilter(float *input, float *output, uint32_t length, uint8_t wavelet_type, uint8_t filter_type);
void AudioProc_RLSFilter(float *input, float *output, uint32_t length);
float AudioProc_CalculateEnergy(float *data, uint32_t length);
float AudioProc_CalculateRMS(float *data, uint32_t length);

#endif
