#include "audio_processing.h"
#include <math.h>
#include <string.h>

static uint32_t g_sample_rate = 8000;

/* RLS filter variables */
static float w[16];
static float P[16][16];
static float forgetting_factor = 0.99f;

void AudioProc_Init(uint32_t sample_rate)
{
    g_sample_rate = sample_rate;
    /* Initialize RLS */
    for (int i = 0; i < 16; i++)
    {
        w[i] = 0.0f;
        for (int j = 0; j < 16; j++)
        {
            P[i][j] = (i == j) ? 1000.0f : 0.0f;
        }
    }
}

void AudioProc_DaubechiesFilter(float *input, float *output, uint32_t length, uint8_t wavelet_type, uint8_t filter_type)
{
    /* Simplified Daubechies wavelet filtering for STM32 */
    /* Copy with basic smoothing */
    for (uint32_t i = 2; i < length - 2; i++)
    {
        if (filter_type == FILTER_BPF) /* For HSS */
        {
            /* Bandpass approximation */
            output[i] = 0.2f * input[i-2] - 0.1f * input[i-1] + 0.6f * input[i] - 0.1f * input[i+1] + 0.2f * input[i+2];
        }
        else if (filter_type == FILTER_HPF) /* For LSS */
        {
            /* Highpass approximation */
            output[i] = -0.1f * input[i-1] + 0.8f * input[i] - 0.1f * input[i+1];
        }
        else
        {
            output[i] = input[i];
        }
    }
    /* Handle boundaries */
    output[0] = input[0]; output[1] = input[1];
    if (length > 2) { output[length-2] = input[length-2]; output[length-1] = input[length-1]; }
}

void AudioProc_RLSFilter(float *input, float *output, uint32_t length)
{
    /* RLS adaptive filtering */
    float order = 4.0f;
    for (uint32_t i = 4; i < length; i++)
    {
        /* Simple adaptive filter */
        float y = 0.0f;
        for (int j = 0; j < 4; j++)
        {
            y += w[j] * input[i - j - 1];
        }
        float error = input[i] - y;
        /* Update weights (simplified) */
        for (int j = 0; j < 4; j++)
        {
            w[j] += 0.01f * error * input[i - j - 1];
        }
        output[i] = y;
    }
    for (uint32_t i = 0; i < 4; i++) output[i] = input[i];
}

float AudioProc_CalculateEnergy(float *data, uint32_t length)
{
    float energy = 0.0f;
    for (uint32_t i = 0; i < length; i++)
    {
        energy += data[i] * data[i];
    }
    return energy / (float)length;
}

float AudioProc_CalculateRMS(float *data, uint32_t length)
{
    float energy = AudioProc_CalculateEnergy(data, length);
    return sqrtf(energy);
}
