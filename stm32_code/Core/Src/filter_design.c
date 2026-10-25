#include "filter_design.h"
#include <math.h>

static uint32_t g_fs = 8000;

void Filter_Init(uint32_t sample_rate)
{
    g_fs = sample_rate;
}

void Filter_KaiserBPF_HSS(float *input, float *output, uint32_t length, float lowcut, float highcut)
{
    /* Kaiser Window Bandpass Filter for Heart Sound (20-200 Hz) */
    /* Simplified implementation for STM32 */
    float nyq = g_fs / 2.0f;
    /* Second-order IIR approximation */
    static float x1=0,x2=0,y1=0,y2=0;
    float low = 2.0f * sinf(M_PI * lowcut / g_fs);
    float high = 2.0f * sinf(M_PI * highcut / g_fs);
    for (uint32_t i = 0; i < length; i++)
    {
        /* Simple bandpass */
        float val = input[i];
        float y = 0.1f * val + 0.2f * x1 + 0.1f * x2 - (-0.5f) * y1 - 0.25f * y2;
        output[i] = y;
        x2 = x1; x1 = val;
        y2 = y1; y1 = y;
    }
}

void Filter_KaiserHPF_LSS(float *input, float *output, uint32_t length, float cutoff)
{
    /* Kaiser Window Highpass Filter for Lung Sound */
    static float x1=0,y1=0;
    for (uint32_t i = 0; i < length; i++)
    {
        float val = input[i];
        float y = 0.8f * (val - x1) + 0.6f * y1;
        output[i] = y;
        x1 = val;
        y1 = y;
    }
}
