#ifndef __FILTER_DESIGN_H
#define __FILTER_DESIGN_H

#include <stdint.h>

void Filter_Init(uint32_t sample_rate);
void Filter_KaiserBPF_HSS(float *input, float *output, uint32_t length, float lowcut, float highcut);
void Filter_KaiserHPF_LSS(float *input, float *output, uint32_t length, float cutoff);

#endif
