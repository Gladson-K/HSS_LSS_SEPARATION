#ifndef __CLASSIFICATION_H
#define __CLASSIFICATION_H

#include <stdint.h>

void Classification_Init(void);
void Classification_Predict(float *features, uint8_t *result, float *confidence);

#endif
