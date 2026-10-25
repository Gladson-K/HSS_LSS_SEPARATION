#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

#include "stm32f4xx_hal.h"

extern volatile uint8_t g_audio_ready_flag;

void Error_Handler(void);

#ifdef __cplusplus
}
#endif

#endif
