/**
  * @file           : main.c
  * @brief          : Main program body for Heart-Lung Sound Separation
  */
#include "main.h"
#include "audio_processing.h"
#include "filter_design.h"
#include "feature_extract.h"
#include "classification.h"

#include <stdio.h>
#include <string.h>
#include <math.h>

ADC_HandleTypeDef hadc1;
DMA_HandleTypeDef hdma_adc1;
TIM_HandleTypeDef htim2;
UART_HandleTypeDef huart2;

#define AUDIO_BUFFER_SIZE 1024
#define SAMPLE_RATE 8000

static uint16_t adc_buffer[AUDIO_BUFFER_SIZE];
static float audio_input[AUDIO_BUFFER_SIZE];
static float hss_output[AUDIO_BUFFER_SIZE];
static float lss_output[AUDIO_BUFFER_SIZE];

volatile uint8_t g_audio_ready_flag = 0;

void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_DMA_Init(void);
static void MX_ADC1_Init(void);
static void MX_TIM2_Init(void);
static void MX_USART2_UART_Init(void);

int main(void)
{
  HAL_Init();
  SystemClock_Config();
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_ADC1_Init();
  MX_TIM2_Init();
  MX_USART2_UART_Init();
  
  AudioProc_Init(SAMPLE_RATE);
  Filter_Init(SAMPLE_RATE);
  FeatureExtract_Init(SAMPLE_RATE);
  Classification_Init();
  
  printf("Heart-Lung Sound Separation System Started\r\n");
  
  HAL_ADC_Start_DMA(&hadc1, (uint32_t*)adc_buffer, AUDIO_BUFFER_SIZE);
  HAL_TIM_Base_Start(&htim2);
  
  while (1)
  {
    if (g_audio_ready_flag)
    {
      for (int i = 0; i < AUDIO_BUFFER_SIZE; i++)
      {
        audio_input[i] = ((float)adc_buffer[i] - 2048.0f) / 2048.0f;
      }
      
      /* Preprocessing and separation per algorithm */
      Filter_KaiserBPF_HSS(audio_input, hss_output, AUDIO_BUFFER_SIZE, 20.0f, 200.0f);
      Filter_KaiserHPF_LSS(audio_input, lss_output, AUDIO_BUFFER_SIZE, 150.0f);
      
      AudioProc_DaubechiesFilter(hss_output, hss_output, AUDIO_BUFFER_SIZE, 4, 0);
      AudioProc_DaubechiesFilter(lss_output, lss_output, AUDIO_BUFFER_SIZE, 6, 1);
      
      AudioProc_RLSFilter(hss_output, hss_output, AUDIO_BUFFER_SIZE);
      AudioProc_RLSFilter(lss_output, lss_output, AUDIO_BUFFER_SIZE);
      
      float features[64];
      FeatureExtract_Compute(audio_input, features, 64);
      
      uint8_t classification_result = 0;
      float confidence = 0.0f;
      Classification_Predict(features, &classification_result, &confidence);
      
      printf("Class:%d Conf:%.1f%% HSS:%.3f LSS:%.3f\r\n",
             classification_result, confidence * 100.0f,
             AudioProc_CalculateEnergy(hss_output, AUDIO_BUFFER_SIZE),
             AudioProc_CalculateEnergy(lss_output, AUDIO_BUFFER_SIZE));
      
      g_audio_ready_flag = 0;
    }
  }
}
