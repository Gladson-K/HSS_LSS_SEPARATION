# STM32 Compatible C Code for Heart-Lung Sound Analysis

Implementation for STM32 microcontroller to perform real-time Heart and Lung sound analysis using stethoscope as input.

## Features
- Stethoscope ADC input acquisition
- Preprocessing: Kaiser Window BPF/HPF, Digital filtering
- Heart/Lung sound separation
- Signal analysis and classification
- Real-time processing optimized for STM32

## Hardware Requirements
- STM32F4/F7/H7 series (recommended)
- ADC for microphone/stethoscope input
- DAC/Serial output for results
- Sufficient RAM/Flash for processing

## Files
- `main.c` - Main application
- `audio_processing.c/h` - Preprocessing and separation algorithms
- `filter_design.c/h` - FIR/IIR filter implementations (Kaiser window)
- `feature_extract.c/h` - Feature extraction
- `classification.c/h` - Classification logic
- `stm32_config.c/h` - Hardware configuration

## Algorithm (from research)
1. Preprocess with Kaiser Window BPF (HSS) / HPF (LSS)
2. Daubechies Wavelet decomposition
3. RLS adaptive filtering for separation
4. Feature extraction and analysis
5. Classification with accuracy estimation
