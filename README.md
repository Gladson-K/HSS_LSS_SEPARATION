# Heart-Lung Sound Separation using NAS (Neural Architecture Search)

This project focuses on purely separating Heart (HSS) and Lung (LSS) sounds from stethoscope recordings using Neural Architecture Search for two input approaches:
1. Direct .wav audio waveform as input (raw audio pipeline)
2. Feature-based input (wavelet/MFCC based features) - trained on top of audio pipeline baseline approach

## Approach
- Based on research from literature review and project PPT
- Uses Kaiser Window BPF/HPF, Daubechies Wavelet Filter, RLS as per algorithm
- NAS to find optimal architecture for both approaches
- Best models trained on full dataset
- STM32 compatible C code for embedded deployment

## Datasets
Multiple datasets provided in the original project including Audio Files, DATA, AS_New, MVP_New, N_New etc.

## Structure
- `preprocessing/`: Signal processing pipelines (filtering, wavelets, features)
- `baselines/`: Baseline models and audio pipeline baseline (as submodule)
- `nas_search/`: NAS implementations for both approaches
- `models/`: Trained best models
- `training/`: Training scripts
- `stm32_code/`: STM32 compatible C implementation
- `docs/`: Research materials
