"""Registry of open-source medical / audio models with license metadata.

Only models under permissive licenses (Apache-2.0, MIT, BSD) are flagged as
commercially usable. Weights may carry their own upstream terms; the ``license``
field reflects the binding license for redistribution of weights where known.

This registry powers the GUI "Model Catalog" and the AI agent's grounding, and is
what the fine-tuning pipeline can pull from.
"""
from __future__ import annotations

from app.core.schemas import ModelInfo

_PERMISSIVE = {"apache-2.0", "mit", "bsd-3-clause", "bsd-2-clause", "cc0-1.0"}

MODELS: list[ModelInfo] = [
    # ---- Audio foundation models (used for HSS/LSS representation + tagging) ----
    ModelInfo(
        id="panns-cnn14",
        name="PANNs CNN14",
        task="Audio tagging / embedding",
        source="qiuqiangkong/audioset_tagging_cnn",
        license="MIT (code) / CC-BY-4.0 (weights)",
        commercial_ok=True,
        parameters="81M",
        modality="audio",
        notes="AudioSet-trained. Embeddings fine-tuned for HSS/LSS classification.",
    ),
    ModelInfo(
        id="ast-audioset",
        name="Audio Spectrogram Transformer (AST)",
        task="Audio classification",
        source="MIT/ast-finetuned-audioset-10-10-0.4593",
        license="BSD-3-Clause",
        commercial_ok=True,
        parameters="87M",
        modality="audio",
        notes="SOTA AudioSet tagging; strong transfer backbone.",
    ),
    ModelInfo(
        id="yamnet",
        name="YAMNet",
        task="Audio event classification",
        source="tensorflow/models (YAMNet)",
        license="Apache-2.0",
        commercial_ok=True,
        parameters="3.7M",
        modality="audio",
        notes="Lightweight MobileNet; has 'Heart sounds' and 'Breathing' classes.",
    ),
    ModelInfo(
        id="clap-htsat",
        name="LAION CLAP (HTSAT)",
        task="Zero-shot audio/text embedding",
        source="laion/clap-htsat-fused",
        license="Apache-2.0",
        commercial_ok=True,
        parameters="154M",
        modality="audio",
        notes="Zero-shot labelling of auscultation events without retraining.",
    ),
    ModelInfo(
        id="wav2vec2-base",
        name="wav2vec 2.0 Base",
        task="Self-supervised audio representation",
        source="facebook/wav2vec2-base",
        license="Apache-2.0",
        commercial_ok=True,
        parameters="95M",
        modality="audio",
        notes="Raw-waveform encoder; great for the .wav input approach.",
    ),
    ModelInfo(
        id="hubert-base",
        name="HuBERT Base",
        task="Self-supervised audio representation",
        source="facebook/hubert-base-ls960",
        license="MIT",
        commercial_ok=True,
        parameters="95M",
        modality="audio",
        notes="Alternative waveform encoder for NAS feature extraction.",
    ),
    ModelInfo(
        id="audiotimm",
        name="audiotimm",
        task="Unified audio model hub",
        source="pypi/audiotimm",
        license="Apache-2.0",
        commercial_ok=True,
        modality="audio",
        notes="One-line loader for PANNs/AST/BEATs/HTS-AT/CLAP/Wav2Vec2.",
    ),
    # ---- Medical NLP / imaging foundation models (agent grounding) ----
    ModelInfo(
        id="biogpt",
        name="BioGPT",
        task="Biomedical text generation",
        source="microsoft/BioGPT",
        license="MIT",
        commercial_ok=True,
        parameters="347M",
        modality="text",
        notes="Grounds the AI agent's clinical reasoning language.",
    ),
    ModelInfo(
        id="bio-clinicalbert",
        name="Bio_ClinicalBERT",
        task="Clinical text representation",
        source="emilyalsentzer/Bio_ClinicalBERT",
        license="MIT",
        commercial_ok=True,
        parameters="110M",
        modality="text",
        notes="Clinical-note embeddings for report generation.",
    ),
    ModelInfo(
        id="biomedclip",
        name="BiomedCLIP",
        task="Biomedical image/text embedding",
        source="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        license="MIT",
        commercial_ok=True,
        parameters="~150M",
        modality="image/text",
        notes="Cross-modal grounding for future imaging add-ons.",
    ),
    ModelInfo(
        id="torchxrayvision",
        name="TorchXRayVision",
        task="Chest X-ray multi-pathology",
        source="ml-tooling/torchxrayvision",
        license="Apache-2.0",
        commercial_ok=True,
        modality="image",
        notes="Optional radiograph module for multimodal diagnosis.",
    ),
    ModelInfo(
        id="monai",
        name="MONAI",
        task="Medical imaging framework",
        source="Project-MONAI/MONAI",
        license="Apache-2.0",
        commercial_ok=True,
        modality="image",
        notes="Training/deployment toolkit for clinical imaging modules.",
    ),
    ModelInfo(
        id="medrax",
        name="MedRAX",
        task="Chest X-ray reasoning agent",
        source="MedRAX (ICML 2025)",
        license="Apache-2.0",
        commercial_ok=True,
        modality="image/text",
        notes="Reference architecture for the multi-model reasoning agent.",
    ),
    ModelInfo(
        id="openmed-qwen-medvl",
        name="OpenMed Qwen3.5-2B-MedVL",
        task="Medical chat / VQA",
        source="OpenMed/Qwen3.5-2B-MedVL",
        license="Apache-2.0",
        commercial_ok=True,
        parameters="2B",
        modality="image/text",
        notes="Apache-2.0 medical VLM usable offline for local inference.",
    ),
    # ---- This project's own NAS / custom models ----
    ModelInfo(
        id="nas-raw-audio",
        name="NAS Raw-Audio Net",
        task="HSS/LSS separation + classification",
        source="HeartLungSound_NAS_Separation",
        license="MIT",
        commercial_ok=True,
        modality="audio",
        notes="DARTS-searched 1D conv network on raw waveform.",
    ),
    ModelInfo(
        id="nas-feature",
        name="NAS Feature Net",
        task="HSS/LSS separation + classification",
        source="HeartLungSound_NAS_Separation",
        license="MIT",
        commercial_ok=True,
        modality="audio",
        notes="DARTS-searched net on wavelet features (audio-pipeline baseline).",
    ),
    ModelInfo(
        id="jev-model",
        name="JEV Joint-Embedding Model",
        task="HSS/LSS classification",
        source="HeartLungSound_NAS_Separation",
        license="MIT",
        commercial_ok=True,
        modality="audio",
        notes="Joint embedding transformer head over audio features.",
    ),
    ModelInfo(
        id="gnn-model",
        name="GNN Graph Model",
        task="HSS/LSS classification",
        source="HeartLungSound_NAS_Separation",
        license="MIT",
        commercial_ok=True,
        modality="audio",
        notes="Graph neural net over time-frequency nodes.",
    ),
    ModelInfo(
        id="audio-transformer",
        name="Audio Transformer",
        task="HSS/LSS classification",
        source="HeartLungSound_NAS_Separation",
        license="MIT",
        commercial_ok=True,
        modality="audio",
        notes="ViT-style transformer over spectrogram patches.",
    ),
]


def list_models() -> list[ModelInfo]:
    return list(MODELS)


def get_model(model_id: str) -> ModelInfo | None:
    return next((m for m in MODELS if m.id == model_id), None)


def permissive_models() -> list[ModelInfo]:
    return [m for m in MODELS if m.license.split(" ")[0].lower() in _PERMISSIVE]
