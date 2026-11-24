# Intended use and claim boundary

**Status: research prototype. Not a medical device. Not for clinical diagnosis or treatment decisions.**

This repository contains signal-separation and triage experiments for heart and lung sounds. Some presentation material uses the phrase "medical diagnosis" and discusses disease probability. Those phrases describe the research goal; they are not a claim that the software is a certified or validated medical product.

## Binding statements

1. The diagnostics helper in `platform/backend/app/core/diagnostics.py` produces a rule-based, research-grade triage label with a confidence score. It is educational and screening support only. It does not diagnose, and its "critical" label is not a clinical alert.
2. No output from this repository may be used as the sole basis for a clinical decision, to delay care, or to reassure a patient.
3. Do not market, label, or deploy this work as a medical device, diagnostic aid, or clinical decision-support system without a documented regulatory pathway, qualified clinical review, and conformity assessment for the target market.
4. Under the EU AI Act as amended in 2026, AI systems that are safety components of regulated products (for example medical devices) move to the high-risk regime on 2 August 2028; stand-alone Annex III high-risk systems move to 2 December 2027. Whether a future version falls into either category must be assessed before launch, not assumed.
5. Auscultation recordings may be special-category personal data. Keep them local, document provenance and consent, and do not upload them to model providers, hosted APIs, issues, or public datasets.

## Required before any product claim

- Written scope and intended-use statement reviewed by a qualified clinician and regulatory counsel.
- Data provenance, licence, and consent record for every recording used in training or evaluation.
- Model card, AI impact assessment, human-oversight design, logging, accuracy and robustness evidence.
- Risk management file and, where applicable, medical-device conformity assessment.

See `../_governance/AI-REGULATIONS.md` and `../_governance/RISK-REGISTER.md`.
