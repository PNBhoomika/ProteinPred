# Title

**ProtFunc: A Hierarchy-Aware, Homology-Complementary Deep Learning System for Protein Function Prediction from DNA Sequences Using Protein Language Models**

# Problem Statement

Sequencing technology has produced hundreds of millions of protein sequences, but only a tiny fraction (well under 1%) have experimentally verified functions. Laboratory validation is slow and expensive, so the annotation gap keeps widening.

Computational methods try to close this gap, but they face three persistent problems:

1. **Homology-based methods such as BLAST fail on remote proteins.** They work well when a close, annotated relative exists, but their accuracy drops sharply when sequence identity to known proteins is low.
2. **Flat classifiers ignore the structure of function.** Gene Ontology (GO) terms form a directed acyclic graph, where a child term implies all its ancestors. Models that treat terms as independent labels produce inconsistent predictions (a specific term scored higher than its own parent) and perform poorly on rare, specific terms with few training examples.
3. **Predictions are often uncalibrated and unusable in practice.** Confidence scores rarely reflect true probabilities, evaluation often suffers from sequence leakage between train and test sets, and most research models are never exposed through a usable tool.

This project addresses these gaps by building an end-to-end system that:

- accepts a raw **DNA sequence**, identifies the open reading frame and translates it into a protein;
- generates a representation with the pretrained protein language model **ESM-2**;
- predicts **GO terms across Molecular Function (MF), Biological Process (BP) and Cellular Component (CC)** with calibrated confidence scores;
- guarantees **hierarchical consistency**, so that the score of a child term never exceeds that of its parent;
- is evaluated rigorously against a **BLAST/DIAMOND homology baseline** on a leakage-controlled split, with results broken down by rare terms and by sequence-identity level;
- is served through a **deployed, tested web application**.

# Central Research Question

> Does hierarchy-aware training on protein language model embeddings improve the prediction of rare and specific GO terms, and of proteins with low sequence identity to known proteins, compared with flat classifiers and homology-based methods?

# Hypotheses

- **H1:** Embeddings from a pretrained protein language model, with a multi-label classifier, outperform BLAST on proteins with low sequence identity to the training set.
- **H2:** Enforcing the GO hierarchy (during training and/or post-processing) improves F-max and the prediction of rare, specific terms over a flat classifier.
- **H3:** Combining model predictions with homology evidence outperforms either source alone.
- **H4:** Calibration (e.g. temperature scaling) makes confidence scores reliable without reducing ranking performance.

# Objectives

1. Build a data pipeline from CAFA 5: parse sequences, GO annotations and the ontology; propagate labels up the hierarchy; select the most frequent terms per aspect.
2. Create a leakage-controlled train/validation/test split by clustering sequences by identity.
3. Extract and cache ESM-2 embeddings for all proteins.
4. Train and compare a linear baseline, an MLP head, a hierarchy-aware model, and an ensemble or hybrid with homology features.
5. Implement a DIAMOND/BLAST homology-transfer baseline using only training sequences.
6. Evaluate with CAFA-standard metrics (F-max, S-min, AUPR), reported per aspect, per term frequency and per sequence-identity bin.
7. Calibrate confidence scores and report reliability diagrams.
8. Build a DNA→ORF→protein module with input validation.
9. Deploy the pipeline as a web application with an API, tests and CI.

# Project Description

**Overview:** ProtFunc is a full-stack ML application. It converts DNA to protein, embeds the protein with ESM-2, and predicts GO terms with a hierarchy-aware multi-label classifier trained on the CAFA 5 dataset. A hybrid component draws on homology evidence from similar known proteins, and a calibration step makes each confidence score interpretable.

**Pipeline:**

```
DNA → ORF detection & translation → ESM-2 embedding
    → multi-label classifier (MF / BP / CC heads)
    → hierarchy enforcement + calibration
    → GO terms with confidence → web app
```

# Expected Outcomes

- A **live, deployed demo** where a user pastes DNA and receives the translated protein and ranked GO predictions grouped by MF, BP and CC.
- A **documented, reproducible repository** with unit, hierarchy-consistency and API tests, and CI.
- A **results table** comparing the BLAST baseline, flat model and hierarchy-aware model(s), including rare-term and low-identity breakdowns.
- An **error analysis** explaining where the model helps, where homology wins, and why.

# Scope and Limitations

- Training data consists of protein sequences, so DNA→protein translation is an inference-time interface rather than something the model learns.
- Predictions are computational hypotheses, not experimental proof.
- Only frequent GO terms are modelled, and GO annotations are incomplete and biased toward well-studied proteins.

# Optional Extensions

- A GO-graph-aware output head (GNN or label embeddings) to help rare terms.
- LoRA fine-tuning of ESM-2 compared with frozen embeddings.
- Out-of-distribution flagging for proteins far from anything in training.
- A tool-using explanation agent that turns predictions and homolog evidence into a plain-language report.

If you want, I can condense this into a one-page abstract, a one-slide version for a review, or a week-by-week plan with the experiments mapped to H1 to H4.
        