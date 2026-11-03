# Error analysis

> **Single hand-picked example plus aggregate metrics; not a systematic error analysis.** The example below is from one real app run; aggregate values are from `results/metrics.csv`. Hypotheses are labeled and not treated as verified explanations.

## Example 1: ubiquitin (hand-picked)

For the synthetic human ubiquitin test sequence, broad terms ranked highest, including binding, protein binding, cytosol, and cytoplasm. This is biologically reasonable for the example. Specific processes such as ubiquitination or proteasomal degradation were not among the top results shown. They are possibly outside the top-200 vocabulary or ranked lower; this is a hypothesis and is not verified.

## Tied scores

Three BP terms had identical scores of 26.4%, and two CC terms had identical scores of 37.6%. This is consistent with the hierarchy-aware model's max-constraint output layer, where a parent scores at least as high as its children. The tied values alone do not prove that this constraint caused the ties.

## Rare vs frequent terms

From `results/metrics.csv`, flat-model rare-term F-max for MF/BP/CC was 0.146/0.186/0.137, compared with frequent-term F-max of 0.503/0.386/0.556. This aggregate result shows lower F-max on rare terms in all three aspects for this run.

## DIAMOND vs ESM-2

Flat-model F-max for MF/BP/CC was 0.474/0.339/0.534; DIAMOND top-hit F-max was 0.314/0.261/0.338. DIAMOND was higher on rare MF terms: 0.230 vs 0.146 for the flat model. Many test proteins had no DIAMOND hit: `results/diamond/hits.tsv` has about 2,142 lines for about 4,000 test proteins.

## Input robustness

The page displayed `Invalid DNA characters: XYZ (allowed: ACGTN)` for `ACGTXYZ`. For `CCCCCCCCCCCCCCCCCC`, it displayed `No ORF found (expected an ATG start codon and a translated peptide)`. `AC` and an empty input produced a readable FastAPI validation message. No minimum ORF length is enforced, so a short random sequence containing an ATG still returns predictions.

## Examples where DIAMOND beats ESM-2

**TODO: needs real predictions.** Add examples only after inspecting held-out labels, predictions from both methods, and train-only DIAMOND hit/bitscore evidence.

## Per-aspect failure cases

**TODO: needs real predictions.** Document specific MF, BP, and CC error patterns from held-out predictions and labels.

## Proteins with no DIAMOND hit

**TODO: needs real predictions.** Identify proteins with no hit and analyze their ESM-2 errors using saved predictions and labels. The aggregate hit count above does not establish per-protein error patterns.
