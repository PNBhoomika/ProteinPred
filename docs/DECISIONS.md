# Decisions

- `PROJECT_SPEC.md` was present and read. The reduced scope in the task overrides its linear baseline, hybrid, calibration, S-min, identity-bin, deployment, and CI items.
- Fixed seed: 42. Subsample: up to 40,000 sequences no longer than 512 residues. Term cap: 200 per aspect, configurable in notebook 01.
- MMseqs2 `easy-cluster` is used with 30% minimum identity and 80% coverage; random split is an explicit recorded fallback.
- ESM-2 is frozen and mean pooled. Training happens only in the supplied Kaggle notebooks; no local embedding extraction or model fitting.
- The API chooses the hierarchy-aware model when exported weights exist. Demo predictions are deterministic illustrative scores and are labeled as demo output.

## Results and limitations

- On a 30%-identity cluster split, the ESM-2 flat classifier has higher F-max than DIAMOND top-hit transfer for MF, BP, and CC.
- The hierarchy-aware model did not improve F-max: MF is effectively tied given the marginal difference in the CSV, with slightly lower F-max on BP and CC. Post-hoc hierarchy enforcement gives only a negligible gain, about +0.001 F-max.
- DIAMOND has higher `rare_fmax` than the flat model on MF.
- Many test proteins had no DIAMOND hit; the retained hit file has substantially fewer rows than the test set has proteins, which lowers DIAMOND's F-max.
- This is one run with one seed and no confidence intervals. Evaluation is on a 30%-identity cluster split; identity bins were not run, so no bin-wise conclusion was assessed.
- The hybrid model, calibration, and deployment are not implemented.
