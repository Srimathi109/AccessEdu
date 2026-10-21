# Historical cricket detector robustness check

> This report covers the earlier YOLO object detector and synthetic fine-tuning experiment. The app now uses the four-class image classifier described in [umpire_classifier_report.md](umpire_classifier_report.md); these historical detector metrics do not describe the active checkpoint.

## Summary

At the time of this experiment, a stress set was generated from training images and the original detector and a fine-tuned candidate were evaluated on it. The candidate improved on the synthetic stress set but regressed on the original validation and tiny test splits, so it was not promoted. The app has since moved to a separately trained image classifier.

## Results

All evaluation runs used 640-pixel inference. Metrics are object-detection mAP, not exact whole-video signal-class accuracy.

| Checkpoint | Original validation mAP50 (15 images) | Original test mAP50 (9 images) | Synthetic stress mAP50 (825 images) |
|---|---:|---:|---:|
| Current checkpoint | 0.879 | 0.995 | 0.724 |
| Fine-tuned candidate | 0.845 | 0.953 | 0.803 |

The current checkpoint's mAP50-95 was 0.731 on validation, 0.870 on the nine-image test split, and 0.562 on the synthetic stress set. The candidate scored 0.608, 0.719, and 0.604 respectively.

The original test split has only nine labeled objects and contains no `Out` examples. Its 0.995 score is therefore not evidence of dependable performance across the four supported classes. The synthetic set contains altered versions of training images the model may already have seen, so its score is a stress check only and must not be presented as an independent test accuracy.

## Synthetic set generation

`training/generate_synthetic_stress.py` creates five deterministic variants of each of the 165 training images: horizontal flip, exposure shift, small rotation/scale/translation with transformed bounding boxes, reduced-resolution blur, and sensor noise. This produces 825 images with YOLO labels. `dataset/augmented_training.yaml` combines these with the original training images for optional fine-tuning; the original validation and test splits stay separate.

Regenerate with:

```powershell
python training/generate_synthetic_stress.py
```

The current checkpoint's metrics are in `runs/detect/runs/detect/accessedu_original_test/` and `runs/detect/runs/detect/accessedu_synthetic_train_stress/`. Candidate training and comparison outputs are under `runs/detect/runs/detect/accessedu_augmented_finetune/` and the sibling `compare_*` / `accessedu_candidate_*` evaluation folders.

## Recommendation

Keep the current checkpoint in the app for now. The synthetic candidate is more robust to these generated changes, but its lower scores on the original held-out and validation sets make promoting it unjustified. Collect new, independently recorded videos for every signal—including `Out`—with different umpires, cameras, distances, lighting, and backgrounds. Label those videos carefully and reserve them exclusively for evaluation. Do not promise perfectly correct predictions; use the model's feedback as practice guidance until an instructor-reviewed independent evaluation supports stronger claims.
