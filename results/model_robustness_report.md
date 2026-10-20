# Cricket signal model robustness check

## Summary

The current checkpoint detects cricket umpire signals, but the available evidence does not support saying its predictions will always be correct. A stress set was generated from training images, then the current checkpoint and a fine-tuned candidate were evaluated on it. The fine-tuned candidate improved on this synthetic stress set but regressed on both the original validation set and the tiny held-out test set, so it was **not promoted**. The app continues to use `runs/detect/accessedu_cricket_v1/weights/best.pt`.

## Results

All evaluation runs used 640-pixel inference. Metrics are object-detection mAP, not exact whole-video signal-class accuracy.

| Checkpoint | Original validation mAP50 (15 images) | Original test mAP50 (9 images) | Synthetic stress mAP50 (825 images) |
|---|---:|---:|---:|
| Current checkpoint | 0.879 | 0.995 | 0.724 |
| Fine-tuned candidate | 0.845 | 0.953 | 0.803 |

The current checkpoint's mAP50-95 was 0.731 on validation, 0.870 on the nine-image test split, and 0.562 on the synthetic stress set. The candidate scored 0.608, 0.719, and 0.604 respectively.

The original test split has only nine labeled objects and contains no `Out` examples. Its 0.995 score is therefore not evidence of dependable performance across all five classes. The synthetic set contains altered versions of training images the model may already have seen, so its score is a stress check only and must not be presented as an independent test accuracy.

## Synthetic set generation

`training/generate_synthetic_stress.py` creates five deterministic variants of each of the 165 training images: horizontal flip, exposure shift, small rotation/scale/translation with transformed bounding boxes, reduced-resolution blur, and sensor noise. This produces 825 images with YOLO labels. `dataset/augmented_training.yaml` combines these with the original training images for optional fine-tuning; the original validation and test splits stay separate.

Regenerate with:

```powershell
python training/generate_synthetic_stress.py
```

The current checkpoint's metrics are in `runs/detect/runs/detect/accessedu_original_test/` and `runs/detect/runs/detect/accessedu_synthetic_train_stress/`. Candidate training and comparison outputs are under `runs/detect/runs/detect/accessedu_augmented_finetune/` and the sibling `compare_*` / `accessedu_candidate_*` evaluation folders.

## Recommendation

Keep the current checkpoint in the app for now. The synthetic candidate is more robust to these generated changes, but its lower scores on the original held-out and validation sets make promoting it unjustified. Collect new, independently recorded videos for every signal—including `Out`—with different umpires, cameras, distances, lighting, and backgrounds. Label those videos carefully and reserve them exclusively for evaluation. Do not promise perfectly correct predictions; use the model's feedback as practice guidance until an instructor-reviewed independent evaluation supports stronger claims.
