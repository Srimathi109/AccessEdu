# AccessEdu Project Completion Report

## Cricket Umpire Signal Practice and Assessment System

**Project status:** Functional local demonstration completed  
**Project area:** Computer vision and sports umpire training  
**Report date:** 20 October 2026

## Executive summary

AccessEdu is a local practice application for trainee cricket umpires. Trainees choose a signal scenario, record or upload a short video, and receive automated feedback comparing the expected signal with the signal most frequently detected in sampled frames. Instructors can create scenarios, add trainee accounts, and review progress summaries.

The project is complete as a working demonstration. It connects the existing cricket YOLO model to a practice workflow and stores attempt summaries. A larger synthetic stress set was generated from training images, and a fine-tuned candidate model was compared with the current checkpoint. The candidate improved on the synthetic set but regressed on the original validation and held-out test sets, so the current checkpoint remains in the app.

**The model can produce useful signal predictions, but the available evaluation does not show that every prediction will be correct.** The independent test split contains only nine images and has no `Out` examples. The larger synthetic set was derived from training data and is not an independent accuracy test.

## Project aim and objectives

The project aims to give trainee umpires a repeatable way to practise standardized signals and review their consistency. The working research question is: *How can computer vision support self-guided practice of cricket umpire signals, and what evidence is needed before that feedback can be trusted?*

Objectives:

- Provide a trainee workflow for selecting a signal scenario and submitting a practice recording.
- Recognize umpire signal categories in sampled video frames using the cricket detection model.
- Compare each prediction with the scenario expectation and save a practice summary.
- Give instructors tools to create scenarios, add trainee accounts, and review progress.
- Evaluate the current checkpoint and an augmented-data candidate while distinguishing synthetic results from independent results.

## Development approach: five sprints

The project was organized into five sprints. The sprint themes below reflect the team's Scrum plan; sprint dates, individual task assignments, and per-sprint completion metrics were not provided.

| Sprint | Focus | Work area |
|---|---|---|
| Sprint 1 | Research & Requirements | Define the training problem, intended users, scope, and requirements. |
| Sprint 2 | Dataset & ML Model | Prepare the cricket signal dataset and develop and evaluate the detection model. |
| Sprint 3 | Real-Time Recognition | Connect model inference to video-frame processing for signal recognition. |
| Sprint 4 | Accessibility & Event Interpretation | Shape the user experience around accessible practice and interpreting detected signal events. |
| Sprint 5 | AccessEdu Application | Integrate the application workflow into the AccessEdu application. |

## Completed system workflow

1. An instructor signs in, creates or activates practice scenarios, and creates trainee accounts.
2. A trainee signs in and chooses a scenario with an expected signal.
3. The trainee records a short browser clip or uploads a supported video file.
4. The application samples up to 24 frames and runs the cricket YOLO model on them.
5. AccessEdu selects the most frequently detected signal, compares it with the expected signal, and displays feedback.
6. The attempt is saved for the trainee’s history and the instructor’s progress summary.

## Implementation

| Component | Implementation | Purpose |
|---|---|---|
| Web application | Flask and server-rendered HTML templates | Local trainee and instructor workflows |
| Video capture | Browser MediaRecorder and file upload | Capture or submit practice clips |
| Inference | Ultralytics YOLO and OpenCV | Classify `No_Ball`, `Out`, `Six`, and `Wide` |
| Persistence | SQLite | Store users, scenarios, and attempt summaries |
| Training assets | Roboflow YOLO dataset and checkpoints | Train and evaluate cricket signal detection |

Automated feedback concerns the detected signal category and confidence. The current system does not evaluate hand position, signal timing, decision correctness, or readiness for umpire certification.

## Dataset and evaluation method

The supplied dataset contains **189 labeled images**: 165 training, 15 validation, and 9 test images. To explore robustness, five deterministic variants were generated for each training image: horizontal flip, exposure shift, small rotation/scale/translation with adjusted bounding boxes, reduced-resolution blur, and sensor noise. This produced an **825-image synthetic stress set**.

The original checkpoint and a candidate fine-tuned for five epochs on the original training images plus synthetic variants were evaluated at 640-pixel inference size. Results below are object-detection mean average precision (mAP), not whole-video signal classification accuracy.

| Checkpoint | Validation mAP50 (15 images) | Held-out test mAP50 (9 images) | Synthetic stress mAP50 (825 images) |
|---|---:|---:|---:|
| Current checkpoint | 0.879 | 0.995 | 0.724 |
| Fine-tuned candidate | 0.845 | 0.953 | 0.803 |

| Checkpoint | Validation mAP50-95 | Held-out test mAP50-95 | Synthetic stress mAP50-95 |
|---|---:|---:|---:|
| Current checkpoint | 0.731 | 0.870 | 0.562 |
| Fine-tuned candidate | 0.608 | 0.719 | 0.604 |

## Findings and model decision

The fine-tuned candidate improved synthetic stress mAP50 by about 0.079, but lowered mAP50 on both original splits. It also lowered mAP50-95 on the original validation and test splits. The candidate was therefore **not promoted**; the app continues to use `runs/detect/accessedu_cricket_v1/weights/best.pt`.

The original detector test score of 0.995 is based on only nine labeled objects and does not cover `Out`. It cannot establish reliable performance across all four supported classes or across new people and recording conditions. The synthetic set uses altered versions of training images that the model may already have seen, so it is a controlled stress check rather than independent accuracy evidence.

## Completion assessment

| Area | Status | Evidence |
|---|---|---|
| Trainee practice flow | Implemented | Scenario selection, video capture/upload, prediction, and result summary |
| Instructor tools | Implemented | Scenario management, trainee account creation, and progress overview |
| Saved progress | Implemented | SQLite attempt history and instructor summaries |
| Model connection | Implemented | Cricket checkpoint used in sampled-frame video inference |
| Independent accuracy evidence | Incomplete | Nine test images; no `Out` examples in the test split |
| Production deployment | Out of scope | Local prototype without production authentication/deployment hardening |

## Limitations

- The dataset is too small to estimate reliability across different umpires, cameras, lighting, distances, and backgrounds.
- Synthetic stress images are derived from training data and cannot replace independently recorded test videos.
- The detector does not assess hand position, timing, signal clarity, decision correctness, or certification readiness.
- Automated feedback can be wrong and should not be treated as an official assessment without instructor validation.
- The app is configured for a local demonstration; demo credentials and SQLite storage are not a production identity or deployment system.

## Conclusion and next steps

AccessEdu is complete as a local research demonstration of a computer-vision-supported umpire practice workflow. It connects a cricket signal detector to trainee practice, saves outcomes, and gives instructors a progress view. The evaluation shows plausible signal detection, but there is not enough independent evidence to promise correct predictions.

Recommended next steps:

1. Collect and label new videos from multiple umpires for every class, especially `Out`, and keep them separate from training data.
2. Evaluate whole-video signal decisions and per-class errors on a larger instructor-reviewed test set.
3. Review false positives, false negatives, and confidence thresholds with qualified umpire instructors.
4. Promote a new checkpoint only if it improves on independent recordings without unacceptable class-specific regressions.
5. Before deployment beyond a local demo, replace demo accounts and add production authentication, security controls, and data-retention policies.

## Reproducibility and references

Run the application from the project root using the instructions in `README.md`. Regenerate the synthetic stress set with `python training/generate_synthetic_stress.py`. The augmented fine-tuning configuration is `dataset/augmented_training.yaml`. Evaluation details are in `results/model_robustness_report.md`, with model run artifacts under `runs/detect`.

The cricket-umpire dataset is attributed to Roboflow Universe under CC BY 4.0. Source: [Cricket Umpire Action Detection dataset](https://universe.roboflow.com/prabu/cricket-umpire-action-detection/dataset/1). Export details are in `README.roboflow.txt`.
