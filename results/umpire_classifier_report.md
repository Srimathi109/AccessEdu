# Cricket umpire classifier report

## Supported classes

The app now supports four umpire signals: `No_Ball`, `Out`, `Six`, and `Wide`. The No_Action class is excluded from the prepared training data and the active checkpoint. Uploaded source files are left intact.

## Data and model

The combined four-class dataset is `dataset/umpire_classifier_four_class`. It combines the four uploaded signal folders with crops from the original YOLO labels. The generated splits contain 172/145/157/157 training images, 14/14/12/16 validation images, and 15/13/17/15 test images for No_Ball, Out, Six, and Wide respectively.

The active model is `runs/classify/accessedu_cricket_four_class/weights/best.pt`, fine-tuned from the prior cricket classifier with the unsupported class removed.

## Held-out results

The four-class checkpoint scored **93.3% top-1 accuracy (56 of 60)** on the held-out test split.

| Class | Test examples | Recall |
|---|---:|---:|
| No_Ball | 15 | 100% |
| Out | 13 | 85% |
| Six | 17 | 100% |
| Wide | 15 | 87% |

These are still-image results from a small test set and do not establish dependable performance on every camera, umpire, or live-video condition. The dataset contains no video sequences.

## Inference display

The standalone webcam window displays one centered class name at the top of the frame. A seven-frame majority vote reduces label flicker; the frame is not covered with boxes or per-class scores. Uploaded-video analysis returns one summarized class after sampling frames.
