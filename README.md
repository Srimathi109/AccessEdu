# AccessEdu — Cricket Umpire Practice and Assessment

AccessEdu is a local web app for trainee cricket umpires. An instructor provides practice scenarios, a trainee records or uploads a signal, the cricket YOLO model analyzes sampled video frames, and the trainee receives a summary. Attempts are saved for progress tracking and instructor review.

## Run locally

Use Python 3.10 or newer. From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`. The SQLite database is created as `accessedu.sqlite3` on first startup.

### Demo accounts

| Role | Username | Password |
|---|---|---|
| Trainee | `trainee` | `trainee123` |
| Instructor | `instructor` | `instructor123` |

These are demonstration credentials for a local prototype. Change them and configure `ACCESSEDU_SECRET_KEY` before deploying beyond a trusted local environment. The app binds to localhost by default.

## How practice works

1. An instructor uses the instructor dashboard to review progress and add or archive signal scenarios.
2. A trainee chooses an active scenario and records a short clip in the browser or selects a video file.
3. The model samples up to 24 frames and predicts the most frequent detected signal.
4. AccessEdu compares that prediction with the scenario's expected signal and provides practice guidance.
5. The attempt is saved in SQLite for the trainee's history and the instructor's summary.

Supported upload extensions are MP4, MOV, AVI, MKV, and WebM, up to 100 MB. Browser camera recording requires camera permission and a secure context (localhost is supported). OpenCV must be able to decode the uploaded video format.

## Model and dataset

The cricket checkpoint is `runs/detect/accessedu_cricket_v1/weights/best.pt`. The root-level `best.pt` and `models/baseline.pt` are legacy sign-language checkpoints and must not be used for cricket signals. The app checks the cricket checkpoint's class names before inference.

Dataset configuration: `dataset/data.yaml`. Classes are `No_Action`, `No_Ball`, `Out`, `Six`, and `Wide`. The dataset contains 189 images (165 train, 15 validation, and 9 test). That is far too small to establish reliable performance across different people, cameras, lighting, and recording conditions.

Automated feedback only reports the signal category recognized in the video, confidence, and whether it matches the practice scenario. The current model does not evaluate fine hand position, signal duration, decision correctness, or readiness for certification. Treat results as practice guidance and validate them with qualified instructors before using them for formal assessment.

An additional 825-image synthetic stress set was generated from altered training images. This is useful for checking basic robustness to lighting, blur, noise, flipping, and small rotations, but it is not an independent accuracy test. The resulting comparison and fine-tuning decision are documented in `results/model_robustness_report.md`; the original checkpoint remains the one used by the app because the fine-tuned candidate regressed on the original held-out splits.

The dataset was exported from Roboflow Universe under CC BY 4.0; see `README.dataset.txt` and `README.roboflow.txt`.
