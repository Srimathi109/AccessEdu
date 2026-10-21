# AccessEdu — Cricket Umpire Practice and Assessment

AccessEdu is an AI-assisted cricket umpire training and assessment prototype. Instructors create signal challenges, trainees perform a signal on camera, and the image classifier returns a category and confidence estimate. Attempts, response time and a private still-image evidence file are saved for instructor feedback, progress tracking and readiness review.

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
2. A trainee chooses an active scenario, consents to private evidence storage and starts the camera; one photo is captured after a five-second countdown.
3. The classifier predicts a signal category from that still image and reports confidence.
4. AccessEdu compares the prediction with the challenge's expected signal and recommends practice based on attempt history.
5. The attempt, response time and photo evidence are stored for the instructor to review, correct and add coaching feedback.
6. Instructors set per-signal practice requirements. Meeting them recommends instructor review; an instructor can then record a completion approval.

Automatic photo capture requires camera permission and a secure context (localhost is supported). Trainees do not need to record a video or press a capture button.

## Model and dataset

The active cricket classifier is `runs/classify/accessedu_cricket_four_class/weights/best.pt`. It predicts `No_Ball`, `Out`, `Six`, or `Wide`; No_Action is not a supported model class. The root-level `best.pt` and `models/baseline.pt` are legacy sign-language checkpoints and must not be used for cricket signals. The app checks that the active checkpoint is a classifier with the expected four class names.

The four-class image-classification dataset is generated at `dataset/umpire_classifier_four_class`, with `train`, `val`, and `test` folders. It includes the supported uploaded signal folders and person crops from the original YOLO labels; the No_Action folder and class-0 crops are excluded. The uploaded dataset contains still images, not video sequences. See `results/umpire_classifier_report.md` for held-out metrics and model limitations.

Automated feedback only reports the signal category recognized in one photo, confidence, and whether it matches the practice scenario. The current model does not analyze movement, fine hand position, signal duration, decision correctness, or readiness for certification. Response time is an approximate browser-side measure from camera start to submission. Treat model output and readiness metrics as practice guidance; only the instructor can approve local course completion, which is not an official umpire qualification.

To regenerate the combined classifier dataset and fine-tune a candidate, run `python training/train_cricket_model.py`. The script starts from the current checkpoint, preserves validation and test images, and writes a four-class candidate under `runs/classify/`. It does not automatically replace the active checkpoint.

The dataset was exported from Roboflow Universe under CC BY 4.0; see `README.dataset.txt` and `README.roboflow.txt`.

## Deployment notes

Do not use Flask's built-in development server for a public deployment. Waitress is included for production serving on Windows; after installing `requirements.txt`, configure a stable, randomly generated `ACCESSEDU_SECRET_KEY`, `ACCESSEDU_HTTPS=1`, `ACCESSEDU_DEMO_ACCOUNTS=0`, `ACCESSEDU_DATABASE` pointing at a new persistent database, and strong `ACCESSEDU_INSTRUCTOR_USERNAME` / `ACCESSEDU_INSTRUCTOR_PASSWORD` values (the bootstrap password must be at least 12 characters). Then run `waitress-serve --listen=127.0.0.1:8080 app:app` behind an HTTPS reverse proxy. Demo accounts are disabled and hidden on the login page in this setup. Form submissions use session-bound CSRF tokens, and responses include baseline browser security headers. The app exposes `GET /health` as a lightweight liveness check.

Uploads are limited to 10 MB. The temporary inference copy is deleted after analysis; with trainee consent, the captured still photo is retained in the private `private_attempts/` directory and is served to signed-in instructors through an authorization-checked route. Set and follow an appropriate evidence-retention policy for your organization. The SQLite database, model checkpoint and private attempt evidence must be on persistent storage at the paths configured in `app.py`; SQLite is intended for a single application instance on a local persistent volume. Back up the database and model files as part of deployment operations. Camera access requires HTTPS outside localhost.

The model checkpoint currently in use is `runs/classify/accessedu_cricket_four_class/weights/best.pt`. Training is provided as the Python script `training/train_cricket_model.py`; this repository does not currently contain a Jupyter `.ipynb` notebook.
