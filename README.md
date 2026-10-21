# AccessEdu — Cricket Umpire Training and Match Scoring

AccessEdu combines two connected but separate workflows:

- **Module 1 — Umpire training and assessment:** instructors create signal challenges, trainees practise them, and instructors review attempts and local course readiness.
- **Module 2 — Instructor-assisted match scoring:** an instructor records a match and can use camera or clip analysis to propose events for scorer confirmation.

Both modules use the same four-class cricket signal image model. Module 2 match data is stored separately from Module 1 challenges and trainee attempts.

## Requirements and local setup

Use Python 3.10 or newer. In PowerShell, run these commands from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000>. On first startup, AccessEdu creates the SQLite database at `accessedu.sqlite3` and initializes the application tables.

### Demonstration accounts

| Role | Username | Password |
|---|---|---|
| Trainee | `trainee` | `trainee123` |
| Instructor | `instructor` | `instructor123` |

These credentials are for local demonstration only. Disable demo accounts and configure a strong secret and instructor credentials before deployment.

## Module 1 — Umpire training and assessment

### Trainee workflow

1. Sign in as a trainee and choose an active signal challenge.
2. Grant camera permission and consent to evidence storage if evidence is desired.
3. The browser captures a still image after a short countdown and submits it for classification.
4. AccessEdu compares the predicted class with the expected signal, shows confidence and feedback, and records the attempt and approximate response time.
5. If consent was given, the captured image is retained privately for authorized instructor review.

Camera access requires a secure browser context: localhost works for local development; deployed sites need HTTPS. The browser camera is activated by the trainee and can be blocked or stopped through browser/device settings.

### Instructor workflow

Instructors can manage scenarios, review trainee attempts, inspect consented evidence, override a prediction, add feedback, and set practice requirements. Meeting local requirements can recommend an instructor review. An instructor makes the final course completion decision. This is not an official umpire qualification.

## Module 2 — Instructor match scoring

Open **Match scoring** from the instructor navigation to create and manage matches. A new match starts with the first listed team batting. The instructor ends each innings after reviewing pending proposals; the second innings starts automatically, and ending the second innings marks the match complete.

### Scoring options

- **Live camera:** select **Start camera** to request access. The app analyzes frames while the camera is on. Three consecutive, stable predictions at or above 72% confidence create a pending event proposal. When one is created, the camera stops; select **Refresh proposals** to load and review it.
- **Recorded clip:** upload a supported clip up to 10 MB and 120 seconds. The server samples frames at roughly one-second intervals. The temporary upload is removed after processing.
- **Manual scorer entry:** use this when the event is clear but AI detection is unavailable or incorrect. Manually entered events are immediately confirmed by the instructor.

### Proposal review and score changes

Camera and clip detections never change the scoreboard by themselves. A proposal appears in **Pending proposals** with the predicted signal, source, confidence, and suggested score. The instructor must confirm or reject it. Rejecting has no scoring effect. Confirming applies these prototype defaults:

| Signal | Suggested score | Delivery |
|---|---:|---|
| Six | +6 runs | Legal ball |
| Out | 1 wicket | Legal ball |
| Wide | +1 run | Extra delivery |
| No Ball | +1 run | Extra delivery |

The scorer can add runs to a Wide or No Ball during confirmation. Confirmed events can be corrected or voided with a required reason; the correction history is retained. Legal balls are capped at the match's overs limit, and an innings cannot accept a new wicket after ten wickets.

### Module 2 limitations

The model classifies individual frames. It does not understand delivery context, count all runs, detect every wicket, judge whether an umpire's decision was correct, or apply every cricket law. Confidence is only a model estimate, not proof of a correct call. The model supports `Six`, `Out`, `Wide`, and `No_Ball`; it has no `No_Action` class. Keep an instructor/scorer in control of every official score change. Module 2 is a scoring aid, not an official scoring authority or certification decision.

The live browser camera is only accessed after the instructor presses **Start camera** and allows permission. Frames are sent for inference and are not saved. A camera, permission, secure-context, or network error can prevent live detection; use manual scorer entry in that case.

## Model and training data

- Active cricket checkpoint: `runs/classify/accessedu_cricket_four_class/weights/best.pt`
- Model training script: `training/train_cricket_model.py`
- Combined classifier dataset: `dataset/umpire_classifier_four_class` (`train`, `val`, and `test` folders)
- Model evaluation notes: `results/umpire_classifier_report.md`

The model predicts `No_Ball`, `Out`, `Six`, or `Wide`. The root-level `best.pt` and `models/baseline.pt` are legacy sign-language checkpoints and are not used by AccessEdu's cricket workflow. The app verifies the active checkpoint's task and class names on load.

The dataset is based on still images, not video sequences, so clip scoring applies single-image classification to sampled frames rather than temporal action recognition. To build a candidate model, run `python training/train_cricket_model.py`; the script writes a candidate under `runs/classify/` and does not automatically replace the active checkpoint. Dataset licensing details are in `README.dataset.txt` and `README.roboflow.txt`.

There is currently **no Jupyter `.ipynb` notebook** in the repository; model training is provided as the Python script above.

## Deployment notes

Do not expose Flask's built-in development server to the public internet. For a Windows deployment, Waitress is included. Configure:

- `ACCESSEDU_SECRET_KEY` with a stable, randomly generated secret.
- `ACCESSEDU_HTTPS=1` when deployed behind HTTPS.
- `ACCESSEDU_DEMO_ACCOUNTS=0` to disable demonstration accounts.
- `ACCESSEDU_DATABASE` to a persistent database path.
- `ACCESSEDU_INSTRUCTOR_USERNAME` and `ACCESSEDU_INSTRUCTOR_PASSWORD` for bootstrap instructor credentials. The bootstrap password must be at least 12 characters.

Run Waitress behind an HTTPS reverse proxy, for example:

```powershell
waitress-serve --listen=127.0.0.1:8080 app:app
```

The app uses session-bound CSRF tokens and baseline browser security headers, and exposes `GET /health` as a liveness check. Keep the SQLite database, model checkpoint, and private Module 1 evidence on persistent storage and back them up. SQLite is intended for a single app instance on a persistent local volume. Module 1 evidence is retained only with trainee consent under `private_attempts/`; establish an appropriate retention policy. Module 2 video uploads are temporary and deleted after analysis. The configured upload limit is 10 MB.
