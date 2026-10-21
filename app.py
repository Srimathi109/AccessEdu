"""AccessEdu local training app for cricket umpire signals."""

from __future__ import annotations

import os
import hmac
import math
import secrets
import shutil
import sqlite3
import tempfile
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

import cv2
from flask import (
    Flask,
    abort,
    flash,
    g,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "runs" / "classify" / "accessedu_cricket_four_class" / "weights" / "best.pt"
EXPECTED_CLASSES = {"No_Ball", "Out", "Six", "Wide"}
_MODEL = None
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ATTEMPT_MEDIA_DIR = ROOT / "private_attempts"
ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
CLASS_LABELS = {
    "No_Ball": "No Ball",
    "Out": "Out",
    "Six": "Six",
    "Wide": "Wide",
}

app = Flask(__name__, template_folder="app/templates", static_folder="app/static")
demo_accounts_enabled = os.environ.get("ACCESSEDU_DEMO_ACCOUNTS", "1").lower() not in {"0", "false", "no"}
app.config.update(
    SECRET_KEY=os.environ.get("ACCESSEDU_SECRET_KEY") or secrets.token_hex(32),
    MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
    DATABASE=os.environ.get("ACCESSEDU_DATABASE", str(ROOT / "accessedu.sqlite3")),
    SHOW_DEMO_ACCOUNTS=demo_accounts_enabled,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("ACCESSEDU_HTTPS", "").lower() in {"1", "true", "yes"},
)


@app.context_processor
def inject_csrf_token():
    def csrf_token():
        if "csrf_token" not in session:
            session["csrf_token"] = secrets.token_urlsafe(32)
        return session["csrf_token"]

    return {"csrf_token": csrf_token}


@app.before_request
def protect_form_posts():
    if request.method == "POST":
        submitted = request.form.get("csrf_token", "")
        expected = session.get("csrf_token", "")
        if not submitted or not expected or not hmac.compare_digest(submitted, expected):
            abort(400)


@app.after_request
def add_security_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(self), microphone=()")
    return response


def database():
    if "db" not in g:
        g.db = sqlite3.connect(app.config["DATABASE"])
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_database(_error=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = sqlite3.connect(app.config["DATABASE"])
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            display_name TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('trainee', 'instructor'))
        );
        CREATE TABLE IF NOT EXISTS scenarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            expected_signal TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1,
            created_by INTEGER REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id),
            scenario_id INTEGER NOT NULL REFERENCES scenarios(id),
            expected_signal TEXT NOT NULL,
            predicted_signal TEXT NOT NULL,
            confidence REAL NOT NULL,
            matched INTEGER NOT NULL,
            frame_count INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS signal_requirements (
            signal TEXT PRIMARY KEY,
            min_attempts INTEGER NOT NULL DEFAULT 3,
            min_match_rate INTEGER NOT NULL DEFAULT 80
        );
        CREATE TABLE IF NOT EXISTS completion_approvals (
            trainee_id INTEGER PRIMARY KEY REFERENCES users(id),
            approved_by INTEGER NOT NULL REFERENCES users(id),
            approved_at TEXT NOT NULL,
            notes TEXT NOT NULL DEFAULT ''
        );
        """
    )
    existing_attempt_columns = {row[1] for row in db.execute("PRAGMA table_info(attempts)")}
    for column, definition in {
        "response_seconds": "REAL NOT NULL DEFAULT 0",
        "instructor_override": "INTEGER",
        "instructor_feedback": "TEXT NOT NULL DEFAULT ''",
        "reviewed_by": "INTEGER REFERENCES users(id)",
        "reviewed_at": "TEXT",
        "evidence_path": "TEXT",
        "evidence_consent_at": "TEXT",
    }.items():
        if column not in existing_attempt_columns:
            db.execute(f"ALTER TABLE attempts ADD COLUMN {column} {definition}")
    db.executemany(
        "INSERT OR IGNORE INTO signal_requirements(signal,min_attempts,min_match_rate) VALUES(?,?,?)",
        [(signal, 3, 80) for signal in sorted(EXPECTED_CLASSES)],
    )
    accounts = []
    if demo_accounts_enabled:
        accounts.extend([
            ("trainee", generate_password_hash("trainee123"), "Demo Trainee", "trainee"),
            ("instructor", generate_password_hash("instructor123"), "Demo Instructor", "instructor"),
        ])
    bootstrap_username = os.environ.get("ACCESSEDU_INSTRUCTOR_USERNAME", "").strip().lower()
    bootstrap_password = os.environ.get("ACCESSEDU_INSTRUCTOR_PASSWORD", "")
    if bootstrap_username and len(bootstrap_password) >= 12:
        accounts.append((bootstrap_username, generate_password_hash(bootstrap_password), "Course Instructor", "instructor"))
    if accounts:
        db.executemany(
            "INSERT OR IGNORE INTO users(username,password_hash,display_name,role) VALUES(?,?,?,?)",
            accounts,
        )
    db.commit()
    instructor = db.execute("SELECT id FROM users WHERE role='instructor' ORDER BY id LIMIT 1").fetchone()
    instructor_id = instructor[0] if instructor else None
    samples = [
        ("Signal practice: Six", "Practise the standard signal for a boundary six.", "Six", 1, instructor_id),
        ("Signal practice: Out", "Practise the standard out signal clearly and hold it briefly.", "Out", 1, instructor_id),
        ("Signal practice: Wide", "Practise the signal used to call a wide delivery.", "Wide", 1, instructor_id),
        ("Signal practice: No Ball", "Practise the signal used to call a no-ball.", "No_Ball", 1, instructor_id),
    ]
    db.executemany(
        "INSERT INTO scenarios(title,description,expected_signal,active,created_by) "
        "SELECT ?,?,?,?,? WHERE NOT EXISTS (SELECT 1 FROM scenarios WHERE title=?)",
        [item + (item[0],) for item in samples],
    )
    create_match_schema(db)
    db.commit()
    db.close()


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)

    return wrapped


def role_required(role):
    def decorator(view):
        @wraps(view)
        @login_required
        def wrapped(*args, **kwargs):
            if session.get("role") != role:
                abort(403)
            return view(*args, **kwargs)

        return wrapped

    return decorator


def get_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    if not MODEL_PATH.is_file():
        raise RuntimeError("The cricket model is missing. Restore the checkpoint at " + str(MODEL_PATH))
    from ultralytics import YOLO

    model = YOLO(str(MODEL_PATH))
    if model.task != "classify" or set(model.names.values()) != EXPECTED_CLASSES:
        raise RuntimeError("The selected checkpoint is not the four-class cricket signal classifier.")
    _MODEL = model
    return _MODEL


def analyze_image(path):
    """Classify one captured or uploaded still image."""
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError("The photo could not be opened. Choose a JPG, PNG, or WebP image.")
    label, confidence = predict_frame(image)
    return label, confidence, 1


def predict_frame(image):
    """Run the shared four-class cricket classifier on an OpenCV frame."""
    model = get_model()
    result = model(image, imgsz=224, verbose=False)[0]
    probabilities = result.probs
    if probabilities is None:
        raise ValueError("The image frame could not be classified. Try another frame.")
    class_id = int(probabilities.top1)
    confidence = float(probabilities.top1conf.item())
    return model.names[class_id], confidence


def feedback_for(expected, predicted, confidence):
    if predicted != expected:
        return (
            f"The model most often recognized {CLASS_LABELS.get(predicted, predicted)}, while this scenario expects "
            f"{CLASS_LABELS.get(expected, expected)}. Review the signal and try again."
        )
    if confidence < 0.55:
        return "The expected signal was recognized, but confidence was low. Hold the signal clearly and keep it in view for longer."
    return "The expected signal was recognized consistently. Keep practising the same clear, steady presentation."


@app.route("/")
def index():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return redirect(url_for("dashboard"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")
        user = database().execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["display_name"] = user["display_name"]
            session["role"] = user["role"]
            destination = request.args.get("next", "")
            if destination.startswith("/") and not destination.startswith("//"):
                return redirect(destination)
            return redirect(url_for("dashboard"))
        flash("That username or password was not recognized.", "error")
    return render_template("login.html")


@app.route("/health")
def health():
    """Lightweight liveness check for a hosting platform."""
    return {"status": "ok"}


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    if session["role"] == "instructor":
        return redirect(url_for("instructor_dashboard"))
    db = database()
    scenarios = db.execute("SELECT * FROM scenarios WHERE active=1 ORDER BY id").fetchall()
    attempts = db.execute(
        "SELECT a.*,COALESCE(a.instructor_override,a.matched) AS effective_match,s.title FROM attempts a JOIN scenarios s ON s.id=a.scenario_id "
        "WHERE a.user_id=? ORDER BY a.id DESC LIMIT 8",
        (session["user_id"],),
    ).fetchall()
    totals = db.execute(
        "SELECT COUNT(*) AS total, COALESCE(SUM(COALESCE(instructor_override,matched)),0) AS correct, "
        "COALESCE(AVG(NULLIF(response_seconds,0)),0) AS avg_response FROM attempts WHERE user_id=?",
        (session["user_id"],),
    ).fetchone()
    rate = round(100 * totals["correct"] / totals["total"]) if totals["total"] else 0
    signal_progress = db.execute(
        "SELECT r.signal,r.min_attempts,r.min_match_rate,COUNT(a.id) AS attempts, "
        "COALESCE(SUM(COALESCE(a.instructor_override,a.matched)),0) AS correct "
        "FROM signal_requirements r LEFT JOIN attempts a ON a.expected_signal=r.signal AND a.user_id=? "
        "GROUP BY r.signal ORDER BY r.signal", (session["user_id"],)
    ).fetchall()
    approval = db.execute(
        "SELECT * FROM completion_approvals WHERE trainee_id=?", (session["user_id"],)
    ).fetchone()
    eligible_for_review = bool(signal_progress) and all(
        item["attempts"] >= item["min_attempts"]
        and (100 * item["correct"] / item["attempts"] if item["attempts"] else 0) >= item["min_match_rate"]
        for item in signal_progress
    )
    recommendations = db.execute(
        "SELECT s.id,s.title,s.expected_signal,COUNT(*) AS misses FROM attempts a "
        "JOIN scenarios s ON s.id=a.scenario_id WHERE a.user_id=? AND a.matched=0 AND s.active=1 "
        "GROUP BY s.id ORDER BY misses DESC,s.id LIMIT 2", (session["user_id"],)
    ).fetchall()
    return render_template("trainee.html", scenarios=scenarios, attempts=attempts, totals=totals, rate=rate,
                           signal_progress=signal_progress, approval=approval if eligible_for_review else None,
                           recommendations=recommendations)


@app.route("/practice/<int:scenario_id>")
@role_required("trainee")
def practice(scenario_id):
    scenario = database().execute(
        "SELECT * FROM scenarios WHERE id=? AND active=1", (scenario_id,)
    ).fetchone()
    if scenario is None:
        abort(404)
    return render_template("practice.html", scenario=scenario)


@app.route("/practice/<int:scenario_id>/submit", methods=["POST"])
@role_required("trainee")
def submit_practice(scenario_id):
    scenario = database().execute(
        "SELECT * FROM scenarios WHERE id=? AND active=1", (scenario_id,)
    ).fetchone()
    if scenario is None:
        abort(404)
    if request.form.get("evidence_consent") != "yes":
        flash("Consent to private photo storage before starting the challenge.", "error")
        return redirect(url_for("practice", scenario_id=scenario_id))
    photo = request.files.get("photo")
    if not photo or not photo.filename:
        flash("Capture or choose a practice photo first.", "error")
        return redirect(url_for("practice", scenario_id=scenario_id))
    filename = secure_filename(photo.filename)
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        flash("Choose a photo file such as JPG, PNG, or WebP.", "error")
        return redirect(url_for("practice", scenario_id=scenario_id))
    temp_path = None
    evidence_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix="." + extension) as temp_file:
            temp_path = temp_file.name
            photo.save(temp_path)
        if os.path.getsize(temp_path) > MAX_UPLOAD_BYTES:
            raise ValueError("That photo is over the 10 MB upload limit. Choose a smaller image.")
        predicted, confidence, frame_count = analyze_image(temp_path)
        ATTEMPT_MEDIA_DIR.mkdir(parents=True, exist_ok=True)
        evidence_path = ATTEMPT_MEDIA_DIR / f"attempt-{secrets.token_hex(12)}.jpg"
        shutil.copyfile(temp_path, evidence_path)
    except (RuntimeError, ValueError) as exc:
        flash(str(exc), "error")
        return redirect(url_for("practice", scenario_id=scenario_id))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
    expected = scenario["expected_signal"]
    matched = predicted == expected and confidence >= 0.35
    try:
        response_seconds = max(0.0, min(float(request.form.get("response_seconds", "0")), 3600.0))
    except ValueError:
        response_seconds = 0.0
    if not math.isfinite(response_seconds):
        response_seconds = 0.0
    cursor = database().execute(
        "INSERT INTO attempts(user_id,scenario_id,expected_signal,predicted_signal,confidence,matched,frame_count,created_at,response_seconds,evidence_path,evidence_consent_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (
            session["user_id"], scenario_id, expected, predicted, confidence, int(matched), frame_count,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            response_seconds, str(evidence_path), datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ),
    )
    database().commit()
    database().execute("DELETE FROM completion_approvals WHERE trainee_id=?", (session["user_id"],))
    database().commit()
    return redirect(url_for("result", attempt_id=cursor.lastrowid))


@app.route("/result/<int:attempt_id>")
@role_required("trainee")
def result(attempt_id):
    attempt = database().execute(
        "SELECT a.*,COALESCE(a.instructor_override,a.matched) AS effective_match,s.title FROM attempts a JOIN scenarios s ON s.id=a.scenario_id "
        "WHERE a.id=? AND a.user_id=?",
        (attempt_id, session["user_id"]),
    ).fetchone()
    if attempt is None:
        abort(404)
    feedback = feedback_for(attempt["expected_signal"], attempt["predicted_signal"], attempt["confidence"])
    return render_template("result.html", attempt=attempt, feedback=feedback)


def trainee_progress(db, trainee_id):
    requirements = db.execute("SELECT * FROM signal_requirements ORDER BY signal").fetchall()
    progress = []
    for requirement in requirements:
        stats = db.execute(
            "SELECT COUNT(*) AS attempts,COALESCE(SUM(COALESCE(instructor_override,matched)),0) AS correct "
            "FROM attempts WHERE user_id=? AND expected_signal=?",
            (trainee_id, requirement["signal"]),
        ).fetchone()
        rate = round(100 * stats["correct"] / stats["attempts"]) if stats["attempts"] else 0
        exact_rate = 100 * stats["correct"] / stats["attempts"] if stats["attempts"] else 0
        ready = stats["attempts"] >= requirement["min_attempts"] and exact_rate >= requirement["min_match_rate"]
        progress.append({"signal": requirement["signal"], "attempts": stats["attempts"], "correct": stats["correct"],
                         "rate": rate, "requirement": requirement, "ready": ready})
    return progress


@app.route("/instructor/attempts/<int:attempt_id>", methods=["GET", "POST"])
@role_required("instructor")
def review_attempt(attempt_id):
    db = database()
    attempt = db.execute(
        "SELECT a.*,u.display_name,s.title FROM attempts a JOIN users u ON u.id=a.user_id "
        "JOIN scenarios s ON s.id=a.scenario_id WHERE a.id=?", (attempt_id,)
    ).fetchone()
    if attempt is None:
        abort(404)
    if request.method == "POST":
        decision = request.form.get("decision")
        feedback = request.form.get("feedback", "").strip()
        if decision not in {"accept_ai", "pass", "needs_practice"} or not feedback:
            flash("Choose a review outcome and add feedback for the trainee.", "error")
            return redirect(url_for("review_attempt", attempt_id=attempt_id))
        override = None if decision == "accept_ai" else int(decision == "pass")
        db.execute(
            "UPDATE attempts SET instructor_override=?,instructor_feedback=?,reviewed_by=?,reviewed_at=? WHERE id=?",
            (override, feedback[:1000], session["user_id"], datetime.now(timezone.utc).isoformat(timespec="seconds"), attempt_id),
        )
        db.execute("DELETE FROM completion_approvals WHERE trainee_id=?", (attempt["user_id"],))
        db.commit()
        flash("Instructor review saved.", "success")
        return redirect(url_for("instructor_trainee", trainee_id=attempt["user_id"]))
    return render_template("review.html", attempt=attempt)


@app.route("/instructor/attempts/<int:attempt_id>/evidence")
@role_required("instructor")
def attempt_evidence(attempt_id):
    attempt = database().execute("SELECT evidence_path FROM attempts WHERE id=?", (attempt_id,)).fetchone()
    if not attempt or not attempt["evidence_path"]:
        abort(404)
    evidence = Path(attempt["evidence_path"]).resolve()
    if ATTEMPT_MEDIA_DIR.resolve() not in evidence.parents or not evidence.is_file():
        abort(404)
    response = send_file(evidence, mimetype="image/jpeg", max_age=0)
    response.headers["Cache-Control"] = "private, no-store"
    return response


@app.route("/instructor/trainees/<int:trainee_id>")
@role_required("instructor")
def instructor_trainee(trainee_id):
    db = database()
    trainee = db.execute("SELECT * FROM users WHERE id=? AND role='trainee'", (trainee_id,)).fetchone()
    if trainee is None:
        abort(404)
    attempts = db.execute(
        "SELECT a.*,COALESCE(a.instructor_override,a.matched) AS effective_match,s.title FROM attempts a JOIN scenarios s ON s.id=a.scenario_id "
        "WHERE a.user_id=? ORDER BY a.id DESC LIMIT 50", (trainee_id,)
    ).fetchall()
    progress = trainee_progress(db, trainee_id)
    pending_review_count = db.execute(
        "SELECT COUNT(*) FROM attempts WHERE user_id=? AND reviewed_at IS NULL", (trainee_id,)
    ).fetchone()[0]
    error_patterns = db.execute(
        "SELECT expected_signal,predicted_signal,COUNT(*) AS occurrences FROM attempts "
        "WHERE user_id=? AND matched=0 GROUP BY expected_signal,predicted_signal "
        "ORDER BY occurrences DESC LIMIT 6", (trainee_id,)
    ).fetchall()
    ready = bool(progress) and all(item["ready"] for item in progress)
    approval = db.execute("SELECT * FROM completion_approvals WHERE trainee_id=?", (trainee_id,)).fetchone()
    return render_template("trainee_detail.html", trainee=trainee, attempts=attempts, progress=progress,
                           error_patterns=error_patterns, ready=ready, pending_review_count=pending_review_count,
                           review_complete=pending_review_count == 0, approval=approval if ready else None)


@app.route("/instructor/requirements", methods=["POST"])
@role_required("instructor")
def update_requirements():
    db = database()
    for signal in EXPECTED_CLASSES:
        try:
            count = int(request.form.get(f"min_attempts_{signal}", ""))
            rate = int(request.form.get(f"min_rate_{signal}", ""))
        except ValueError:
            flash("Enter whole numbers for each signal requirement.", "error")
            return redirect(url_for("instructor_dashboard"))
        if not 1 <= count <= 50 or not 50 <= rate <= 100:
            flash("Attempts must be 1–50 and recognition targets must be 50–100%.", "error")
            return redirect(url_for("instructor_dashboard"))
        db.execute("UPDATE signal_requirements SET min_attempts=?,min_match_rate=? WHERE signal=?", (count, rate, signal))
    db.commit()
    db.execute("DELETE FROM completion_approvals")
    db.commit()
    flash("Readiness criteria updated.", "success")
    return redirect(url_for("instructor_dashboard"))


@app.route("/instructor/trainees/<int:trainee_id>/approve", methods=["POST"])
@role_required("instructor")
def approve_completion(trainee_id):
    db = database()
    trainee = db.execute("SELECT id FROM users WHERE id=? AND role='trainee'", (trainee_id,)).fetchone()
    progress = trainee_progress(db, trainee_id)
    pending_review_count = db.execute(
        "SELECT COUNT(*) FROM attempts WHERE user_id=? AND reviewed_at IS NULL", (trainee_id,)
    ).fetchone()[0]
    if trainee is None or not progress or not all(item["ready"] for item in progress) or pending_review_count:
        flash("Complete all attempt reviews and meet the current readiness criteria before approval.", "error")
        return redirect(url_for("instructor_trainee", trainee_id=trainee_id))
    db.execute(
        "INSERT INTO completion_approvals(trainee_id,approved_by,approved_at,notes) VALUES(?,?,?,?) "
        "ON CONFLICT(trainee_id) DO UPDATE SET approved_by=excluded.approved_by,approved_at=excluded.approved_at,notes=excluded.notes",
        (trainee_id, session["user_id"], datetime.now(timezone.utc).isoformat(timespec="seconds"), request.form.get("notes", "").strip()[:1000]),
    )
    db.commit()
    flash("Completion approval recorded by instructor.", "success")
    return redirect(url_for("instructor_trainee", trainee_id=trainee_id))


@app.route("/instructor")
@role_required("instructor")
def instructor_dashboard():
    db = database()
    trainees = db.execute(
        "SELECT u.id,u.display_name,COUNT(a.id) AS attempts,COALESCE(SUM(COALESCE(a.instructor_override,a.matched)),0) AS correct,"
        "MAX(a.created_at) AS last_attempt FROM users u LEFT JOIN attempts a ON a.user_id=u.id "
        "WHERE u.role='trainee' GROUP BY u.id ORDER BY u.display_name"
    ).fetchall()
    scenarios = db.execute("SELECT * FROM scenarios ORDER BY active DESC,id").fetchall()
    pending_reviews = db.execute(
        "SELECT a.id,a.expected_signal,a.predicted_signal,a.confidence,a.created_at,u.display_name,s.title "
        "FROM attempts a JOIN users u ON u.id=a.user_id JOIN scenarios s ON s.id=a.scenario_id "
        "WHERE a.reviewed_at IS NULL ORDER BY a.id DESC LIMIT 20"
    ).fetchall()
    requirements = db.execute("SELECT * FROM signal_requirements ORDER BY signal").fetchall()
    return render_template("instructor.html", trainees=trainees, scenarios=scenarios,
                           pending_reviews=pending_reviews, requirements=requirements)


@app.route("/instructor/scenarios", methods=["POST"])
@role_required("instructor")
def create_scenario():
    title = request.form.get("title", "").strip()
    description = request.form.get("description", "").strip()
    signal = request.form.get("expected_signal", "")
    if not title or not description or signal not in {"Six", "Out", "Wide", "No_Ball"}:
        flash("Enter a title, description, and valid expected signal.", "error")
        return redirect(url_for("instructor_dashboard"))
    database().execute(
        "INSERT INTO scenarios(title,description,expected_signal,created_by) VALUES(?,?,?,?)",
        (title[:100], description[:500], signal, session["user_id"]),
    )
    database().commit()
    flash("Practice scenario created.", "success")
    return redirect(url_for("instructor_dashboard"))


@app.route("/instructor/trainees", methods=["POST"])
@role_required("instructor")
def create_trainee():
    username = request.form.get("username", "").strip().lower()
    display_name = request.form.get("display_name", "").strip()
    password = request.form.get("password", "")
    if not username or not display_name or len(password) < 8:
        flash("Enter a username, display name, and password with at least 8 characters.", "error")
        return redirect(url_for("instructor_dashboard"))
    try:
        database().execute(
            "INSERT INTO users(username,password_hash,display_name,role) VALUES(?,?,?,'trainee')",
            (username, generate_password_hash(password), display_name[:80]),
        )
        database().commit()
    except sqlite3.IntegrityError:
        flash("That username is already in use.", "error")
        return redirect(url_for("instructor_dashboard"))
    flash("Trainee account created.", "success")
    return redirect(url_for("instructor_dashboard"))


@app.route("/instructor/scenarios/<int:scenario_id>/toggle", methods=["POST"])
@role_required("instructor")
def toggle_scenario(scenario_id):
    db = database()
    scenario = db.execute("SELECT active FROM scenarios WHERE id=?", (scenario_id,)).fetchone()
    if scenario is None:
        abort(404)
    db.execute("UPDATE scenarios SET active=? WHERE id=?", (0 if scenario["active"] else 1, scenario_id))
    db.commit()
    return redirect(url_for("instructor_dashboard"))


@app.errorhandler(403)
def forbidden(_error):
    return render_template("message.html", title="Access denied", message="This page is not available for your account."), 403


@app.errorhandler(413)
def upload_too_large(_error):
    flash("That photo is over the 10 MB upload limit. Choose a smaller image.", "error")
    return redirect(request.referrer or url_for("dashboard"))


from match_module import create_match_blueprint, create_match_schema

app.register_blueprint(create_match_blueprint(database, role_required, predict_frame, MODEL_PATH))

init_db()

if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)), debug=False)
