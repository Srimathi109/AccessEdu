"""AccessEdu local training app for cricket umpire signals."""

from __future__ import annotations

import os
import hmac
import secrets
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
ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
CLASS_LABELS = {
    "No_Ball": "No Ball",
    "Out": "Out",
    "Six": "Six",
    "Wide": "Wide",
}

app = Flask(__name__, template_folder="app/templates", static_folder="app/static")
app.config.update(
    SECRET_KEY=os.environ.get("ACCESSEDU_SECRET_KEY") or secrets.token_hex(32),
    MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
    DATABASE=str(ROOT / "accessedu.sqlite3"),
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
        """
    )
    accounts = [
        ("trainee", generate_password_hash("trainee123"), "Demo Trainee", "trainee"),
        ("instructor", generate_password_hash("instructor123"), "Demo Instructor", "instructor"),
    ]
    db.executemany(
        "INSERT OR IGNORE INTO users(username,password_hash,display_name,role) VALUES(?,?,?,?)",
        accounts,
    )
    db.commit()
    instructor_id = db.execute("SELECT id FROM users WHERE username='instructor'").fetchone()[0]
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
    model = get_model()
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError("The photo could not be opened. Choose a JPG, PNG, or WebP image.")
    result = model(image, imgsz=224, verbose=False)[0]
    probabilities = result.probs
    if probabilities is None:
        raise ValueError("The selected image could not be classified. Choose another photo.")
    class_id = int(probabilities.top1)
    confidence = float(probabilities.top1conf.item())
    return model.names[class_id], confidence, 1


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
        "SELECT a.*,s.title FROM attempts a JOIN scenarios s ON s.id=a.scenario_id "
        "WHERE a.user_id=? ORDER BY a.id DESC LIMIT 8",
        (session["user_id"],),
    ).fetchall()
    totals = db.execute(
        "SELECT COUNT(*) AS total, COALESCE(SUM(matched),0) AS correct FROM attempts WHERE user_id=?",
        (session["user_id"],),
    ).fetchone()
    rate = round(100 * totals["correct"] / totals["total"]) if totals["total"] else 0
    return render_template("trainee.html", scenarios=scenarios, attempts=attempts, totals=totals, rate=rate)


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
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix="." + extension) as temp_file:
            temp_path = temp_file.name
            photo.save(temp_path)
        if os.path.getsize(temp_path) > MAX_UPLOAD_BYTES:
            raise ValueError("That photo is over the 10 MB upload limit. Choose a smaller image.")
        predicted, confidence, frame_count = analyze_image(temp_path)
    except (RuntimeError, ValueError) as exc:
        flash(str(exc), "error")
        return redirect(url_for("practice", scenario_id=scenario_id))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
    expected = scenario["expected_signal"]
    matched = predicted == expected and confidence >= 0.35
    cursor = database().execute(
        "INSERT INTO attempts(user_id,scenario_id,expected_signal,predicted_signal,confidence,matched,frame_count,created_at) "
        "VALUES(?,?,?,?,?,?,?,?)",
        (
            session["user_id"], scenario_id, expected, predicted, confidence, int(matched), frame_count,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ),
    )
    database().commit()
    return redirect(url_for("result", attempt_id=cursor.lastrowid))


@app.route("/result/<int:attempt_id>")
@role_required("trainee")
def result(attempt_id):
    attempt = database().execute(
        "SELECT a.*,s.title FROM attempts a JOIN scenarios s ON s.id=a.scenario_id "
        "WHERE a.id=? AND a.user_id=?",
        (attempt_id, session["user_id"]),
    ).fetchone()
    if attempt is None:
        abort(404)
    feedback = feedback_for(attempt["expected_signal"], attempt["predicted_signal"], attempt["confidence"])
    return render_template("result.html", attempt=attempt, feedback=feedback)


@app.route("/instructor")
@role_required("instructor")
def instructor_dashboard():
    db = database()
    trainees = db.execute(
        "SELECT u.id,u.display_name,COUNT(a.id) AS attempts,COALESCE(SUM(a.matched),0) AS correct,"
        "MAX(a.created_at) AS last_attempt FROM users u LEFT JOIN attempts a ON a.user_id=u.id "
        "WHERE u.role='trainee' GROUP BY u.id ORDER BY u.display_name"
    ).fetchall()
    recent = db.execute(
        "SELECT a.*,u.display_name,s.title FROM attempts a JOIN users u ON u.id=a.user_id "
        "JOIN scenarios s ON s.id=a.scenario_id ORDER BY a.id DESC LIMIT 20"
    ).fetchall()
    scenarios = db.execute("SELECT * FROM scenarios ORDER BY active DESC,id").fetchall()
    return render_template("instructor.html", trainees=trainees, recent=recent, scenarios=scenarios)


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


init_db()

if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)), debug=False)
