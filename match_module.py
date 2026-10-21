"""Isolated match-scoring workflow for AccessEdu Module 2."""

from __future__ import annotations

import os
import secrets
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
from flask import Blueprint, abort, flash, jsonify, redirect, render_template, request, session, url_for
from werkzeug.utils import secure_filename

MATCH_LABELS = {"Six": "SIX", "Out": "OUT", "Wide": "WIDE", "No_Ball": "NO BALL"}
MATCH_SIGNALS = set(MATCH_LABELS)
CONFIDENCE_THRESHOLD = 0.72
STABLE_DETECTIONS = 3
VIDEO_MAX_SECONDS = 120
VIDEO_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


def create_match_schema(db):
    """Create additive Module 2 tables without altering Module 1 records."""
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            overs_limit INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'live' CHECK(status IN ('live','completed')),
            created_by INTEGER NOT NULL REFERENCES users(id),
            created_at TEXT NOT NULL,
            completed_at TEXT
        );
        CREATE TABLE IF NOT EXISTS match_innings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER NOT NULL REFERENCES matches(id),
            innings_no INTEGER NOT NULL,
            batting_team TEXT NOT NULL,
            runs INTEGER NOT NULL DEFAULT 0,
            wickets INTEGER NOT NULL DEFAULT 0,
            legal_balls INTEGER NOT NULL DEFAULT 0,
            wide_runs INTEGER NOT NULL DEFAULT 0,
            no_ball_runs INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'live' CHECK(status IN ('live','completed')),
            created_at TEXT NOT NULL,
            UNIQUE(match_id, innings_no)
        );
        CREATE TABLE IF NOT EXISTS match_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER NOT NULL REFERENCES matches(id),
            innings_id INTEGER NOT NULL REFERENCES match_innings(id),
            signal TEXT NOT NULL,
            confidence REAL,
            source TEXT NOT NULL CHECK(source IN ('live','video','manual')),
            source_offset_seconds REAL,
            client_id TEXT NOT NULL,
            model_version TEXT,
            detected_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','confirmed','rejected','corrected','void')),
            runs_awarded INTEGER NOT NULL DEFAULT 0,
            wickets_awarded INTEGER NOT NULL DEFAULT 0,
            legal_ball INTEGER NOT NULL DEFAULT 0,
            wide_runs INTEGER NOT NULL DEFAULT 0,
            no_ball_runs INTEGER NOT NULL DEFAULT 0,
            scorer_id INTEGER REFERENCES users(id),
            confirmed_at TEXT,
            scorer_note TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_match_events_match ON match_events(match_id, id DESC);
        CREATE INDEX IF NOT EXISTS idx_match_events_repeat ON match_events(match_id, innings_id, signal, detected_at);
        CREATE TABLE IF NOT EXISTS match_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER NOT NULL REFERENCES matches(id),
            innings_id INTEGER NOT NULL REFERENCES match_innings(id),
            client_id TEXT NOT NULL,
            source TEXT NOT NULL CHECK(source IN ('live','video')),
            signal TEXT NOT NULL,
            confidence REAL NOT NULL,
            source_offset_seconds REAL,
            observed_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_match_observations_stream
            ON match_observations(match_id, innings_id, client_id, id DESC);
        CREATE TABLE IF NOT EXISTS match_event_corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER NOT NULL REFERENCES match_events(id),
            changed_by INTEGER NOT NULL REFERENCES users(id),
            old_status TEXT NOT NULL,
            new_status TEXT NOT NULL,
            old_runs INTEGER NOT NULL,
            new_runs INTEGER NOT NULL,
            old_wickets INTEGER NOT NULL,
            new_wickets INTEGER NOT NULL,
            old_legal_ball INTEGER NOT NULL,
            new_legal_ball INTEGER NOT NULL,
            reason TEXT NOT NULL,
            changed_at TEXT NOT NULL
        );
        """
    )


def create_match_blueprint(database, role_required, predict_frame, model_path: Path):
    bp = Blueprint("matches", __name__, url_prefix="/matches")

    def utc_now():
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def get_match(db, match_id):
        match = db.execute("SELECT * FROM matches WHERE id=?", (match_id,)).fetchone()
        if match is None:
            abort(404)
        return match

    def active_innings(db, match_id):
        innings = db.execute(
            "SELECT * FROM match_innings WHERE match_id=? AND status='live' ORDER BY innings_no DESC LIMIT 1",
            (match_id,),
        ).fetchone()
        if innings is None:
            abort(409, description="This match has no active innings.")
        return innings

    def model_version():
        try:
            return f"{model_path.name}:{int(model_path.stat().st_mtime)}"
        except OSError:
            return model_path.name

    def proposed_score(signal):
        if signal == "Six":
            return 6, 0, 1, 0, 0
        if signal == "Out":
            return 0, 1, 1, 0, 0
        if signal == "Wide":
            return 1, 0, 0, 1, 0
        if signal == "No_Ball":
            return 1, 0, 0, 0, 1
        raise ValueError("Unsupported signal")

    def maybe_propose(db, match_id, innings_id, client_id, source, signal, confidence, offset):
        if confidence < CONFIDENCE_THRESHOLD:
            return None
        observations = db.execute(
            "SELECT signal,confidence,observed_at FROM match_observations "
            "WHERE match_id=? AND innings_id=? AND client_id=? AND source=? ORDER BY id DESC LIMIT ?",
            (match_id, innings_id, client_id, source, STABLE_DETECTIONS),
        ).fetchall()
        if len(observations) != STABLE_DETECTIONS or any(
            row["signal"] != signal or row["confidence"] < CONFIDENCE_THRESHOLD for row in observations
        ):
            return None

        if source == "video":
            duplicate = db.execute(
                "SELECT id FROM match_events WHERE match_id=? AND innings_id=? AND client_id=? AND source='video' "
                "AND signal=? AND source_offset_seconds BETWEEN ? AND ? LIMIT 1",
                (match_id, innings_id, client_id, signal, max(0, offset - 8), offset + 8),
            ).fetchone()
        else:
            cutoff = (datetime.now(timezone.utc) - timedelta(seconds=15)).isoformat(timespec="seconds")
            duplicate = db.execute(
                "SELECT id FROM match_events WHERE match_id=? AND innings_id=? AND client_id=? AND signal=? "
                "AND detected_at>=? LIMIT 1",
                (match_id, innings_id, client_id, signal, cutoff),
            ).fetchone()
        if duplicate:
            return None

        runs, wickets, legal_ball, wide_runs, no_ball_runs = proposed_score(signal)
        cursor = db.execute(
            "INSERT INTO match_events(match_id,innings_id,signal,confidence,source,source_offset_seconds,client_id,model_version,detected_at,runs_awarded,wickets_awarded,legal_ball,wide_runs,no_ball_runs) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (match_id, innings_id, signal, confidence, source, offset, client_id, model_version(), utc_now(),
             runs, wickets, legal_ball, wide_runs, no_ball_runs),
        )
        return cursor.lastrowid

    def process_frame(db, match_id, innings_id, client_id, source, frame, offset=None):
        signal, confidence = predict_frame(frame)
        if signal not in MATCH_SIGNALS:
            return signal, confidence, None
        db.execute(
            "INSERT INTO match_observations(match_id,innings_id,client_id,source,signal,confidence,source_offset_seconds,observed_at) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (match_id, innings_id, client_id, source, signal, confidence, offset, utc_now()),
        )
        event_id = maybe_propose(db, match_id, innings_id, client_id, source, signal, confidence, offset or 0)
        return signal, confidence, event_id

    @bp.route("", methods=["GET", "POST"])
    @role_required("instructor")
    def index():
        db = database()
        if request.method == "POST":
            title = request.form.get("title", "").strip()[:100]
            home_team = request.form.get("home_team", "").strip()[:80]
            away_team = request.form.get("away_team", "").strip()[:80]
            try:
                overs_limit = int(request.form.get("overs_limit", ""))
            except ValueError:
                overs_limit = 0
            if not title or not home_team or not away_team or home_team.casefold() == away_team.casefold() or not 1 <= overs_limit <= 50:
                flash("Enter a match title, two different teams, and an overs limit from 1 to 50.", "error")
                return redirect(url_for("matches.index"))
            cursor = db.execute(
                "INSERT INTO matches(title,home_team,away_team,overs_limit,created_by,created_at) VALUES(?,?,?,?,?,?)",
                (title, home_team, away_team, overs_limit, session["user_id"], utc_now()),
            )
            match_id = cursor.lastrowid
            db.execute(
                "INSERT INTO match_innings(match_id,innings_no,batting_team,created_at) VALUES(?,1,?,?)",
                (match_id, home_team, utc_now()),
            )
            db.commit()
            return redirect(url_for("matches.detail", match_id=match_id))

        matches = db.execute(
            "SELECT m.*,i.runs,i.wickets,i.legal_balls,i.batting_team FROM matches m "
            "LEFT JOIN match_innings i ON i.match_id=m.id AND i.status='live' "
            "ORDER BY CASE m.status WHEN 'live' THEN 0 ELSE 1 END,m.id DESC"
        ).fetchall()
        return render_template("matches.html", matches=matches)

    @bp.route("/<int:match_id>")
    @role_required("instructor")
    def detail(match_id):
        db = database()
        match = get_match(db, match_id)
        innings = db.execute(
            "SELECT * FROM match_innings WHERE match_id=? ORDER BY innings_no", (match_id,)
        ).fetchall()
        current = next((item for item in innings if item["status"] == "live"), innings[-1] if innings else None)
        events = db.execute(
            "SELECT e.*,u.display_name AS scorer_name FROM match_events e LEFT JOIN users u ON u.id=e.scorer_id "
            "WHERE e.match_id=? ORDER BY e.id DESC LIMIT 80", (match_id,)
        ).fetchall()
        corrections = db.execute(
            "SELECT c.*,u.display_name FROM match_event_corrections c JOIN users u ON u.id=c.changed_by "
            "JOIN match_events e ON e.id=c.event_id WHERE e.match_id=? ORDER BY c.id DESC LIMIT 30", (match_id,)
        ).fetchall()
        pending_count = db.execute(
            "SELECT COUNT(*) FROM match_events WHERE match_id=? AND status='pending'", (match_id,)
        ).fetchone()[0]
        return render_template("match_detail.html", match=match, innings=innings, current=current, events=events,
                               corrections=corrections, pending_count=pending_count,
                               confidence_threshold=CONFIDENCE_THRESHOLD)

    @bp.route("/<int:match_id>/detect", methods=["POST"])
    @role_required("instructor")
    def detect(match_id):
        db = database()
        match = get_match(db, match_id)
        if match["status"] != "live":
            return jsonify(error="This match is complete."), 409
        innings = active_innings(db, match_id)
        frame_file = request.files.get("frame")
        if not frame_file:
            return jsonify(error="No camera frame was received."), 400
        frame_bytes = frame_file.read(2 * 1024 * 1024 + 1)
        if len(frame_bytes) > 2 * 1024 * 1024:
            return jsonify(error="Camera frame exceeds 2 MB."), 413
        image = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return jsonify(error="The camera frame could not be decoded."), 400
        try:
            client_id = request.form.get("client_id", "")[:80]
            if not client_id:
                return jsonify(error="Camera session identifier is missing."), 400
            signal, confidence, event_id = process_frame(db, match_id, innings["id"], client_id, "live", image)
            db.commit()
        except (RuntimeError, ValueError) as exc:
            db.rollback()
            return jsonify(error=str(exc)), 503
        return jsonify(signal=signal, label=MATCH_LABELS.get(signal, signal), confidence=confidence,
                       candidate=confidence >= CONFIDENCE_THRESHOLD, event_id=event_id,
                       event_created=event_id is not None)

    @bp.route("/<int:match_id>/video", methods=["POST"])
    @role_required("instructor")
    def process_video(match_id):
        db = database()
        match = get_match(db, match_id)
        if match["status"] != "live":
            flash("This match has already been completed.", "error")
            return redirect(url_for("matches.detail", match_id=match_id))
        innings = active_innings(db, match_id)
        upload = request.files.get("video")
        if not upload or not upload.filename:
            flash("Choose a recorded match clip first.", "error")
            return redirect(url_for("matches.detail", match_id=match_id))
        filename = secure_filename(upload.filename)
        extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if extension not in VIDEO_EXTENSIONS:
            flash("Choose an MP4, MOV, AVI, MKV or WebM clip.", "error")
            return redirect(url_for("matches.detail", match_id=match_id))
        temp_path = None
        created_events = set()
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix="." + extension) as temp_file:
                temp_path = temp_file.name
                upload.save(temp_file)
            if os.path.getsize(temp_path) > 10 * 1024 * 1024:
                raise ValueError("Match clips are limited to 10 MB.")
            capture = cv2.VideoCapture(temp_path)
            try:
                fps = capture.get(cv2.CAP_PROP_FPS) or 25
                frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                duration = frame_total / fps if frame_total > 0 else 0
                if not capture.isOpened() or duration <= 0:
                    raise ValueError("The clip could not be opened. Try a supported MP4 or MOV file.")
                if duration > VIDEO_MAX_SECONDS:
                    raise ValueError("Recorded clips are limited to 120 seconds.")
                sample_count = min(int(duration) + 1, 120)
                client_id = secrets.token_urlsafe(16)
                for sample_index in range(sample_count):
                    offset = min(float(sample_index), max(0.0, duration - 0.01))
                    capture.set(cv2.CAP_PROP_POS_MSEC, offset * 1000)
                    ok, frame = capture.read()
                    if not ok:
                        continue
                    _signal, _confidence, event_id = process_frame(
                        db, match_id, innings["id"], client_id, "video", frame, offset
                    )
                    if event_id:
                        created_events.add(event_id)
                db.commit()
            finally:
                capture.release()
        except (OSError, RuntimeError, ValueError) as exc:
            db.rollback()
            flash(str(exc), "error")
            return redirect(url_for("matches.detail", match_id=match_id))
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
        flash(f"Clip analyzed. {len(created_events)} event proposal(s) added for scorer review.", "success")
        return redirect(url_for("matches.detail", match_id=match_id))

    @bp.route("/<int:match_id>/manual-event", methods=["POST"])
    @role_required("instructor")
    def manual_event(match_id):
        db = database()
        match = get_match(db, match_id)
        if match["status"] != "live":
            abort(409, description="This match is complete.")
        innings = active_innings(db, match_id)
        signal = request.form.get("signal", "")
        if signal not in MATCH_SIGNALS:
            flash("Choose one of the supported umpire signals.", "error")
            return redirect(url_for("matches.detail", match_id=match_id))
        runs, wickets, legal_ball, wide_runs, no_ball_runs = proposed_score(signal)
        if legal_ball and innings["legal_balls"] >= match["overs_limit"] * 6:
            flash("The innings has reached its overs limit. End it before scoring more legal deliveries.", "error")
            return redirect(url_for("matches.detail", match_id=match_id))
        if wickets and innings["wickets"] >= 10:
            flash("The innings has reached ten wickets.", "error")
            return redirect(url_for("matches.detail", match_id=match_id))
        db.execute(
            "INSERT INTO match_events(match_id,innings_id,signal,source,client_id,detected_at,status,runs_awarded,wickets_awarded,legal_ball,wide_runs,no_ball_runs,scorer_id,confirmed_at) "
            "VALUES(?,?,?,'manual',?,?,'confirmed',?,?,?,?,?,?,?)",
            (match_id, innings["id"], signal, f"manual-{session['user_id']}", utc_now(), runs, wickets, legal_ball,
             wide_runs, no_ball_runs, session["user_id"], utc_now()),
        )
        db.execute(
            "UPDATE match_innings SET runs=runs+?,wickets=wickets+?,legal_balls=legal_balls+?,wide_runs=wide_runs+?,no_ball_runs=no_ball_runs+? WHERE id=?",
            (runs, wickets, legal_ball, wide_runs, no_ball_runs, innings["id"]),
        )
        db.commit()
        flash("Manually entered match event recorded.", "success")
        return redirect(url_for("matches.detail", match_id=match_id))

    @bp.route("/events/<int:event_id>/confirm", methods=["POST"])
    @role_required("instructor")
    def confirm_event(event_id):
        db = database()
        event = db.execute("SELECT * FROM match_events WHERE id=?", (event_id,)).fetchone()
        if event is None:
            abort(404)
        match = get_match(db, event["match_id"])
        if event["status"] != "pending" or match["status"] != "live":
            abort(409, description="This proposal is no longer awaiting confirmation.")
        try:
            additional_runs = int(request.form.get("additional_runs", "0"))
        except ValueError:
            additional_runs = -1
        if not 0 <= additional_runs <= 6:
            flash("Additional runs must be from 0 to 6.", "error")
            return redirect(url_for("matches.detail", match_id=event["match_id"]))
        innings = db.execute("SELECT * FROM match_innings WHERE id=?", (event["innings_id"],)).fetchone()
        if innings is None or innings["status"] != "live":
            abort(409, description="This innings has ended.")
        if event["legal_ball"] and innings["legal_balls"] >= match["overs_limit"] * 6:
            flash("The innings has reached its overs limit. End it before scoring more legal deliveries.", "error")
            return redirect(url_for("matches.detail", match_id=event["match_id"]))
        if event["wickets_awarded"] and innings["wickets"] >= 10:
            flash("The innings has reached ten wickets.", "error")
            return redirect(url_for("matches.detail", match_id=event["match_id"]))
        runs = event["runs_awarded"] + additional_runs
        wide_runs = event["wide_runs"] + (additional_runs if event["signal"] == "Wide" else 0)
        no_ball_runs = event["no_ball_runs"] + (additional_runs if event["signal"] == "No_Ball" else 0)
        db.execute(
            "UPDATE match_events SET status='confirmed',runs_awarded=?,wide_runs=?,no_ball_runs=?,scorer_id=?,confirmed_at=?,scorer_note=? WHERE id=? AND status='pending'",
            (runs, wide_runs, no_ball_runs, session["user_id"], utc_now(), request.form.get("note", "").strip()[:500], event_id),
        )
        db.execute(
            "UPDATE match_innings SET runs=runs+?,wickets=wickets+?,legal_balls=legal_balls+?,wide_runs=wide_runs+?,no_ball_runs=no_ball_runs+? WHERE id=?",
            (runs, event["wickets_awarded"], event["legal_ball"], wide_runs, no_ball_runs, event["innings_id"]),
        )
        db.commit()
        flash("Event confirmed and scoreboard updated.", "success")
        return redirect(url_for("matches.detail", match_id=event["match_id"]))

    @bp.route("/events/<int:event_id>/reject", methods=["POST"])
    @role_required("instructor")
    def reject_event(event_id):
        db = database()
        event = db.execute("SELECT * FROM match_events WHERE id=?", (event_id,)).fetchone()
        if event is None:
            abort(404)
        if event["status"] != "pending":
            abort(409, description="Only pending proposals can be rejected.")
        db.execute(
            "UPDATE match_events SET status='rejected',scorer_id=?,confirmed_at=?,scorer_note=? WHERE id=?",
            (session["user_id"], utc_now(), request.form.get("note", "").strip()[:500], event_id),
        )
        db.commit()
        flash("Event proposal rejected. No score was changed.", "success")
        return redirect(url_for("matches.detail", match_id=event["match_id"]))

    @bp.route("/events/<int:event_id>/correct", methods=["POST"])
    @role_required("instructor")
    def correct_event(event_id):
        db = database()
        event = db.execute("SELECT * FROM match_events WHERE id=?", (event_id,)).fetchone()
        if event is None:
            abort(404)
        if event["status"] not in {"confirmed", "corrected"}:
            abort(409, description="Only confirmed events can be corrected.")
        reason = request.form.get("reason", "").strip()
        action = request.form.get("action", "correct")
        if not reason:
            flash("Add a reason for the correction.", "error")
            return redirect(url_for("matches.detail", match_id=event["match_id"]))
        if action == "void":
            new_status, new_runs, new_wickets, new_legal_ball, new_wides, new_no_balls = "void", 0, 0, 0, 0, 0
        else:
            try:
                new_runs = int(request.form.get("runs_awarded", ""))
                new_wickets = int(request.form.get("wickets_awarded", ""))
                new_legal_ball = int(request.form.get("legal_ball", ""))
            except ValueError:
                new_runs, new_wickets, new_legal_ball = -1, -1, -1
            if not 0 <= new_runs <= 20 or not 0 <= new_wickets <= 1 or new_legal_ball not in {0, 1}:
                flash("Enter valid corrected runs, wickets, and ball status.", "error")
                return redirect(url_for("matches.detail", match_id=event["match_id"]))
            new_wides = new_runs if event["signal"] == "Wide" else (1 if event["status"] != "void" and event["wide_runs"] else 0)
            new_no_balls = new_runs if event["signal"] == "No_Ball" else (1 if event["status"] != "void" and event["no_ball_runs"] else 0)
            new_status = "corrected"

        db.execute(
            "INSERT INTO match_event_corrections(event_id,changed_by,old_status,new_status,old_runs,new_runs,old_wickets,new_wickets,old_legal_ball,new_legal_ball,reason,changed_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (event_id, session["user_id"], event["status"], new_status, event["runs_awarded"], new_runs,
             event["wickets_awarded"], new_wickets, event["legal_ball"], new_legal_ball, reason[:500], utc_now()),
        )
        db.execute(
            "UPDATE match_innings SET runs=runs+?,wickets=wickets+?,legal_balls=legal_balls+?,wide_runs=wide_runs+?,no_ball_runs=no_ball_runs+? WHERE id=?",
            (new_runs - event["runs_awarded"], new_wickets - event["wickets_awarded"],
             new_legal_ball - event["legal_ball"], new_wides - event["wide_runs"],
             new_no_balls - event["no_ball_runs"], event["innings_id"]),
        )
        updated = db.execute("SELECT runs,wickets,legal_balls,wide_runs,no_ball_runs FROM match_innings WHERE id=?", (event["innings_id"],)).fetchone()
        if any(value < 0 for value in updated):
            db.rollback()
            flash("Correction would make the scoreboard negative. No changes were saved.", "error")
            return redirect(url_for("matches.detail", match_id=event["match_id"]))
        db.execute(
            "UPDATE match_events SET status=?,runs_awarded=?,wickets_awarded=?,legal_ball=?,wide_runs=?,no_ball_runs=?,scorer_id=?,confirmed_at=?,scorer_note=? WHERE id=?",
            (new_status, new_runs, new_wickets, new_legal_ball, new_wides, new_no_balls,
             session["user_id"], utc_now(), reason[:500], event_id),
        )
        db.commit()
        flash("Event correction saved in the match history.", "success")
        return redirect(url_for("matches.detail", match_id=event["match_id"]))

    @bp.route("/<int:match_id>/end-innings", methods=["POST"])
    @role_required("instructor")
    def end_innings(match_id):
        db = database()
        match = get_match(db, match_id)
        innings = active_innings(db, match_id)
        pending = db.execute("SELECT COUNT(*) FROM match_events WHERE innings_id=? AND status='pending'", (innings["id"],)).fetchone()[0]
        if pending:
            flash("Confirm or reject all pending proposals before ending the innings.", "error")
            return redirect(url_for("matches.detail", match_id=match_id))
        db.execute("UPDATE match_innings SET status='completed' WHERE id=?", (innings["id"],))
        if innings["innings_no"] == 1:
            db.execute(
                "INSERT INTO match_innings(match_id,innings_no,batting_team,created_at) VALUES(?,2,?,?)",
                (match_id, match["away_team"], utc_now()),
            )
            flash("First innings completed. Second innings started.", "success")
        else:
            db.execute("UPDATE matches SET status='completed',completed_at=? WHERE id=?", (utc_now(), match_id))
            flash("Match completed.", "success")
        db.commit()
        return redirect(url_for("matches.detail", match_id=match_id))

    return bp
