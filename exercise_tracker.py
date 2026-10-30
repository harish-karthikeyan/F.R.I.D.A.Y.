
import sqlite3
from datetime import date, timedelta


def log_exercise(
    db_path,
    profile_id,
    exercise_name,
    duration_minutes,
    calories_burned,
):
    if not exercise_name.strip():
        raise ValueError("Exercise name cannot be empty.")

    if duration_minutes <= 0:
        raise ValueError("Duration must be greater than zero.")

    if calories_burned < 0:
        raise ValueError("Calories burned cannot be negative.")

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO exercise_logs (
                profile_id,
                exercise_name,
                duration_minutes,
                calories_burned,
                activity_date
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                profile_id,
                exercise_name.strip(),
                duration_minutes,
                calories_burned,
                date.today().isoformat(),
            ),
        )


def get_daily_exercise(db_path, profile_id, activity_date=None):
    activity_date = activity_date or date.today()

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row

        rows = conn.execute(
            """
            SELECT exercise_name, duration_minutes, calories_burned,
                   activity_date
            FROM exercise_logs
            WHERE profile_id = ? AND activity_date = ?
            ORDER BY id DESC
            """,
            (profile_id, activity_date.isoformat()),
        ).fetchall()

        return [dict(row) for row in rows]


def get_daily_calories_burned(db_path, profile_id, activity_date=None):
    records = get_daily_exercise(db_path, profile_id, activity_date)

    return sum(record["calories_burned"] for record in records)


def get_activity_streak(db_path, profile_id):
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT activity_date
            FROM exercise_logs
            WHERE profile_id = ?
            ORDER BY activity_date DESC
            """,
            (profile_id,),
        ).fetchall()

    recorded_dates = {
        date.fromisoformat(row[0]) for row in rows
    }

    today = date.today()

    # A streak can include today or yesterday.
    current = today if today in recorded_dates else today - timedelta(days=1)

    if current not in recorded_dates:
        return 0

    streak = 0

    while current in recorded_dates:
        streak += 1
        current -= timedelta(days=1)

    return streak
