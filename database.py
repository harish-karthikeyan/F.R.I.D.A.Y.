
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "friday.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                age INTEGER NOT NULL,
                weight REAL NOT NULL,
                height REAL NOT NULL,
                goal_weight REAL NOT NULL,
                diet_goal TEXT NOT NULL DEFAULT 'Maintain Weight',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS food_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id INTEGER NOT NULL,
                food_name TEXT NOT NULL,
                quantity REAL NOT NULL,
                unit TEXT NOT NULL DEFAULT 'g',
                meal_type TEXT NOT NULL DEFAULT 'Snack',
                calories REAL NOT NULL DEFAULT 0,
                protein REAL NOT NULL DEFAULT 0,
                carbohydrates REAL NOT NULL DEFAULT 0,
                fat REAL NOT NULL DEFAULT 0,
                fibre REAL NOT NULL DEFAULT 0,
                logged_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (profile_id)
                    REFERENCES profiles(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS dietary_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id INTEGER NOT NULL UNIQUE,
                preferences TEXT DEFAULT '',
                allergies TEXT DEFAULT '',
                restrictions TEXT DEFAULT '',
                FOREIGN KEY (profile_id)
                    REFERENCES profiles(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS exercise_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id INTEGER NOT NULL,
                exercise_name TEXT NOT NULL,
                duration_minutes INTEGER NOT NULL DEFAULT 0,
                calories_burned REAL NOT NULL DEFAULT 0,
                activity_date TEXT NOT NULL DEFAULT (date('now', 'localtime')),
                FOREIGN KEY (profile_id)
                    REFERENCES profiles(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS daily_streaks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id INTEGER NOT NULL,
                activity_date TEXT NOT NULL,
                UNIQUE(profile_id, activity_date),
                FOREIGN KEY (profile_id)
                    REFERENCES profiles(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_food_logs_profile_date
                ON food_logs(profile_id, logged_at);

            CREATE INDEX IF NOT EXISTS idx_exercise_logs_profile_date
                ON exercise_logs(profile_id, activity_date);
        """)


def create_profile(
    name,
    age,
    weight,
    height,
    goal_weight,
    diet_goal="Maintain Weight",
):
    name = name.strip()

    if not name:
        raise ValueError("Name cannot be empty.")

    if age <= 0 or weight <= 0 or height <= 0 or goal_weight <= 0:
        raise ValueError("Age, weight, height and goal weight must be positive.")

    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO profiles (
                name, age, weight, height, goal_weight, diet_goal
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            name, age, weight, height, goal_weight, diet_goal
        ))

        profile_id = cursor.lastrowid

        conn.execute("""
            INSERT INTO dietary_preferences (profile_id)
            VALUES (?)
        """, (profile_id,))

        return profile_id


def get_profiles():
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM profiles ORDER BY name
        """).fetchall()

        return [dict(row) for row in rows]


def get_profile(profile_id):
    with get_connection() as conn:
        row = conn.execute("""
            SELECT * FROM profiles WHERE id = ?
        """, (profile_id,)).fetchone()

        return dict(row) if row else None


def add_food_log(
    profile_id,
    food_name,
    quantity,
    unit,
    meal_type,
    calories,
    protein,
    carbohydrates,
    fat,
    fibre,
):
    if quantity <= 0:
        raise ValueError("Quantity must be greater than zero.")

    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO food_logs (
                profile_id, food_name, quantity, unit, meal_type,
                calories, protein, carbohydrates, fat, fibre
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            profile_id, food_name, quantity, unit, meal_type,
            calories, protein, carbohydrates, fat, fibre
        ))

        return cursor.lastrowid


def get_food_logs(profile_id, selected_date=None):
    with get_connection() as conn:
        if selected_date is None:
            rows = conn.execute("""
                SELECT * FROM food_logs
                WHERE profile_id = ?
                ORDER BY logged_at DESC, id DESC
            """, (profile_id,)).fetchall()
        else:
            rows = conn.execute("""
                SELECT * FROM food_logs
                WHERE profile_id = ?
                  AND date(logged_at, 'localtime') = ?
                ORDER BY logged_at DESC, id DESC
            """, (profile_id, str(selected_date))).fetchall()

        return [dict(row) for row in rows]


def delete_food_log(log_id, profile_id):
    with get_connection() as conn:
        conn.execute("""
            DELETE FROM food_logs
            WHERE id = ? AND profile_id = ?
        """, (log_id, profile_id))


def save_dietary_preferences(
    profile_id,
    preferences,
    allergies,
    restrictions,
):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO dietary_preferences (
                profile_id, preferences, allergies, restrictions
            )
            VALUES (?, ?, ?, ?)
            ON CONFLICT(profile_id) DO UPDATE SET
                preferences = excluded.preferences,
                allergies = excluded.allergies,
                restrictions = excluded.restrictions
        """, (
            profile_id, preferences, allergies, restrictions
        ))


def get_dietary_preferences(profile_id):
    with get_connection() as conn:
        row = conn.execute("""
            SELECT * FROM dietary_preferences
            WHERE profile_id = ?
        """, (profile_id,)).fetchone()

        return dict(row) if row else None


def add_exercise_log(
    profile_id,
    exercise_name,
    duration_minutes,
    calories_burned,
    activity_date=None,
):
    if duration_minutes < 0 or calories_burned < 0:
        raise ValueError("Exercise duration and calories burned cannot be negative.")

    with get_connection() as conn:
        if activity_date is None:
            conn.execute("""
                INSERT INTO exercise_logs (
                    profile_id, exercise_name,
                    duration_minutes, calories_burned
                )
                VALUES (?, ?, ?, ?)
            """, (
                profile_id, exercise_name,
                duration_minutes, calories_burned
            ))
        else:
            conn.execute("""
                INSERT INTO exercise_logs (
                    profile_id, exercise_name,
                    duration_minutes, calories_burned, activity_date
                )
                VALUES (?, ?, ?, ?, ?)
            """, (
                profile_id, exercise_name,
                duration_minutes, calories_burned, str(activity_date)
            ))


def get_exercise_logs(profile_id, selected_date=None):
    with get_connection() as conn:
        if selected_date is None:
            rows = conn.execute("""
                SELECT * FROM exercise_logs
                WHERE profile_id = ?
                ORDER BY activity_date DESC, id DESC
            """, (profile_id,)).fetchall()
        else:
            rows = conn.execute("""
                SELECT * FROM exercise_logs
                WHERE profile_id = ? AND activity_date = ?
                ORDER BY id DESC
            """, (profile_id, str(selected_date))).fetchall()

        return [dict(row) for row in rows]


def mark_daily_activity(profile_id, activity_date=None):
    with get_connection() as conn:
        if activity_date is None:
            conn.execute("""
                INSERT OR IGNORE INTO daily_streaks (
                    profile_id, activity_date
                )
                VALUES (?, date('now', 'localtime'))
            """, (profile_id,))
        else:
            conn.execute("""
                INSERT OR IGNORE INTO daily_streaks (
                    profile_id, activity_date
                )
                VALUES (?, ?)
            """, (profile_id, str(activity_date)))


def get_streak(profile_id):
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT activity_date
            FROM daily_streaks
            WHERE profile_id = ?
            ORDER BY activity_date DESC
        """, (profile_id,)).fetchall()

    dates = [row["activity_date"] for row in rows]

    if not dates:
        return 0

    from datetime import date, timedelta

    today = date.today()
    latest = date.fromisoformat(dates[0])

    if latest not in (today, today - timedelta(days=1)):
        return 0

    streak = 0
    expected = latest

    for recorded in dates:
        current = date.fromisoformat(recorded)

        if current == expected:
            streak += 1
            expected -= timedelta(days=1)
        elif current < expected:
            break

    return streak


init_db()
