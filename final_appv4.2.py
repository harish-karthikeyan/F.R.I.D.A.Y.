
import base64
from contextlib import contextmanager
import hashlib
import hmac
import io
import secrets
import sqlite3
from pathlib import Path
from datetime import date, timedelta

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

import pandas as pd
import streamlit as st


# ============================================================
# F.R.I.D.A.Y. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "friday_email.db"
FONT_PATH = BASE_DIR / "assets" / "MolganRegular.otf"


MODEL_NAME = "therealcyberlord/vit-indian-food"

@st.cache_resource
def load_food_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()

    return processor, model, device


def predict_food(image):
    processor, model, device = load_food_model()

    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]
        top_results = torch.topk(probabilities, k=5)

    predictions = []
    for score, index in zip(top_results.values, top_results.indices):
        label = model.config.id2label[index.item()]
        predictions.append((str(label), float(score.item())))

    return predictions

st.set_page_config(
    page_title="F.R.I.D.A.Y.ai",
    page_icon="🤸🏼",
    layout="wide",
)

TODAY = date.today()

FOODS = {
    "Almonds": (579, 21.2, 21.6, 49.9, 12.5),
    "Apple": (52, 0.3, 13.8, 0.2, 2.4),
    "Banana": (89, 1.1, 22.8, 0.3, 2.6),
    "Boiled Egg": (155, 12.6, 1.1, 10.6, 0),
    "Brown Rice, Cooked": (123, 2.7, 25.6, 1, 1.6),
    "Chapati": (297, 9.8, 46.1, 7.5, 7),
    "Chicken Breast": (165, 31, 0, 3.6, 0),
    "Curd": (61, 3.5, 4.7, 3.3, 0),
    "Dosa": (168, 4, 29, 3.5, 1),
    "Idli": (146, 4, 29, 0.5, 1),
    "Milk": (61, 3.2, 4.8, 3.3, 0),
    "Oats, Dry": (389, 16.9, 66.3, 6.9, 10.6),
    "Orange": (47, 0.9, 11.8, 0.1, 2.4),
    "Paneer": (265, 18.3, 1.2, 20.8, 0),
    "Peanuts": (567, 25.8, 16.1, 49.2, 8.5),
    "Potato, Boiled": (87, 1.9, 20.1, 0.1, 1.8),
    "Rice, Cooked": (130, 2.7, 28.2, 0.3, 0.4),
    "Sambar": (65, 3.5, 10, 1.5, 3),
    "Spinach": (23, 2.9, 3.6, 0.4, 2.2),
    "Sweet Potato": (86, 1.6, 20.1, 0.1, 3),
    "Tomato": (18, 0.9, 3.9, 0.2, 1.2),
    "Vegetable Salad": (35, 1.5, 7, 0.3, 2.5),
    "Wheat Bread": (247, 13, 41, 4.2, 7),
    "Idiyappam": (150, 2.5, 32, 0.4, 1),
    "Pongal": (150, 4, 24, 4, 2),
    "Puttu": (180, 4, 35, 2, 3),
    "Ragi Dosa": (150, 4, 28, 2, 3),
    "Ragi Porridge": (95, 3, 18, 1.5, 2.5),
    "Rasam": (30, 1.5, 5, 0.5, 1),
    "Upma": (150, 4, 24, 4, 2),
    "Vegetable Kurma": (120, 3, 12, 6, 3),
    "Dal, Cooked": (116, 9, 20, 0.4, 7.9),
    "Fish, Cooked": (160, 26, 0, 5, 0),
    "Tofu": (76, 8, 1.9, 4.8, 0.3),
    "Broccoli": (34, 2.8, 6.6, 0.4, 2.6),
    "Carrot": (41, 0.9, 9.6, 0.2, 2.8),
    "Cucumber": (15, 0.7, 3.6, 0.1, 0.5),
    "Papaya": (43, 0.5, 10.8, 0.3, 1.7),
    "Guava": (68, 2.6, 14.3, 1, 5.4),
    "Mango": (60, 0.8, 15, 0.4, 1.6),
    "Watermelon": (30, 0.6, 7.6, 0.2, 0.4),
    "Chickpeas, Cooked": (164, 8.9, 27.4, 2.6, 7.6),
    "Kidney Beans, Cooked": (127, 8.7, 22.8, 0.5, 6.4),
    "Quinoa, Cooked": (120, 4.4, 21.3, 1.9, 2.8),
    "Sweet Corn": (96, 3.4, 21, 1.5, 2.4),
    "Olive Oil": (884, 0, 0, 100, 0),
    "Coconut": (354, 3.3, 15.2, 33.5, 9),
    "Dates": (282, 2.5, 75, 0.4, 8),
    "Raisins": (299, 3.1, 79.2, 0.5, 3.7),
    "Dark Chocolate": (598, 7.8, 45.9, 42.6, 10.9),
}

MEALS = ["Breakfast", "Lunch", "Dinner", "Snack"]

EXERCISES = {
    "Walking": 3.5,
    "Brisk Walking": 4.5,
    "Running": 8,
    "Cycling": 6.5,
    "Yoga": 2.5,
    "Strength Training": 5,
    "Skipping": 10,
    "Swimming": 7,
    "Dancing": 5,
    "Stretching": 2,
}


# ============================================================
# DATABASE
# ============================================================

def db():
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    return conn


def columns(conn, table):
    return {
        row["name"]
        for row in conn.execute(f'PRAGMA table_info("{table}")')
    }


def add_missing_columns(conn, table, definitions):
    existing = columns(conn, table)
    for name, definition in definitions.items():
        if name not in existing:
            conn.execute(
                f'ALTER TABLE "{table}" ADD COLUMN "{name}" {definition}'
            )


def initialize_db():
    """Create a fresh email-keyed database schema if it does not exist."""
    with db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS profiles (
                email TEXT PRIMARY KEY COLLATE NOCASE,
                name TEXT NOT NULL,
                age INTEGER DEFAULT 22,
                weight REAL DEFAULT 60,
                height REAL DEFAULT 165,
                goal_weight REAL DEFAULT 60,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS food_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_email TEXT NOT NULL,
                food_name TEXT DEFAULT 'Unknown food',
                quantity REAL DEFAULT 100,
                meal TEXT DEFAULT 'Snack',
                calories REAL DEFAULT 0,
                protein REAL DEFAULT 0,
                carbs REAL DEFAULT 0,
                fat REAL DEFAULT 0,
                fibre REAL DEFAULT 0,
                log_date TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS dietary_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                diet_type TEXT DEFAULT 'No restriction',
                allergies TEXT DEFAULT '',
                restrictions TEXT DEFAULT '',
                goal TEXT DEFAULT 'Maintain weight'
            );

            CREATE TABLE IF NOT EXISTS exercise_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_email TEXT NOT NULL,
                exercise TEXT DEFAULT 'Walking',
                duration INTEGER DEFAULT 30,
                calories_burned REAL DEFAULT 0,
                activity_date TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS daily_streaks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_email TEXT NOT NULL COLLATE NOCASE,
                activity_date TEXT NOT NULL,
                UNIQUE(profile_email, activity_date)
            );
        """)
        add_missing_columns(conn, "profiles", {
            "gender": "TEXT DEFAULT 'Male'",
            "cycle_length": "INTEGER DEFAULT 28",
            "period_length": "INTEGER DEFAULT 5",
            "last_period_start": "TEXT DEFAULT ''",
            "password_hash": "TEXT DEFAULT ''",
        })
        add_missing_columns(conn, "dietary_preferences", {
            "health_conditions": "TEXT DEFAULT ''",
        })


initialize_db()


def all_rows(sql, params=()):
    with db() as conn:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]


def one_row(sql, params=()):
    with db() as conn:
        row = conn.execute(sql, params).fetchone()
        return dict(row) if row else None


def write(sql, params=()):
    with db() as conn:
        cursor = conn.execute(sql, params)
        conn.commit()
        return cursor.lastrowid


def make_password_hash(password, salt=None):
    """PBKDF2-HMAC-SHA256 password storage; never store a plain-text password."""
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 310_000)
    return f"pbkdf2_sha256$310000${salt.hex()}${digest.hex()}"


def verify_password(password, stored_hash):
    try:
        algorithm, rounds, salt_hex, digest_hex = stored_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        candidate = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(rounds)
        ).hex()
        return hmac.compare_digest(candidate, digest_hex)
    except (ValueError, TypeError, AttributeError):
        return False


def bmi_category(weight_kg, height_cm):
    height_m = max(float(height_cm) / 100.0, 0.5)
    bmi = float(weight_kg) / (height_m * height_m)
    if bmi < 18.5:
        category = "Below the standard adult range"
    elif bmi < 25:
        category = "Standard adult range"
    elif bmi < 30:
        category = "Above the standard adult range"
    else:
        category = "High BMI range"
    return round(bmi, 1), category


def personalized_meal_plan(profile, preferences, goal):
    """Rule-based food ideas filtered by diet type, allergies, restrictions and conditions."""
    conditions = {
        x.strip().lower() for x in str(preferences.get("health_conditions") or "").split(",") if x.strip()
    }
    allergies = [x.strip().lower() for x in str(preferences.get("allergies") or "").split(",") if x.strip()]
    restrictions = [x.strip().lower() for x in str(preferences.get("restrictions") or "").split(",") if x.strip()]
    allergen_aliases = {
        "milk": ["milk", "curd", "paneer", "yogurt", "whey"],
        "dairy": ["milk", "curd", "paneer", "whey"],
        "peanut": ["peanut", "groundnut"],
        "nuts": ["almond", "peanut", "coconut"],
        "tree nuts": ["almond", "coconut"],
        "gluten": ["wheat", "chapati", "bread", "upma"],
        "wheat": ["wheat", "chapati", "bread"],
        "egg": ["egg"],
        "fish": ["fish"],
        "soy": ["tofu", "soy"],
    }
    blocked_terms = set(allergies + restrictions)
    for term in list(blocked_terms):
        blocked_terms.update(allergen_aliases.get(term, []))
    diet_type = str(preferences.get("diet_type") or "No restriction")
    # Calculate BMI from weight and height, then classify the single BMI value.
    # bmi_category(value) is the app-wide category helper used by the BMI dashboard.
    bmi = bmi_calc(
        float(profile.get("weight") or 60),
        float(profile.get("height") or 165),
    )
    bmi_label = bmi_category(bmi)

    meal_plan = {
        "Breakfast": ["Oats, Dry", "Ragi Porridge", "Idli", "Ragi Dosa", "Boiled Egg"],
        "Lunch": ["Brown Rice, Cooked", "Dal, Cooked", "Vegetable Salad", "Chicken Breast", "Tofu", "Fish, Cooked"],
        "Snack": ["Apple", "Guava", "Papaya", "Curd", "Almonds"],
        "Dinner": ["Chapati", "Spinach", "Dal, Cooked", "Tofu", "Chicken Breast", "Vegetable Salad"],
    }

    def allowed(food):
        low = food.lower()
        if any(term in low for term in blocked_terms):
            return False
        if diet_type == "Vegan" and any(x in low for x in ["curd", "milk", "paneer", "egg", "chicken", "fish"]):
            return False
        if diet_type == "Vegetarian" and any(x in low for x in ["chicken", "fish", "egg"]):
            return False
        if diet_type == "Eggetarian" and any(x in low for x in ["chicken", "fish"]):
            return False
        if diet_type == "Pescatarian" and "chicken" in low:
            return False
        if "peanut" in low and "peanut" in allergies:
            return False
        # General, non-prescriptive choices for common conditions.
        if "diabetes" in conditions and food in {"Dates", "Raisins", "Mango", "Dark Chocolate"}:
            return False
        if "high blood pressure" in conditions and food in {"Pongal", "Vegetable Kurma"}:
            return False
        if "high cholesterol" in conditions and food in {"Coconut", "Olive Oil", "Paneer"}:
            return False
        return True

    filtered = {meal: [f for f in foods if allowed(f)] for meal, foods in meal_plan.items()}
    if goal == "Lose weight" or bmi >= 25:
        headline = "Focus on balanced portions, vegetables, fibre-rich foods and protein at meals. Avoid aggressive calorie restriction."
    elif goal in {"Gain weight", "Build muscle"} or bmi < 18.5:
        headline = "Prioritise regular meals and adequate energy, with protein-containing foods and snacks as tolerated."
    else:
        headline = "Aim for balanced portions across meals, a variety of vegetables and fruit, whole grains and protein sources."

    notes = []
    if "diabetes" in conditions:
        notes.append("Diabetes: pair carbohydrate foods with protein/fibre, choose unsweetened drinks, and follow your clinician's glucose and medication plan.")
    if "high blood pressure" in conditions:
        notes.append("High blood pressure: favour fresh, minimally processed foods and limit added salt; follow clinician advice on sodium targets.")
    if "high cholesterol" in conditions:
        notes.append("High cholesterol: favour soluble-fibre foods and unsaturated fats; keep saturated fat modest.")
    if "pcos" in conditions:
        notes.append("PCOS: consistent balanced meals with fibre and protein may help support overall metabolic health; needs vary by person.")
    if "thyroid condition" in conditions:
        notes.append("Thyroid condition: keep medication instructions from your clinician; food timing can matter for some medicines.")
    if "kidney disease" in conditions:
        notes.append("Kidney disease selected: nutrient needs can vary substantially. This app will not prescribe potassium, phosphorus, protein or fluid limits; consult a renal dietitian.")
    if "heart disease" in conditions:
        notes.append("Heart disease selected: follow your clinician's individual nutrition and fluid advice; these are general food ideas only.")
    if "none" in conditions:
        conditions.clear()
    if not any(filtered.values()):
        notes.append("Your current filters exclude all built-in suggestions. Review allergies, restrictions and diet preference.")
    return bmi, bmi_label, headline, filtered, notes


def get_logs(profile_email, selected_date=None):
    if selected_date is not None:
        return all_rows("""
            SELECT * FROM food_logs
            WHERE profile_email = ? AND log_date = ?
            ORDER BY id DESC
        """, (profile_email, str(selected_date)))

    return all_rows("""
        SELECT * FROM food_logs
        WHERE profile_email = ?
        ORDER BY log_date DESC, id DESC
    """, (profile_email,))


def nutrition(food, grams):
    kcal, protein, carbs, fat, fibre = FOODS[food]
    factor = float(grams) / 100
    return {
        "calories": round(kcal * factor, 1),
        "protein": round(protein * factor, 1),
        "carbs": round(carbs * factor, 1),
        "fat": round(fat * factor, 1),
        "fibre": round(fibre * factor, 1),
    }


def totals(logs):
    result = {
        key: 0.0
        for key in ["calories", "protein", "carbs", "fat", "fibre"]
    }
    for item in logs:
        for key in result:
            result[key] += float(item.get(key) or 0)
    return {key: round(value, 1) for key, value in result.items()}


def add_food(profile_email, food, grams, meal, log_date):
    n = nutrition(food, grams)
    write("""
        INSERT INTO food_logs
        (profile_email, food_name, quantity, meal, calories,
         protein, carbs, fat, fibre, log_date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        profile_email, food, float(grams), meal, n["calories"],
        n["protein"], n["carbs"], n["fat"], n["fibre"],
        str(log_date),
    ))
    record_day(profile_email, log_date)


def record_day(profile_email, activity_date):
    write("""
        INSERT OR IGNORE INTO daily_streaks
        (profile_email, activity_date) VALUES (?, ?)
    """, (profile_email, str(activity_date)))


def streak_count(profile_email):
    rows = all_rows("""
        SELECT activity_date FROM daily_streaks
        WHERE profile_email = ?
    """, (profile_email,))

    dates = set()
    for row in rows:
        try:
            dates.add(date.fromisoformat(row["activity_date"][:10]))
        except (ValueError, TypeError):
            pass

    current = date.today()
    if current not in dates:
        current -= timedelta(days=1)
    if current not in dates:
        return 0

    count = 0
    while current in dates:
        count += 1
        current -= timedelta(days=1)
    return count


def bmi_calc(weight, height):
    height_m = float(height) / 100
    return float(weight) / height_m ** 2 if height_m > 0 else 0


def bmi_category(value):
    if value < 18.5:
        return "Underweight"
    if value < 25:
        return "Normal range"
    if value < 30:
        return "Overweight"
    return "Obesity range"


def calorie_target(age, weight, height, activity, gender="Female"):
    # Mifflin-St Jeor estimate; actual needs vary by individual.
    sex_constant = 5 if gender == "Male" else -161
    bmr = 10 * weight + 6.25 * height - 5 * age + sex_constant
    factor = {
        "Sedentary": 1.2,
        "Lightly active": 1.375,
        "Moderately active": 1.55,
        "Very active": 1.725,
    }[activity]
    return max(0, round(bmr * factor))


def burned_calories(exercise, minutes, weight):
    return round(EXERCISES[exercise] * 3.5 * weight / 200 * minutes, 1)


def to_csv(rows):
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig")


def to_pdf(title, rows, subtitle="F.R.I.D.A.Y. · Food & Fitness Report"):
    """Create a compact PDF report. ReportLab is required for PDF export."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    except ImportError as exc:
        raise RuntimeError("PDF export requires ReportLab. Run: python -m pip install reportlab") from exc
    buffer = io.BytesIO()
    wide = bool(rows and len(rows[0]) > 7)
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4) if wide else A4,
                            rightMargin=12*mm, leftMargin=12*mm,
                            topMargin=12*mm, bottomMargin=12*mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="FridayTitle", parent=styles["Title"],
                              textColor=colors.HexColor("#173D32"), fontSize=20, leading=24))
    story = [Paragraph("F.R.I.D.A.Y.", styles["FridayTitle"]),
             Paragraph(title, styles["Heading2"]),
             Paragraph(subtitle.replace("&", "&amp;"), styles["Normal"]), Spacer(1, 8*mm)]
    if rows:
        headers = list(rows[0].keys())
        data = [headers] + [[str(row.get(h, "")) for h in headers] for row in rows]
        table = Table(data, repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#287A53")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("GRID", (0,0), (-1,-1), .4, colors.HexColor("#CBD6C4")),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F0F4E8")]),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("FONTSIZE", (0,0), (-1,-1), 8),
            ("LEFTPADDING", (0,0), (-1,-1), 5), ("RIGHTPADDING", (0,0), (-1,-1), 5),
            ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ]))
        story.append(table)
    else:
        story.append(Paragraph("No records available.", styles["Normal"]))
    doc.build(story)
    return buffer.getvalue()


def macro_targets(calories, gender):
    return {"calories": round(float(calories)),
            "protein": round(float(calories)*0.20/4, 1),
            "carbs": round(float(calories)*0.50/4, 1),
            "fat": round(float(calories)*0.30/9, 1),
            "fibre": 38 if gender == "Male" else 25}


def menstrual_status(profile, on_date=None):
    on_date = on_date or date.today()
    try:
        start = date.fromisoformat(str(profile.get("last_period_start") or "")[:10])
    except ValueError:
        return None
    cycle = max(20, min(45, int(profile.get("cycle_length") or 28)))
    period = max(1, min(10, int(profile.get("period_length") or 5)))
    if start > on_date:
        return {"cycle_day": 1, "phase": "Period date is in the future", "next_period": start,
                "days_until_period": (start-on_date).days, "cycle_length": cycle}
    elapsed = (on_date-start).days
    day = elapsed % cycle + 1
    next_period = start + timedelta(days=(elapsed//cycle+1)*cycle)
    ov_start, ov_end = max(period+1, cycle-16), max(period+1, cycle-12)
    if day <= period: phase = "Menstrual"
    elif ov_start <= day <= ov_end: phase = "Ovulatory window (estimated)"
    elif day < ov_start: phase = "Follicular"
    else: phase = "Luteal"
    return {"cycle_day": day, "phase": phase, "next_period": next_period,
            "days_until_period": (next_period-on_date).days, "cycle_length": cycle}


# ============================================================
# THEME
# ============================================================

font_css = ""
if FONT_PATH.exists():
    try:
        encoded = base64.b64encode(FONT_PATH.read_bytes()).decode("ascii")
        font_css = f"""
        @font-face {{
            font-family: FridayMolgan;
            src: url('data:font/otf;base64,{encoded}') format('opentype');
        }}
        """
    except OSError:
        pass

st.markdown(f"""
<style>
{font_css}

:root {{
    --olive: #E9EBCF;
    --card: rgba(249, 250, 235, 0.70);
    --forest: #173D32;
    --green: #287A53;
    --text: #20352D;
    --muted: #52695C;
    --border: rgba(255, 255, 244, 0.72);
    --control: #292932;
    --cream: #F4F5E6;
}}

html, body {{
    background: #E9EBCF !important;
    margin: 0;
}}

/* Abstract botanical gradient canvas */
[data-testid="stAppViewContainer"] {{
    color: var(--text) !important;
    background:
        radial-gradient(ellipse at 12% 12%, rgba(184, 204, 122, 0.72) 0%, rgba(184, 204, 122, 0) 34%),
        radial-gradient(ellipse at 88% 18%, rgba(80, 139, 105, 0.30) 0%, rgba(80, 139, 105, 0) 32%),
        radial-gradient(ellipse at 76% 82%, rgba(224, 190, 105, 0.26) 0%, rgba(224, 190, 105, 0) 30%),
        linear-gradient(135deg, #F0F1D9 0%, #E0E8C8 46%, #EAF0DC 100%) !important;
    background-attachment: fixed !important;
    position: relative;
    isolation: isolate;
}}

/* Soft blurred abstract shapes behind the app, no image assets required */
[data-testid="stAppViewContainer"]::before,
[data-testid="stAppViewContainer"]::after {{
    content: "";
    position: fixed;
    z-index: -1;
    pointer-events: none;
    border-radius: 48% 52% 63% 37% / 42% 38% 62% 58%;
    filter: blur(2px);
}}

[data-testid="stAppViewContainer"]::before {{
    width: 34rem;
    height: 28rem;
    top: 8rem;
    right: -8rem;
    transform: rotate(-24deg);
    background: linear-gradient(135deg, rgba(40, 122, 83, 0.16), rgba(199, 216, 140, 0.08));
    border: 1px solid rgba(255, 255, 255, 0.28);
}}

[data-testid="stAppViewContainer"]::after {{
    width: 28rem;
    height: 24rem;
    bottom: -7rem;
    left: -8rem;
    transform: rotate(28deg);
    background: linear-gradient(135deg, rgba(226, 191, 102, 0.13), rgba(57, 116, 81, 0.10));
    border: 1px solid rgba(255, 255, 255, 0.24);
}}

[data-testid="stMain"], [data-testid="stHeader"] {{
    background: transparent !important;
}}

[data-testid="stMainBlockContainer"] {{
    max-width: 1600px !important;
    padding: 2rem 2.5rem 4rem !important;
}}

body, p, label, li,
[data-testid="stMarkdownContainer"] {{
    font-size: 1rem !important;
    line-height: 1.5 !important;
}}

h1, h2, h3 {{
    color: var(--forest) !important;
    font-weight: 800 !important;
}}

.friday-title {{
    font-family: FridayMolgan, Georgia, serif !important;
    font-size: clamp(3rem, 7vw, 5.5rem) !important;
    line-height: 1.15 !important;
    letter-spacing: 0.12rem;
    color: var(--forest) !important;
    text-transform: uppercase !important;
    margin: 0.3rem 0;
}}

.friday-subtitle {{
    color: var(--muted) !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.08rem;
}}

.friday-divider {{
    height: 4px;
    max-width: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, #287A53 0%, #91AC59 56%, rgba(145, 172, 89, 0.06) 100%);
    margin: 1rem 0 2rem;
}}

.section-heading {{
    font-size: 1.5rem !important;
    font-weight: 800 !important;
    color: var(--forest) !important;
    margin: 1.8rem 0 1rem;
}}

[data-testid="stVerticalBlockBorderWrapper"] {{
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 18px !important;
    box-shadow: 0 12px 32px rgba(35, 67, 48, 0.075), inset 0 1px 0 rgba(255, 255, 255, 0.58) !important;
    backdrop-filter: blur(18px) saturate(125%);
    -webkit-backdrop-filter: blur(18px) saturate(125%);
}}

[data-testid="stWidgetLabel"] *,
.stSelectbox label, .stNumberInput label,
.stDateInput label, .stTextInput label,
.stTextArea label, .stFileUploader label {{
    color: var(--text) !important;
    font-size: 0.95rem !important;
}}

/* Dark input controls, light readable text */
[data-baseweb="select"] > div,
[data-testid="stNumberInput"] input,
[data-testid="stDateInput"] input,
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {{
    background: var(--control) !important;
    color: var(--cream) !important;
    -webkit-text-fill-color: var(--cream) !important;
    border-color: #41414D !important;
    border-radius: 9px !important;
    font-size: 1rem !important;
}}

[data-baseweb="select"] *,
[data-testid="stNumberInput"] button {{
    color: var(--cream) !important;
}}

[data-testid="stNumberInput"] button {{
    background: var(--control) !important;
    border-color: #41414D !important;
}}

/* Account popover: opaque, readable botanical glass instead of black/translucent */
[data-testid="stPopover"] > button {{
    background: rgba(242, 246, 222, 0.94) !important;
    color: #234334 !important;
    -webkit-text-fill-color: #234334 !important;
    border: 1px solid rgba(255, 255, 245, 0.95) !important;
    border-radius: 14px !important;
    box-shadow: 0 8px 24px rgba(35, 67, 48, 0.12) !important;
}}
[data-testid="stPopover"] > button * {{
    color: #234334 !important;
    -webkit-text-fill-color: #234334 !important;
}}
[data-testid="stPopoverBody"],
[data-testid="stPopoverBody"] > div,
[data-baseweb="popover"] > div {{
    background: rgba(244, 247, 228, 0.98) !important;
    color: #203C30 !important;
    border: 1px solid rgba(255, 255, 248, 0.95) !important;
    border-radius: 18px !important;
    box-shadow: 0 18px 48px rgba(28, 52, 37, 0.22) !important;
    backdrop-filter: blur(22px) saturate(130%) !important;
    -webkit-backdrop-filter: blur(22px) saturate(130%) !important;
}}
[data-testid="stPopoverBody"] p,
[data-testid="stPopoverBody"] label,
[data-testid="stPopoverBody"] [data-testid="stMarkdownContainer"] {{
    color: #203C30 !important;
}}

/* Dropdown popup: dark text on a light surface */
[data-baseweb="popover"] > div,
[data-baseweb="menu"],
[role="listbox"] {{
    background: var(--card) !important;
}}

[role="option"], [role="option"] *,
[data-baseweb="menu"] * {{
    color: var(--text) !important;
}}

[role="option"]:hover {{
    background: var(--olive) !important;
}}

/* Radio and checkbox labels */
[data-testid="stRadio"] label,
[data-testid="stRadio"] label p,
[data-testid="stCheckbox"] label,
[data-testid="stCheckbox"] label p {{
    color: var(--text) !important;
    font-size: 1rem !important;
}}

/* File uploader */
[data-testid="stFileUploaderDropzone"] {{
    background: var(--control) !important;
    border: 1px solid #41414D !important;
    border-radius: 10px !important;
}}

[data-testid="stFileUploaderDropzone"] *,
[data-testid="stFileUploader"] small {{
    color: var(--cream) !important;
}}

[data-testid="stFileUploaderDropzone"] button {{
    background: #353540 !important;
    color: var(--cream) !important;
}}

/* All ordinary buttons */
.stButton > button,
.stDownloadButton > button {{
    background: var(--green) !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    border: 1px solid var(--green) !important;
    border-radius: 10px !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    min-height: 2.8rem !important;
}}

.stButton > button *,
.stDownloadButton > button * {{
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}}

.stButton > button:hover,
.stDownloadButton > button:hover {{
    background: var(--forest) !important;
    border-color: var(--forest) !important;
}}

/* FIX: form submit buttons, including Save Preferences and Log Exercise */
[data-testid="stFormSubmitButton"] button,
[data-testid="stFormSubmitButton"] > button {{
    background: var(--green) !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    border: 1px solid var(--green) !important;
    border-radius: 10px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    min-height: 2.8rem !important;
}}

[data-testid="stFormSubmitButton"] button *,
[data-testid="stFormSubmitButton"] > button * {{
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}}

[data-testid="stFormSubmitButton"] button:hover,
[data-testid="stFormSubmitButton"] > button:hover {{
    background: var(--forest) !important;
    border-color: var(--forest) !important;
}}

/* Metrics */
[data-testid="stMetric"] {{
    background: rgba(249, 250, 235, 0.62) !important;
    border: 1px solid rgba(255, 255, 244, 0.78) !important;
    border-radius: 16px !important;
    padding: 1rem;
    box-shadow: 0 8px 24px rgba(35, 67, 48, 0.06), inset 0 1px 0 rgba(255, 255, 255, 0.65);
    backdrop-filter: blur(16px) saturate(125%);
    -webkit-backdrop-filter: blur(16px) saturate(125%);
}}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] * {{
    color: var(--muted) !important;
}}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * {{
    color: var(--forest) !important;
}}

hr {{
    border-color: var(--border) !important;
}}

/* High-contrast Streamlit alerts, including dietary allergy warnings */
[data-testid="stAlert"] {{
    background-color: #FFF3CD !important;
    border: 1px solid #E3C66B !important;
    border-radius: 10px !important;
}}

[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] div {{
    color: #664D03 !important;
    -webkit-text-fill-color: #664D03 !important;
}}

footer {{
    visibility: hidden;
}}

/* Improve readability of Streamlit warning and info messages */
[data-testid="stAlert"] {{
    background-color: #FFF3CD !important;
    border: 1px solid #E3C66B !important;
    border-radius: 10px !important;
}}

[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] div {{
    color: #664D03 !important;
    -webkit-text-fill-color: #664D03 !important;
}}


/* ========================================================
   F.R.I.D.A.Y. VISUAL UPGRADE: CLEAR GLASS + MOTION
   ======================================================== */
:root {{
    --glass-surface: rgba(255, 255, 255, 0.66);
    --glass-surface-strong: rgba(255, 255, 255, 0.80);
    --glass-stroke: rgba(255, 255, 255, 0.94);
    --glass-shadow: 0 18px 48px rgba(24, 62, 43, 0.13), 0 3px 10px rgba(24, 62, 43, 0.06);
    --glass-highlight: inset 0 1px 0 rgba(255, 255, 255, 0.96);
}}

/* More distinct abstract forms behind the content */
[data-testid="stAppViewContainer"]::before {{
    width: 38rem;
    height: 32rem;
    top: 6rem;
    right: -11rem;
    background: radial-gradient(circle at 35% 35%, rgba(255,255,255,.55), rgba(113,165,119,.22) 42%, rgba(113,165,119,0) 72%);
    border: 1px solid rgba(255,255,255,.45);
    box-shadow: inset 0 0 80px rgba(255,255,255,.20), 0 0 80px rgba(61,117,76,.08);
    animation: friday-float-a 18s ease-in-out infinite alternate;
}}

[data-testid="stAppViewContainer"]::after {{
    width: 32rem;
    height: 29rem;
    bottom: -9rem;
    left: -8rem;
    background: radial-gradient(circle at 60% 35%, rgba(255,255,255,.52), rgba(221,188,103,.23) 40%, rgba(221,188,103,0) 72%);
    border: 1px solid rgba(255,255,255,.42);
    box-shadow: inset 0 0 75px rgba(255,255,255,.18), 0 0 80px rgba(190,166,89,.08);
    animation: friday-float-b 22s ease-in-out infinite alternate;
}}

@keyframes friday-float-a {{
    0% {{ transform: translate3d(0, 0, 0) rotate(-24deg) scale(1); }}
    100% {{ transform: translate3d(-28px, 24px, 0) rotate(-13deg) scale(1.06); }}
}}
@keyframes friday-float-b {{
    0% {{ transform: translate3d(0, 0, 0) rotate(28deg) scale(1); }}
    100% {{ transform: translate3d(24px, -20px, 0) rotate(17deg) scale(1.08); }}
}}
@keyframes friday-enter {{
    from {{ opacity: 0; transform: translateY(12px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes friday-glow {{
    0%, 100% {{ box-shadow: 0 0 0 rgba(40,122,83,0); }}
    50% {{ box-shadow: 0 0 24px rgba(40,122,83,.13); }}
}}

/* Give bordered containers a clearly visible frosted-glass edge */
[data-testid="stVerticalBlockBorderWrapper"] {{
    position: relative !important;
    overflow: hidden !important;
    background: linear-gradient(135deg, rgba(255,255,255,.82), rgba(248,252,241,.57)) !important;
    border: 1px solid var(--glass-stroke) !important;
    border-radius: 22px !important;
    box-shadow: var(--glass-shadow), var(--glass-highlight) !important;
    backdrop-filter: blur(24px) saturate(145%) !important;
    -webkit-backdrop-filter: blur(24px) saturate(145%) !important;
    animation: friday-enter .55s ease both;
    transition: transform .28s ease, box-shadow .28s ease, border-color .28s ease, background .28s ease;
}}
[data-testid="stVerticalBlockBorderWrapper"]::before {{
    content: "";
    position: absolute;
    inset: 0 0 auto 0;
    height: 3px;
    pointer-events: none;
    background: linear-gradient(90deg, rgba(40,122,83,.85), rgba(176,199,111,.75), rgba(255,255,255,.1));
    opacity: .8;
}}
[data-testid="stVerticalBlockBorderWrapper"]:hover {{
    transform: translateY(-3px);
    border-color: rgba(255,255,255,1) !important;
    box-shadow: 0 24px 56px rgba(24,62,43,.17), inset 0 1px 0 rgba(255,255,255,.98) !important;
}}

/* Stronger title and a subtle entrance */
.friday-title {{
    text-shadow: 0 3px 0 rgba(255,255,255,.34), 0 12px 34px rgba(23,61,50,.10);
    animation: friday-enter .75s cubic-bezier(.2,.8,.2,1) both;
}}
.friday-subtitle {{
    display: inline-block;
    padding: .42rem .78rem;
    border: 1px solid rgba(255,255,255,.78);
    border-radius: 999px;
    background: rgba(255,255,255,.42);
    box-shadow: inset 0 1px 0 rgba(255,255,255,.8);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    animation: friday-enter .9s .08s ease both;
}}
.friday-divider {{
    height: 5px;
    border-radius: 999px;
    box-shadow: 0 4px 14px rgba(40,122,83,.20);
    animation: friday-glow 5s ease-in-out infinite;
}}
.section-heading {{
    display: inline-flex;
    align-items: center;
    padding: .55rem 1rem;
    border-radius: 14px;
    background: linear-gradient(135deg, rgba(255,255,255,.75), rgba(255,255,255,.38));
    border: 1px solid rgba(255,255,255,.9);
    box-shadow: 0 8px 22px rgba(24,62,43,.08), inset 0 1px 0 rgba(255,255,255,.95);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    animation: friday-enter .45s ease both;
}}

/* Metrics should read as separate dashboard tiles, not flat text */
[data-testid="stMetric"] {{
    min-height: 112px;
    background: linear-gradient(145deg, rgba(255,255,255,.88), rgba(245,250,237,.64)) !important;
    border: 1px solid rgba(255,255,255,.98) !important;
    border-radius: 20px !important;
    padding: 1.05rem 1.15rem !important;
    box-shadow: 0 14px 30px rgba(24,62,43,.10), inset 0 1px 0 rgba(255,255,255,.98) !important;
    backdrop-filter: blur(22px) saturate(145%) !important;
    -webkit-backdrop-filter: blur(22px) saturate(145%) !important;
    transition: transform .22s ease, box-shadow .22s ease;
}}
[data-testid="stMetric"]:hover {{
    transform: translateY(-4px) scale(1.01);
    box-shadow: 0 20px 36px rgba(24,62,43,.15), inset 0 1px 0 rgba(255,255,255,1) !important;
}}
[data-testid="stMetricLabel"] {{
    font-weight: 700 !important;
    letter-spacing: .025em;
}}
[data-testid="stMetricValue"] {{
    font-weight: 850 !important;
    letter-spacing: -.035em;
}}

/* Olive target badges: harmonise Streamlit metric deltas with the botanical palette */
[data-testid="stMetricDelta"] {{
    display: inline-flex !important;
    align-items: center !important;
    width: fit-content !important;
    max-width: 100% !important;
    padding: .28rem .58rem !important;
    margin-top: .35rem !important;
    border-radius: 999px !important;
    background: rgba(105, 119, 54, .20) !important;
    border: 1px solid rgba(105, 119, 54, .28) !important;
    color: #59652F !important;
    -webkit-text-fill-color: #59652F !important;
}}
[data-testid="stMetricDelta"] * {{
    color: #59652F !important;
    -webkit-text-fill-color: #59652F !important;
    font-weight: 750 !important;
}}
[data-testid="stMetricDelta"] svg {{
    fill: #59652F !important;
    color: #59652F !important;
}}

/* Make fields visibly sit above the glass surface */
[data-baseweb="select"] > div,
[data-testid="stNumberInput"] input,
[data-testid="stDateInput"] input,
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {{
    box-shadow: inset 0 1px 3px rgba(0,0,0,.18), 0 1px 0 rgba(255,255,255,.10) !important;
    border: 1px solid rgba(255,255,255,.17) !important;
    transition: border-color .2s ease, box-shadow .2s ease, transform .2s ease;
}}
[data-baseweb="select"] > div:focus-within,
[data-testid="stNumberInput"] input:focus,
[data-testid="stDateInput"] input:focus,
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {{
    border-color: #83B78E !important;
    box-shadow: 0 0 0 3px rgba(40,122,83,.20), inset 0 1px 3px rgba(0,0,0,.14) !important;
}}

/* Buttons have depth and a quick, restrained interaction */
.stButton > button,
.stDownloadButton > button,
[data-testid="stFormSubmitButton"] button {{
    box-shadow: 0 7px 16px rgba(23,61,50,.16), inset 0 1px 0 rgba(255,255,255,.20) !important;
    transition: transform .18s ease, box-shadow .18s ease, filter .18s ease !important;
}}
.stButton > button:hover,
.stDownloadButton > button:hover,
[data-testid="stFormSubmitButton"] button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 11px 22px rgba(23,61,50,.22), inset 0 1px 0 rgba(255,255,255,.22) !important;
    filter: brightness(1.05);
}}
.stButton > button:active,
.stDownloadButton > button:active,
[data-testid="stFormSubmitButton"] button:active {{
    transform: translateY(0) scale(.985) !important;
}}

/* Expander and tabs receive their own visual boundaries */
[data-testid="stExpander"] {{
    background: rgba(255,255,255,.48) !important;
    border: 1px solid rgba(255,255,255,.88) !important;
    border-radius: 16px !important;
    box-shadow: 0 10px 24px rgba(24,62,43,.07) !important;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
}}
[data-baseweb="tab-list"] {{
    gap: .45rem;
    padding: .35rem;
    border: 1px solid rgba(255,255,255,.78);
    border-radius: 16px;
    background: rgba(255,255,255,.38);
    backdrop-filter: blur(14px);
}}
[data-baseweb="tab"] {{
    border-radius: 11px !important;
    transition: background .2s ease, transform .2s ease;
}}
[data-baseweb="tab"][aria-selected="true"] {{
    background: rgba(40,122,83,.13) !important;
    box-shadow: inset 0 0 0 1px rgba(40,122,83,.18);
}}

/* ========================================================
   FLOATING NUTRITION + FITNESS BACKGROUND
   Large, scattered decorative elements sit behind the app UI.
   ======================================================== */
.friday-side-decor {{
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    overflow: hidden;
    contain: layout paint;
}}
/* Explicit stacking keeps every interactive/content layer in front. */
[data-testid="stMain"],
[data-testid="stHeader"],
[data-testid="stMainBlockContainer"] {{
    position: relative;
    z-index: 1;
}}
.friday-float-item {{
    position: absolute;
    display: grid;
    place-items: center;
    width: clamp(4.2rem, 6vw, 6.6rem);
    height: clamp(4.2rem, 6vw, 6.6rem);
    border-radius: 1.8rem;
    font-size: clamp(2.5rem, 3.8vw, 4rem);
    line-height: 1;
    opacity: .48;
    background: linear-gradient(145deg, rgba(255,255,255,.46), rgba(255,255,255,.12));
    border: 1px solid rgba(255,255,255,.62);
    box-shadow: 0 18px 44px rgba(24,62,43,.09), inset 0 1px 0 rgba(255,255,255,.78);
    backdrop-filter: blur(7px) saturate(125%);
    -webkit-backdrop-filter: blur(7px) saturate(125%);
    will-change: transform;
    animation: friday-background-drift 11s ease-in-out infinite;
}}
/* Scattered around the canvas, with the centre kept comparatively clear. */
.friday-float-item:nth-child(1)  {{ left: 2.5%;  top: 15%;  animation-delay: -1s; }}
.friday-float-item:nth-child(2)  {{ left: 12%;   top: 58%;  animation-delay: -5s; }}
.friday-float-item:nth-child(3)  {{ left: 4.5%;  top: 82%;  animation-delay: -8s; }}
.friday-float-item:nth-child(4)  {{ left: 25%;   top: 7%;   animation-delay: -3s; }}
.friday-float-item:nth-child(5)  {{ left: 31%;   top: 77%;  animation-delay: -7s; }}
.friday-float-item:nth-child(6)  {{ right: 3%;  top: 20%;  animation-delay: -4s; }}
.friday-float-item:nth-child(7)  {{ right: 13%; top: 63%;  animation-delay: -9s; }}
.friday-float-item:nth-child(8)  {{ right: 4%;  top: 84%;  animation-delay: -2s; }}
.friday-float-item:nth-child(9)  {{ right: 27%; top: 8%;   animation-delay: -6s; }}
.friday-float-item:nth-child(10) {{ right: 31%; top: 78%;  animation-delay: -10s; }}
.friday-float-item:nth-child(odd) {{ border-radius: 48% 42% 50% 44%; }}
.friday-float-item:nth-child(even) {{ border-radius: 1.45rem 2rem 1.35rem 2rem; }}
.friday-float-item.fruit {{ box-shadow: 0 18px 44px rgba(24,62,43,.08), 0 0 30px rgba(231,166,76,.10), inset 0 1px 0 rgba(255,255,255,.78); }}
.friday-float-item.fitness {{ box-shadow: 0 18px 44px rgba(24,62,43,.08), 0 0 30px rgba(40,122,83,.12), inset 0 1px 0 rgba(255,255,255,.78); }}
@keyframes friday-background-drift {{
    0%, 100% {{ transform: translate3d(0, 0, 0) rotate(-7deg) scale(.96); }}
    35% {{ transform: translate3d(10px, -18px, 0) rotate(4deg) scale(1.03); }}
    68% {{ transform: translate3d(-8px, 10px, 0) rotate(8deg) scale(.99); }}
}}
@media (max-width: 1250px) {{
    .friday-float-item {{ opacity: .34; width: 4rem; height: 4rem; font-size: 2.5rem; }}
    .friday-float-item:nth-child(4), .friday-float-item:nth-child(5),
    .friday-float-item:nth-child(9), .friday-float-item:nth-child(10) {{ opacity: .20; }}
}}
@media (max-width: 850px) {{
    .friday-side-decor {{ opacity: .65; }}
    .friday-float-item {{ width: 3.2rem; height: 3.2rem; font-size: 2rem; opacity: .22; }}
    .friday-float-item:nth-child(2n) {{ display: none; }}
}}
@media (prefers-reduced-motion: reduce) {{
    .friday-float-item {{ animation: none !important; }}
}}

/* Ensure the Account popover itself and its form controls use the light olive surface. */
[data-testid="stPopoverBody"] [data-testid="stTextInput"] input,
[data-testid="stPopoverBody"] [data-testid="stNumberInput"] input,
[data-testid="stPopoverBody"] [data-testid="stDateInput"] input,
[data-testid="stPopoverBody"] [data-baseweb="select"] > div {{
    background: #fffef4 !important;
    color: #203c30 !important;
    -webkit-text-fill-color: #203c30 !important;
    border-color: #c5d0a0 !important;
}}
[data-testid="stPopoverBody"] [data-testid="stTextInput"] input::placeholder {{
    color: #647564 !important;
    -webkit-text-fill-color: #647564 !important;
}}
[data-testid="stPopoverBody"] [data-testid="stButton"] button,
[data-testid="stPopoverBody"] [data-testid="stFormSubmitButton"] button {{
    background: #687a3d !important;
    color: #fffef4 !important;
    border: 1px solid #596b32 !important;
    border-radius: 10px !important;
}}

</style>
""", unsafe_allow_html=True)

# Make the account trigger match the pale olive glass theme instead of Streamlit's dark default.
st.markdown("""
<style>
div[data-testid="stPopover"] > button,
button[data-testid="stPopoverButton"] {
    background: rgba(239, 242, 211, 0.94) !important;
    color: #214638 !important;
    border: 1px solid rgba(255, 255, 240, 0.95) !important;
    border-radius: 14px !important;
    box-shadow: 0 8px 22px rgba(45, 65, 36, 0.12) !important;
    font-weight: 700 !important;
}
div[data-testid="stPopover"] > button:hover,
button[data-testid="stPopoverButton"]:hover {
    background: rgba(220, 228, 177, 0.98) !important;
    color: #173d32 !important;
    border-color: #a9b878 !important;
}
div[data-testid="stPopoverBody"] {
    background: rgba(239, 242, 211, 0.99) !important;
    color: #214638 !important;
    border: 1px solid rgba(255, 255, 240, 0.98) !important;
    border-radius: 20px !important;
    box-shadow: 0 18px 48px rgba(30, 48, 32, 0.20) !important;
    padding: 1.25rem 1.35rem 1.4rem !important;
    min-width: 360px !important;
    max-width: min(92vw, 520px) !important;
}
div[data-testid="stPopoverBody"] [data-testid="stVerticalBlock"] {
    gap: 1rem !important;
}
div[data-testid="stPopoverBody"] h2,
div[data-testid="stPopoverBody"] h3 {
    margin-top: .35rem !important;
    margin-bottom: .8rem !important;
}
div[data-testid="stPopoverBody"] [data-testid="stRadio"] {
    padding: .35rem 0 .55rem !important;
}
.friday-login-panel-spacing {
    padding: 1.4rem 1.5rem 1.5rem;
}
div[data-testid="stPopoverBody"] label,
div[data-testid="stPopoverBody"] p,
div[data-testid="stPopoverBody"] h1,
div[data-testid="stPopoverBody"] h2,
div[data-testid="stPopoverBody"] h3 {
    color: #214638 !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# DECORATIVE NUTRITION + FITNESS FOREGROUND
# ============================================================

st.markdown("""
<div class="friday-side-decor" aria-hidden="true">
  <span class="friday-float-item fruit">🍊</span>
  <span class="friday-float-item fruit">🍓</span>
  <span class="friday-float-item fruit">🥝</span>
  <span class="friday-float-item fruit">🍌</span>
  <span class="friday-float-item fruit">🍎</span>
  <span class="friday-float-item fitness">🏋️</span>
  <span class="friday-float-item fitness">💪</span>
  <span class="friday-float-item fitness">🏃</span>
  <span class="friday-float-item fitness">🚴</span>
  <span class="friday-float-item fitness">🏆</span>
</div>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

header_left, header_right = st.columns([5, 1], vertical_alignment="center")
with header_left:
    st.markdown(
        '<div class="friday-title">F.R.I.D.A.Y.</div>'
        '<div class="friday-subtitle">'
        'FOOD RECOMMENDATION & INTELLIGENT DIETARY ANALYSIS FOR YOU'
        '</div><div class="friday-divider"></div>',
        unsafe_allow_html=True,
    )

@contextmanager
def account_panel():
    """Show login/profile setup centered before authentication, then in the header popover."""
    if st.session_state.get("authenticated_profile_email"):
        with header_right:
            with st.popover("👤 Account", use_container_width=True):
                yield
    else:
        # Before login, keep the access panel left-aligned instead of centered.
        login_panel_col, empty_space = st.columns([1.35, 2.65], gap="large")
        with login_panel_col:
            with st.container(border=True):
                st.markdown('<div class="friday-login-panel-spacing">', unsafe_allow_html=True)
                yield
                st.markdown('</div>', unsafe_allow_html=True)


with account_panel():
    st.markdown("### Account & profile")
    signed_in_email = st.session_state.get("authenticated_profile_email")
    if signed_in_email:
        st.success("Signed in")
        st.caption(st.session_state.get("authenticated_profile_name") or signed_in_email)
        st.caption(signed_in_email)
        if st.button("Log out", key="friday_header_logout", use_container_width=True):
            st.session_state.pop("authenticated_profile_email", None)
            st.session_state.pop("authenticated_profile_name", None)
            st.rerun()
    else:
        st.caption("Create a profile or select an existing account to sign in.")

    # ============================================================
    # 01. USER PROFILE
    # ============================================================

    st.markdown("### Profile setup")
    profiles = all_rows("SELECT * FROM profiles ORDER BY name COLLATE NOCASE, email COLLATE NOCASE")

    mode = st.radio(
        "Profile option",
        ["Existing Profile", "Add New Profile"],
        horizontal=True,
    )

    if mode == "Add New Profile":
        with st.container(border=True):
            left, right = st.columns(2)
            with left:
                new_name = st.text_input("Full name", key="new_profile_name")
                new_email = st.text_input("Email ID", placeholder="you@example.com", key="new_profile_email")
                new_password = st.text_input("Create password", type="password", help="Use at least 10 characters.", key="new_profile_password")
                new_password_confirm = st.text_input("Confirm password", type="password", key="new_profile_password_confirm")
                new_age = st.number_input("Age", 1, 120, 22, key="new_profile_age")
                new_weight = st.number_input("Weight (kg)", 1.0, 400.0, 60.0, step=0.5, key="new_profile_weight")
                new_gender = st.selectbox("Sex for nutrition estimates", ["Female", "Male"], key="new_profile_gender")
            with right:
                new_height = st.number_input("Height (cm)", 50.0, 250.0, 165.0, step=0.5, key="new_profile_height")
                new_goal_weight = st.number_input("Goal weight (kg)", 1.0, 400.0, 60.0, step=0.5, key="new_profile_goal_weight")
            new_cycle_length, new_period_length, new_last_period = 28, 5, TODAY
            if new_gender == "Female":
                st.markdown("#### Menstrual cycle details")
                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    new_last_period = st.date_input("First day of your last period", value=TODAY, max_value=TODAY, key="new_profile_last_period")
                with mc2:
                    new_cycle_length = st.number_input("Usual cycle length (days)", 20, 45, 28, key="new_profile_cycle_length")
                with mc3:
                    new_period_length = st.number_input("Usual period duration (days)", 1, 10, 5, key="new_profile_period_length")
                st.caption("Cycle dates are estimates and can vary. This is not medical or contraception guidance.")
            if st.button("＋ Create Profile", use_container_width=True, key="create_profile_button"):
                clean_name, clean_email = new_name.strip(), new_email.strip().lower()
                if not clean_name:
                    st.error("Enter a profile name.")
                elif not clean_email or "@" not in clean_email or "." not in clean_email.rsplit("@", 1)[-1]:
                    st.error("Enter a valid email address.")
                elif len(new_password) < 10:
                    st.error("Use a password with at least 10 characters.")
                elif new_password != new_password_confirm:
                    st.error("The passwords do not match.")
                elif one_row("SELECT email FROM profiles WHERE email = ? COLLATE NOCASE", (clean_email,)):
                    st.error("This email is already registered. Choose Existing Profile or use another email.")
                else:
                    try:
                        write("""INSERT INTO profiles
                            (email, name, age, weight, height, goal_weight, gender, cycle_length, period_length, last_period_start, password_hash)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (clean_email, clean_name, int(new_age), float(new_weight), float(new_height),
                             float(new_goal_weight), new_gender, int(new_cycle_length), int(new_period_length),
                             new_last_period.isoformat() if new_gender == "Female" else "", make_password_hash(new_password)))
                        write("INSERT INTO dietary_preferences (profile_email) VALUES (?)", (clean_email,))
                        st.session_state["selected_profile_email"] = clean_email
                        st.session_state["authenticated_profile_email"] = clean_email
                        st.session_state["authenticated_profile_name"] = clean_name
                        st.success("Profile created.")
                        st.rerun()
                    except sqlite3.IntegrityError:
                        st.error("This email is already registered.")
        profiles = all_rows("SELECT * FROM profiles ORDER BY name COLLATE NOCASE, email COLLATE NOCASE")

    if not profiles:
        st.info("Create a profile to start using F.R.I.D.A.Y.")
        st.stop()

    profile_map = {
        f"{p['name']} · {p['email']}": p["email"]
        for p in profiles
    }

    saved_id = st.session_state.get("selected_profile_email")
    labels = list(profile_map.keys())
    default_index = 0

    for i, label in enumerate(labels):
        if profile_map[label] == saved_id:
            default_index = i
            break

    selected_label = st.selectbox(
        "Select account / profile", labels, index=default_index,
        help="Select the account you want to access. A password is required for private records."
    )

    profile_email = profile_map[selected_label]
    st.session_state["selected_profile_email"] = profile_email
    profile = one_row(
        "SELECT * FROM profiles WHERE email = ? COLLATE NOCASE", (profile_email,)
    )

    if not profile:
        st.error("Could not load this profile.")
        st.stop()

    # Password gate: profile selection alone never grants access to its private logs.
    authenticated_email = st.session_state.get("authenticated_profile_email")
    if authenticated_email and authenticated_email.casefold() != profile_email.casefold():
        st.session_state.pop("authenticated_profile_email", None)
        st.session_state.pop("authenticated_profile_name", None)
        authenticated_email = None

    stored_hash = str(profile.get("password_hash") or "")
    if authenticated_email is None:
        st.markdown("### Secure profile login")
        st.caption(f"Sign in to access the private nutrition history for {profile['name']}.")
        if not stored_hash:
            st.info("This older profile has no password yet. Set a password once to secure it.")
            with st.form(f"set_initial_password_{profile_email}"):
                initial_pw = st.text_input("Create password", type="password")
                initial_pw2 = st.text_input("Confirm password", type="password")
                set_pw = st.form_submit_button("Set password and sign in", use_container_width=True)
            if set_pw:
                if len(initial_pw) < 10:
                    st.error("Use a password with at least 10 characters.")
                elif initial_pw != initial_pw2:
                    st.error("The passwords do not match.")
                else:
                    write("UPDATE profiles SET password_hash=? WHERE email=? COLLATE NOCASE", (make_password_hash(initial_pw), profile_email))
                    st.session_state["authenticated_profile_email"] = profile_email
                    st.rerun()
        else:
            with st.form(f"login_form_{profile_email}"):
                login_pw = st.text_input("Password", type="password")
                login_submit = st.form_submit_button("Unlock profile", use_container_width=True)
            if login_submit:
                if verify_password(login_pw, stored_hash):
                    st.session_state["authenticated_profile_email"] = profile_email
                    st.session_state["authenticated_profile_name"] = profile.get("name", "")
                    st.rerun()
                else:
                    st.error("Incorrect password. Please try again.")
        st.caption("Passwords are stored as salted PBKDF2 hashes, not plain text. This is an app-level login; email ownership is not independently verified.")
        st.stop()

profile_gender = profile.get("gender") or "Male"
if profile_gender == "Female":
    with st.container(border=True):
        st.markdown("#### Menstrual cycle tracking")
        raw_start = str(profile.get("last_period_start") or "").strip()
        try: default_start = date.fromisoformat(raw_start[:10]) if raw_start else TODAY
        except ValueError: default_start = TODAY
        default_start = min(default_start, TODAY)
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            last_period_start = st.date_input("First day of last period", value=default_start, max_value=TODAY, key=f"last_period_{profile_email}")
        with mc2:
            cycle_length = st.number_input("Usual cycle length (days)", 20, 45, int(profile.get("cycle_length") or 28), key=f"cycle_length_{profile_email}")
        with mc3:
            period_length = st.number_input("Usual period duration (days)", 1, 10, int(profile.get("period_length") or 5), key=f"period_length_{profile_email}")
        if st.button("Save Menstrual Details", key=f"save_cycle_{profile_email}"):
            write("UPDATE profiles SET last_period_start=?, cycle_length=?, period_length=? WHERE email=? COLLATE NOCASE",
                  (last_period_start.isoformat(), int(cycle_length), int(period_length), profile_email))
            st.success("Menstrual details updated.")
            st.rerun()
        cycle_info = menstrual_status({**profile, "last_period_start": last_period_start.isoformat(), "cycle_length": cycle_length, "period_length": period_length}, TODAY)
        if cycle_info:
            st.metric("Estimated cycle phase", cycle_info["phase"])
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Cycle day", f"Day {cycle_info['cycle_day']}")
            cc2.metric("Estimated next period", cycle_info["next_period"].strftime("%d %b %Y"))
            cc3.metric("Days until next period", str(cycle_info["days_until_period"]))
        st.caption("Calendar estimates only. Actual period and ovulation dates can vary, especially with irregular cycles.")


# ============================================================
# 02. BMI & BODY METRICS
# ============================================================

st.markdown(
    '<div class="section-heading">BMI & Body Metrics</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    c1, c2, c3 = st.columns(3)

    with c1:
        weight = st.number_input(
            "Current weight (kg)", 1.0, 400.0,
            float(profile.get("weight") or 60),
            step=0.5, key=f"weight_{profile_email}",
        )

    with c2:
        height = st.number_input(
            "Height (cm)", 50.0, 250.0,
            float(profile.get("height") or 165),
            step=0.5, key=f"height_{profile_email}",
        )

    with c3:
        age = st.number_input(
            "Age", 1, 120, int(profile.get("age") or 22),
            key=f"age_{profile_email}",
        )

    if st.button("Save Body Metrics"):
        write("""
            UPDATE profiles SET weight=?, height=?, age=? WHERE email=? COLLATE NOCASE
        """, (float(weight), float(height), int(age), profile_email))
        st.success("Body metrics updated.")
        st.rerun()

    bmi = bmi_calc(weight, height)
    category = bmi_category(bmi)

    a, b, c = st.columns(3)
    a.metric("BMI", f"{bmi:.1f}")
    b.metric("BMI Category", category)
    c.metric("Height", f"{height:.1f} cm")

    st.caption("BMI is a screening measure, not a diagnosis.")


# ============================================================
# 03. FOOD SEARCH & QUANTITY
# ============================================================

st.markdown(
    '<div class="section-heading">Food Search & Quantity</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    c1, c2, c3 = st.columns([2, 1, 1])

    with c1:
        selected_food = st.selectbox("Search food", sorted(FOODS))

    with c2:
        quantity = st.number_input(
            "Quantity (g)", 1, 5000, 100, step=10
        )

    with c3:
        meal = st.selectbox("Meal", MEALS)

    preview = nutrition(selected_food, quantity)
    st.caption(f"Estimated energy: {preview['calories']:.1f} kcal")

    if st.button("＋ Add Food to Daily Log"):
        add_food(profile_email, selected_food, quantity, meal, TODAY)
        st.success(f"{selected_food} added to today's log.")
        st.rerun()


# ============================================================
# 04. FOOD IMAGE RECOGNITION
# ============================================================

st.markdown(
    '<div class="section-heading">Food Image Recognition</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    uploaded = st.file_uploader(
        "Upload a food photo",
        type=["png", "jpg", "jpeg", "webp"],
    )

    if uploaded:
        image_col, details_col = st.columns([1, 2])

        with image_col:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded food photo", use_container_width=True)

        with details_col:
            if st.button("Identify Food", key="identify_food_model"):
                try:
                    with st.spinner("Analysing food image..."):
                        st.session_state["friday_food_predictions"] = predict_food(image)
                except Exception as exc:
                    st.error("Food recognition failed. Check your internet connection and model dependencies.")
                    st.exception(exc)

            predictions = st.session_state.get("friday_food_predictions", [])

            if predictions:
                st.markdown("**AI predictions**")
                for label, score in predictions:
                    st.write(f"{label}: {score * 100:.2f}%")

                def match_food_label(label):
                    cleaned = str(label).strip().lower().replace("_", " ").replace("-", " ")
                    for food_name in FOODS:
                        if food_name.lower() == cleaned:
                            return food_name
                    aliases = {
                        "dosa": "Dosa",
                        "idli": "Idli",
                        "chapati": "Chapati",
                        "roti": "Chapati",
                        "rice": "Rice, Cooked",
                        "pongal": "Pongal",
                        "puttu": "Puttu",
                        "upma": "Upma",
                        "sambar": "Sambar",
                        "rasam": "Rasam",
                        "idiyappam": "Idiyappam",
                        "ragi dosa": "Ragi Dosa",
                        "dal": "Dal, Cooked",
                        "egg": "Boiled Egg",
                        "boiled egg": "Boiled Egg",
                        "chicken": "Chicken Breast",
                        "fish": "Fish, Cooked",
                        "banana": "Banana",
                        "apple": "Apple",
                        "mango": "Mango",
                        "papaya": "Papaya",
                        "chapathi": "Chapati",
                    }
                    if cleaned in aliases:
                        return aliases[cleaned]
                    for food_name in FOODS:
                        if food_name.lower().startswith(cleaned):
                            return food_name
                    return None

                suggested_food = match_food_label(predictions[0][0])
                food_options = sorted(FOODS)
                suggested_index = food_options.index(suggested_food) if suggested_food in food_options else 0

                image_food = st.selectbox(
                    "Confirm food",
                    food_options,
                    index=suggested_index,
                    key="image_food",
                )
                image_quantity = st.number_input(
                    "Estimated quantity (g)", 1, 5000, 100,
                    step=10, key="image_quantity"
                )
                image_meal = st.selectbox(
                    "Meal", MEALS, key="image_meal"
                )

                image_nutrition = nutrition(image_food, image_quantity)
                st.write(
                    f"Estimated energy: {image_nutrition['calories']:.1f} kcal"
                )

                if st.button("＋ Add Confirmed Food", key="add_recognised_food"):
                    add_food(
                        profile_email, image_food, image_quantity,
                        image_meal, TODAY
                    )
                    st.success("Confirmed food added.")
                    st.rerun()


# ============================================================
# 05. NUTRITION CALCULATION
# ============================================================

st.markdown(
    '<div class="section-heading">Nutrition Calculation</div>',
    unsafe_allow_html=True,
)

nutrition_date = st.date_input(
    "Select nutrition date", value=TODAY
)

day_logs = get_logs(profile_email, nutrition_date)
day_totals = totals(day_logs)

with st.container(border=True):
    metric_cols = st.columns(5)
    metric_values = [
        ("Calories", f"{day_totals['calories']:.0f} kcal"),
        ("Protein", f"{day_totals['protein']:.1f} g"),
        ("Carbs", f"{day_totals['carbs']:.1f} g"),
        ("Fat", f"{day_totals['fat']:.1f} g"),
        ("Fibre", f"{day_totals['fibre']:.1f} g"),
    ]

    for col, (label, value) in zip(metric_cols, metric_values):
        col.metric(label, value)

    if day_logs:
        df = pd.DataFrame(day_logs)
        fields = [
            "id", "food_name", "quantity", "meal", "calories",
            "protein", "carbs", "fat", "fibre", "log_date",
        ]

        for field in fields:
            if field not in df:
                df[field] = ""

        st.dataframe(
            df[fields].rename(columns={
                "id": "ID",
                "food_name": "Food",
                "quantity": "Quantity (g)",
                "meal": "Meal",
                "calories": "Calories",
                "protein": "Protein (g)",
                "carbs": "Carbs (g)",
                "fat": "Fat (g)",
                "fibre": "Fibre (g)",
                "log_date": "Date",
            }),
            use_container_width=True,
            hide_index=True,
        )

        remove_id = st.selectbox(
            "Choose a food entry to remove",
            [item["id"] for item in day_logs],
            format_func=lambda item: next(
                f"{row['food_name']} · {row['quantity']} g · ID {row['id']}"
                for row in day_logs if row["id"] == item
            ),
        )

        if st.button("Remove Selected Food"):
            write(
                "DELETE FROM food_logs WHERE id=? AND profile_email=?",
                (remove_id, profile_email),
            )
            st.success("Food entry removed.")
            st.rerun()

        food_export_format = st.selectbox("Food log download format", ["CSV", "PDF"], key="food_export_format")
        export_food_rows = df[fields].rename(columns={"id":"ID", "food_name":"Food", "quantity":"Quantity (g)", "meal":"Meal", "calories":"Calories", "protein":"Protein (g)", "carbs":"Carbs (g)", "fat":"Fat (g)", "fibre":"Fibre (g)", "log_date":"Date"}).to_dict(orient="records")
        try:
            if food_export_format == "CSV":
                st.download_button("Download Daily Food Log", to_csv(export_food_rows), file_name=f"friday_food_log_{nutrition_date}.csv", mime="text/csv")
            else:
                st.download_button("Download Daily Food Log", to_pdf(f"Daily Food Log · {nutrition_date}", export_food_rows, f"Profile: {profile['name']} · {profile_email}"), file_name=f"friday_food_log_{nutrition_date}.pdf", mime="application/pdf")
        except RuntimeError as pdf_error:
            st.error(str(pdf_error))
    else:
        st.info("No food entries recorded for this date.")


# ============================================================
# 06. PERSONALIZED DIET PLAN
# ============================================================

st.markdown(
    '<div class="section-heading">Personalized Diet Plan</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    preferences = one_row(
        "SELECT * FROM dietary_preferences WHERE profile_email=?",
        (profile_email,),
    ) or {}

    goals = [
        "Lose weight", "Maintain weight", "Gain weight", "Build muscle"
    ]

    saved_goal = preferences.get("goal", "Maintain weight")
    if saved_goal not in goals:
        saved_goal = "Maintain weight"

    goal = st.selectbox(
        "Primary goal", goals, index=goals.index(saved_goal)
    )

    activity = st.selectbox(
        "Daily activity level",
        ["Sedentary", "Lightly active", "Moderately active", "Very active"],
        index=1,
    )

    maintenance = calorie_target(age, weight, height, activity, profile_gender)

    if goal == "Lose weight":
        target = max(1200, maintenance - 300)
    elif goal == "Gain weight":
        target = maintenance + 250
    elif goal == "Build muscle":
        target = maintenance + 200
    else:
        target = maintenance

    st.metric("Estimated daily energy target", f"{target} kcal")
    daily_targets = macro_targets(target, profile_gender)
    today_macro_totals = totals(get_logs(profile_email, TODAY))
    st.subheader("Daily targets vs consumed today")
    macro_cards = [("Calories", today_macro_totals["calories"], daily_targets["calories"], "kcal"),
                   ("Protein", today_macro_totals["protein"], daily_targets["protein"], "g"),
                   ("Carbohydrates", today_macro_totals["carbs"], daily_targets["carbs"], "g"),
                   ("Fat", today_macro_totals["fat"], daily_targets["fat"], "g"),
                   ("Fibre", today_macro_totals["fibre"], daily_targets["fibre"], "g")]
    macro_cols = st.columns(5)
    for col, (name, consumed, goal_value, unit) in zip(macro_cols, macro_cards):
        with col:
            col.metric(name, f"{consumed:.0f} {unit}" if unit == "kcal" else f"{consumed:.1f} {unit}",
                       f"Target {goal_value:.0f} {unit}" if unit == "kcal" else f"Target {goal_value:.1f} {unit}")
            progress = min(1.0, max(0.0, float(consumed) / max(float(goal_value), 1.0)))
            st.progress(progress, text=f"{round(progress*100)}% of target")
    st.caption("Targets use a general 20% protein / 50% carbohydrate / 30% fat energy split. Estimates are not a medical prescription.")

    remaining = target - day_totals["calories"]
    if remaining >= 0:
        st.write(f"Estimated energy remaining: **{remaining:.0f} kcal**.")
    else:
        st.write(f"Intake exceeds the estimated target by **{abs(remaining):.0f} kcal**.")

    bmi_value, bmi_group, personal_headline, filtered_meals, condition_notes = personalized_meal_plan(profile, preferences, goal)
    st.subheader("Your profile-based plan")
    st.write(f"**BMI:** {bmi_value} · {bmi_group}")
    st.write(personal_headline)
    st.caption("Meal options below are filtered using your saved diet preference, listed allergies, foods to avoid, BMI range, goal and selected health conditions.")

    display_meal_names = ["Breakfast", "Lunch", "Snack", "Dinner"]
    for meal_name in display_meal_names:
        options = filtered_meals.get(meal_name, [])
        st.markdown(f"**{meal_name}**")
        if options:
            st.write(" · ".join(options[:4]))
        else:
            st.warning(f"No built-in {meal_name.lower()} options remain after applying your allergy and diet filters. Please review Dietary Customization.")

    if condition_notes:
        st.subheader("Health-condition considerations")
        for note in condition_notes:
            st.info(note)

    if preferences.get("allergies"):
        st.warning(f"Allergy filter active: {preferences['allergies']}. Verify package labels and cross-contact; this app cannot guarantee allergen-free meals.")

    if profile_gender == "Female":
        cycle_diet = menstrual_status(profile, TODAY)
        if cycle_diet:
            phase = cycle_diet["phase"]
            phase_ideas = {
                "Menstrual": ["Include iron-containing foods such as dal, chickpeas, spinach, or lean meat, paired with vitamin-C-rich fruit.", "Choose regular balanced meals and stay hydrated; dal or soup can be comfortable options."],
                "Follicular": ["Build balanced meals around vegetables, fruit, whole grains, and protein sources.", "Include fibre-rich foods and adequate fluids."],
                "Ovulatory window (estimated)": ["Keep meals balanced with colourful vegetables, fruit, whole grains, and protein.", "Include calcium-rich foods such as curd or milk, plus water throughout the day."],
                "Luteal": ["Plan satisfying meals and snacks with protein plus fibre, such as curd with fruit or dal with whole grains.", "Include magnesium-containing foods such as nuts, seeds, legumes, and leafy greens."],
                "Period date is in the future": ["Enter the actual first day of your most recent period to estimate the current phase."]}
            st.subheader(f"Cycle-aware nutrition · {phase}")
            st.caption(f"Estimated cycle day {cycle_diet['cycle_day']} of {cycle_diet['cycle_length']}; next period estimate {cycle_diet['next_period']:%d %b %Y}.")
            for idea in phase_ideas.get(phase, []): st.markdown(f"- {idea}")
            st.caption("General food ideas only, not treatment for menstrual symptoms. Nutritional needs vary.")
        else:
            st.info("Add your last period start date in Menstrual cycle tracking to see cycle-aware food ideas.")

    st.caption(
        "General suggestions only. Calorie needs vary by individual, "
        "and this is not a medical diet prescription."
    )


# ============================================================
# 07. EXERCISE & STREAK MAINTAINER
# ============================================================

st.markdown(
    '<div class="section-heading">Exercise & Streak Maintainer</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    with st.form("exercise_form", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)

        with c1:
            exercise = st.selectbox("Exercise", list(EXERCISES))

        with c2:
            duration = st.number_input(
                "Duration (minutes)", 1, 1440, 30, step=5
            )

        with c3:
            exercise_date = st.date_input("Activity date", value=TODAY)

        save_exercise = st.form_submit_button(
            "＋ Log Exercise", use_container_width=True
        )

        if save_exercise:
            burn = burned_calories(exercise, duration, weight)

            write("""
                INSERT INTO exercise_logs
                (profile_email, exercise, duration, calories_burned, activity_date)
                VALUES (?, ?, ?, ?, ?)
            """, (
                profile_email, exercise, int(duration), burn,
                str(exercise_date),
            ))

            record_day(profile_email, exercise_date)
            st.success(f"Exercise saved. Estimated burn: {burn:.0f} kcal.")
            st.rerun()

    st.metric("Current activity streak", f"{streak_count(profile_email)} days")

    exercise_logs = all_rows("""
        SELECT * FROM exercise_logs
        WHERE profile_email=?
        ORDER BY activity_date DESC, id DESC
    """, (profile_email,))

    if exercise_logs:
        exercise_df = pd.DataFrame(exercise_logs)
        wanted = [
            c for c in [
                "exercise", "duration", "calories_burned", "activity_date"
            ] if c in exercise_df.columns
        ]

        st.dataframe(
            exercise_df[wanted].rename(columns={
                "exercise": "Exercise",
                "duration": "Minutes",
                "calories_burned": "Estimated kcal burned",
                "activity_date": "Date",
            }),
            use_container_width=True,
            hide_index=True,
        )

        exercise_export_format = st.selectbox("Exercise log download format", ["CSV", "PDF"], key="exercise_export_format")
        export_exercise_rows = exercise_df[wanted].rename(columns={"exercise":"Exercise", "duration":"Minutes", "calories_burned":"Estimated kcal burned", "activity_date":"Date"}).to_dict(orient="records")
        try:
            if exercise_export_format == "CSV":
                st.download_button("Download Exercise Log", to_csv(export_exercise_rows), file_name="friday_exercise_log.csv", mime="text/csv")
            else:
                st.download_button("Download Exercise Log", to_pdf("Exercise Log", export_exercise_rows, f"Profile: {profile['name']} · {profile_email}"), file_name="friday_exercise_log.pdf", mime="application/pdf")
        except RuntimeError as pdf_error:
            st.error(str(pdf_error))
    else:
        st.info("No exercise activities recorded yet.")

    st.caption("Exercise calorie estimates are approximate.")


# ============================================================
# 08. NUTRITIONAL ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-heading">Nutritional Analysis</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    history = get_logs(profile_email)

    if history:
        history_df = pd.DataFrame(history)
        history_df["log_date"] = pd.to_datetime(
            history_df["log_date"], errors="coerce"
        )
        history_df = history_df.dropna(subset=["log_date"])

        if not history_df.empty:
            daily = history_df.groupby(
                history_df["log_date"].dt.date
            ).agg(
                Calories=("calories", "sum"),
                Protein=("protein", "sum"),
                Carbs=("carbs", "sum"),
                Fat=("fat", "sum"),
                Fibre=("fibre", "sum"),
            ).sort_index()

            st.subheader("Daily calorie history")
            st.line_chart(daily[["Calories"]])

            st.subheader("Macronutrients for selected date")
            macro = pd.DataFrame({
                "Nutrient": ["Protein", "Carbs", "Fat", "Fibre"],
                "Grams": [
                    day_totals["protein"], day_totals["carbs"],
                    day_totals["fat"], day_totals["fibre"],
                ],
            }).set_index("Nutrient")

            st.bar_chart(macro)

            st.subheader("Recent daily totals")
            st.dataframe(daily.tail(14), use_container_width=True)
        else:
            st.info("No valid dated nutrition records are available.")
    else:
        st.info("Log foods to unlock nutritional analysis.")


# ============================================================
# 09. DIETARY CUSTOMIZATION
# ============================================================

st.markdown(
    '<div class="section-heading">Dietary Customization</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    preferences = one_row(
        "SELECT * FROM dietary_preferences WHERE profile_email=?",
        (profile_email,),
    ) or {}

    diet_types = [
        "No restriction", "Vegetarian", "Vegan",
        "Eggetarian", "Pescatarian", "Non-vegetarian",
    ]

    saved_diet = preferences.get("diet_type", "No restriction")
    if saved_diet not in diet_types:
        saved_diet = "No restriction"

    with st.form("diet_preferences_form"):
        diet_type = st.selectbox(
            "Dietary preference",
            diet_types,
            index=diet_types.index(saved_diet),
        )

        allergies = st.text_input(
            "Food allergies",
            value=preferences.get("allergies") or "",
            placeholder="Example: peanuts, milk",
        )

        restrictions = st.text_area(
            "Food restrictions or foods to avoid",
            value=preferences.get("restrictions") or "",
            placeholder="Enter ingredients or foods to avoid",
        )

        health_condition_options = [
            "None", "Diabetes", "High blood pressure", "High cholesterol",
            "PCOS", "Thyroid condition", "Kidney disease", "Heart disease", "Anemia",
        ]
        saved_conditions = [x.strip() for x in str(preferences.get("health_conditions") or "").split(",") if x.strip()]
        saved_conditions = [x for x in saved_conditions if x in health_condition_options]
        health_conditions = st.multiselect(
            "Health conditions relevant to food planning",
            health_condition_options,
            default=saved_conditions,
            help="Select only conditions you want considered. These suggestions do not replace clinician advice.",
        )

        save_preferences = st.form_submit_button(
            "Save Dietary Preferences", use_container_width=True
        )

        if save_preferences:
            write("""
                INSERT INTO dietary_preferences
                (profile_email, diet_type, allergies, restrictions, goal, health_conditions)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(profile_email) DO UPDATE SET
                    diet_type=excluded.diet_type,
                    allergies=excluded.allergies,
                    restrictions=excluded.restrictions,
                    goal=excluded.goal,
                    health_conditions=excluded.health_conditions
            """, (
                profile_email, diet_type, allergies.strip(),
                restrictions.strip(), goal,
                ", ".join([x for x in health_conditions if x != "None"]),
            ))

            st.success("Dietary preferences saved.")
            st.rerun()

    if preferences.get("allergies"):
        st.warning(
            f"Recorded allergies: {preferences['allergies']}. "
            "Check ingredients and cross-contamination carefully."
        )


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

st.markdown(
    '<div class="section-heading">Final Summary</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    today_logs = get_logs(profile_email, TODAY)
    today_totals = totals(today_logs)

    today_exercise = all_rows("""
        SELECT * FROM exercise_logs
        WHERE profile_email=? AND activity_date=?
    """, (profile_email, TODAY.isoformat()))

    burn = round(sum(
        float(item.get("calories_burned") or 0)
        for item in today_exercise
    ), 1)

    goal_weight = float(profile.get("goal_weight") or weight)

    st.subheader(f"Daily report · {TODAY:%d %B %Y}")
    st.write(f"**Profile:** {profile['name']}")
    st.write(f"**BMI:** {bmi:.1f} · {category}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Food energy consumed", f"{today_totals['calories']:.0f} kcal")
    c2.metric("Estimated exercise burn", f"{burn:.0f} kcal")
    c3.metric(
        "Intake minus logged exercise",
        f"{today_totals['calories'] - burn:.0f} kcal",
    )

    n1, n2, n3, n4 = st.columns(4)
    n1.metric("Protein", f"{today_totals['protein']:.1f} g")
    n2.metric("Carbs", f"{today_totals['carbs']:.1f} g")
    n3.metric("Fat", f"{today_totals['fat']:.1f} g")
    n4.metric("Fibre", f"{today_totals['fibre']:.1f} g")

    st.subheader("Weight goal")
    st.write(f"Current weight: **{weight:.1f} kg**")
    st.write(f"Goal weight: **{goal_weight:.1f} kg**")
    st.write(f"Difference to goal: **{abs(goal_weight - weight):.1f} kg**")

    report = [{
        "Profile": profile["name"],
        "Date": TODAY.isoformat(),
        "Age": int(age),
        "Height_cm": float(height),
        "Weight_kg": float(weight),
        "BMI": round(bmi, 1),
        "BMI_Category": category,
        "Calories_Consumed_kcal": today_totals["calories"],
        "Exercise_Burn_kcal": burn,
        "Protein_g": today_totals["protein"],
        "Carbs_g": today_totals["carbs"],
        "Fat_g": today_totals["fat"],
        "Fibre_g": today_totals["fibre"],
        "Goal_Weight_kg": goal_weight,
        "Gender": profile_gender,
        "Daily_Calorie_Target_kcal": daily_targets["calories"],
        "Protein_Target_g": daily_targets["protein"],
        "Carbs_Target_g": daily_targets["carbs"],
        "Fat_Target_g": daily_targets["fat"],
        "Fibre_Target_g": daily_targets["fibre"],
    }]
    if profile_gender == "Female":
        cycle_report = menstrual_status(profile, TODAY)
        if cycle_report:
            report[0]["Cycle_Phase_Estimate"] = cycle_report["phase"]
            report[0]["Estimated_Next_Period"] = cycle_report["next_period"].isoformat()

    summary_export_format = st.selectbox("Daily summary download format", ["CSV", "PDF"], key="summary_export_format")
    try:
        if summary_export_format == "CSV":
            st.download_button("Download Daily Summary", to_csv(report), file_name=f"friday_summary_{TODAY}.csv", mime="text/csv", use_container_width=True)
        else:
            st.download_button("Download Daily Summary", to_pdf(f"Daily Summary · {TODAY}", report, f"Profile: {profile['name']} · {profile_email}"), file_name=f"friday_summary_{TODAY}.pdf", mime="application/pdf", use_container_width=True)
    except RuntimeError as pdf_error:
        st.error(str(pdf_error))


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; padding:18px 0;">
        <div style="font-weight:800; color:#173D32; font-size:1.1rem;">
            F.R.I.D.A.Y.
        </div>
        <div style="color:#52695C; font-size:0.9rem;">
            Smart nutrition tracking. Better-informed daily choices.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
