
def generate_diet_plan(
    foods,
    goal,
    preferences=None,
    allergies=None,
    restrictions=None,
):
    preferences = preferences or []
    allergies = allergies or []
    restrictions = restrictions or []

    goal = goal.strip().lower()

    blocked_words = [
        item.strip().lower()
        for item in allergies
        if item.strip()
    ]

    available_foods = {}

    for name, nutrition in foods.items():
        name_lower = name.lower()

        # Exclude foods containing a listed allergen.
        if any(word in name_lower for word in blocked_words):
            continue

        # Apply basic vegetarian/vegan filtering.
        if (
            ("vegan" in [p.lower() for p in preferences]
             or "vegetarian" in [p.lower() for p in preferences])
            and any(word in name_lower for word in [
                "chicken", "fish", "meat", "egg"
            ])
        ):
            continue

        if "vegan" in [p.lower() for p in preferences] and any(
            word in name_lower
            for word in ["milk", "curd", "paneer", "cheese"]
        ):
            continue

        available_foods[name] = nutrition

    if not available_foods:
        return {
            "Breakfast": [],
            "Lunch": [],
            "Dinner": [],
            "Snack": [],
        }

    if goal == "lose weight":
        ranked = sorted(
            available_foods.items(),
            key=lambda item: item[1].get("calories", 0),
        )
    elif goal == "gain weight":
        ranked = sorted(
            available_foods.items(),
            key=lambda item: item[1].get("calories", 0),
            reverse=True,
        )
    else:
        ranked = sorted(
            available_foods.items(),
            key=lambda item: item[1].get("protein", 0),
            reverse=True,
        )

    # Simple food suggestions, not a medically personalized diet.
    names = [name for name, _ in ranked]

    plan = {
        "Breakfast": names[:2],
        "Lunch": names[2:4],
        "Dinner": names[4:6],
        "Snack": names[6:8],
    }

    return plan
