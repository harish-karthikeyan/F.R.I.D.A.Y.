
def analyze_nutrition(food_logs, targets=None):
    """
    Summarize nutrition from database food-log records.

    Expected fields:
    calories, protein, carbohydrates, fat, fibre
    """

    nutrients = [
        "calories",
        "protein",
        "carbohydrates",
        "fat",
        "fibre",
    ]

    totals = {name: 0.0 for name in nutrients}

    for entry in food_logs:
        for name in nutrients:
            totals[name] += float(entry.get(name, 0) or 0)

    result = {
        "totals": totals,
        "targets": targets or {},
        "remaining": {},
        "percentages": {},
    }

    for name in nutrients:
        if targets and name in targets:
            target = float(targets[name])

            result["remaining"][name] = target - totals[name]
            result["percentages"][name] = (
                totals[name] / target * 100 if target > 0 else 0
            )

    return result


def format_nutrition_report(analysis):
    totals = analysis["totals"]
    lines = [
        "F.R.I.D.A.Y. Nutrition Analysis",
        f"Calories: {totals['calories']:.1f} kcal",
        f"Protein: {totals['protein']:.1f} g",
        f"Carbohydrates: {totals['carbohydrates']:.1f} g",
        f"Fat: {totals['fat']:.1f} g",
        f"Fibre: {totals['fibre']:.1f} g",
    ]

    for name, remaining in analysis["remaining"].items():
        lines.append(
            f"{name.title()} remaining: {remaining:.1f}"
        )

    return "\n".join(lines)
