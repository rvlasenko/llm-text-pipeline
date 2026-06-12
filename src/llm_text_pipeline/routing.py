from llm_text_pipeline.schemas import TextCategory

ANSWER_STYLE: dict[TextCategory, str] = {
    TextCategory.SUPPORT: "Write a structured, technical answer. Use short numbered steps. Be precise and actionable. Avoid marketing language.",
    TextCategory.COMPLAINT: "Write a careful, empathetic answer. Acknowledge the frustration and apologize where appropriate. Stay calm and reassuring, then state the next concrete step.",
    TextCategory.SALES: "Write a short, persuasive answer. Highlight one concrete benefit and end with a clear, low-pressure call to action. Keep it concise.",
    TextCategory.FEEDBACK: "Write a warm, appreciative answer. Thank the user and acknowledge their input neutrally, without defensiveness or sales pressure.",
    TextCategory.GENERAL_QUESTION: "Write a factual, neutral answer. Explain clearly and briefly, without emotional tone or sales language.",
}


def get_answer_style(category: TextCategory) -> str:
    return ANSWER_STYLE[category]


# Fail on startup if a category has no style, instead of a late KeyError in production.
missing = set(TextCategory) - set(ANSWER_STYLE)
if missing:
    raise RuntimeError(f"ANSWER_STYLE is missing styles for: {sorted(c.value for c in missing)}")
