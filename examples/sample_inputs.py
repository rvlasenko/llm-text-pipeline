from dataclasses import dataclass

from llm_text_pipeline.schemas import TextCategory


@dataclass(frozen=True)
class SampleInput:
    text: str
    expected_category: TextCategory


SAMPLE_INPUTS: list[SampleInput] = [
    SampleInput(
        text="I bought a subscription yesterday, but I still cannot access premium features. The payment was successful, and I received a receipt. Help.",
        expected_category=TextCategory.SUPPORT,
    ),
    SampleInput(
        text="How do I connect my account to the Slack? I followed the setup docs, but clicking the Connect button does nothing.",
        expected_category=TextCategory.SUPPORT,
    ),
    SampleInput(
        text="I was charged twice this month and nobody answered my emails for a whole week. This is unacceptable and I want a refund.",
        expected_category=TextCategory.COMPLAINT,
    ),
    SampleInput(
        text="Your app crashed in the middle of my client presentation and made me look unprofessional. I am extremely frustrated.",
        expected_category=TextCategory.COMPLAINT,
    ),
    SampleInput(
        text="I am comparing your Pro and Enterprise plans for a team of 20 people. Which one offers the best value, and do you provide discounts?",
        expected_category=TextCategory.SALES,
    ),
    SampleInput(
        text="Is there a discount if I upgrade from the monthly plan to the yearly plan today?",
        expected_category=TextCategory.SALES,
    ),
    SampleInput(
        text="Just wanted to say the new dashboard layout is much cleaner and far easier to navigate than before.",
        expected_category=TextCategory.FEEDBACK,
    ),
    SampleInput(
        text="The dark mode looks nice, but I think the contrast on the buttons could be improved.",
        expected_category=TextCategory.FEEDBACK,
    ),
    SampleInput(
        text="How does your data retention policy work, and where exactly are account backups stored?",
        expected_category=TextCategory.GENERAL_QUESTION,
    ),
    SampleInput(
        text="I have too many tasks this week and I feel overwhelmed. I do not know what to start with...",
        expected_category=TextCategory.GENERAL_QUESTION,
    ),
]
