import logging
from pathlib import Path

from dotenv import load_dotenv

from examples.sample_inputs import SAMPLE_INPUTS
from llm_text_pipeline.llm_client import LLMClient
from llm_text_pipeline.pipeline import process_text
from llm_text_pipeline.result_writer import save_json_result

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s %(message)s",
)


def main() -> None:
    client = LLMClient()
    matches = 0

    for index, sample in enumerate(SAMPLE_INPUTS, start=1):
        result = process_text(text=sample.text, client=client)

        output_path = Path(f"outputs/result_{index}.json")
        save_json_result(result=result, output_path=output_path)

        analysis = result.result
        self_check = result.trace.self_check

        is_match = analysis.category is sample.expected_category
        matches += is_match
        marker = "OK " if is_match else "MISS"

        print(
            f"[{marker}] expected={sample.expected_category.value:<16} "
            f"predicted={analysis.category.value:<16} "
            f"self_check={self_check.verdict.value:<7} -> {output_path}"
        )

    print(f"\nCategory match: {matches}/{len(SAMPLE_INPUTS)}")


if __name__ == "__main__":
    main()
