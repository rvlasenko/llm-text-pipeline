from pathlib import Path
from dotenv import load_dotenv

from examples.sample_inputs import SAMPLE_INPUTS
from src.llm_text_pipeline.llm_client import LLMClient
from src.llm_text_pipeline.pipeline import process_text_summary
from src.llm_text_pipeline.result_writer import save_json_result

load_dotenv()


def main() -> None:
    client = LLMClient()

    for index, text in enumerate(SAMPLE_INPUTS, start=1):
        result = process_text_summary(
            text=text,
            client=client,
        )

        output_path = Path(f"outputs/result_{index}.json")

        save_json_result(
            result=result,
            output_path=output_path,
        )

        print(f"Saved result to {output_path}")


if __name__ == "__main__":
    main()
