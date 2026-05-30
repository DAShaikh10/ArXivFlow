"""
Script for LM powered Named Entity Recognition (NER), hence LMNER on ArXiv abstracts using vLLM.

`@author`: DAShaikh10
"""

import json
import os
import re
import time
from typing import Dict, List

import pandas as pd
from codecarbon import EmissionsTracker
from codecarbon.output_methods.logger import LoggerOutput
from pandas import DataFrame
from vllm import LLM, RequestOutput, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

import wandb
from src.utils import logger, resolve_path

from . import config
from .schema import FIELD_TO_LABEL, ArXivEntities


# pylint: disable=too-many-branches, too-many-locals, too-many-statements
def convert_to_label_studio(abstracts: DataFrame, model_name: str, outputs: List[RequestOutput]) -> List[Dict]:
    """
    Convert the raw model outputs to the Label Studio annotation format.

    Args:
        abstracts (DataFrame): The original DataFrame containing the abstracts and their corresponding arXiv IDs.
        outputs (List[RequestOutput]): The list of LLM outputs containing the extracted entities in JSON format.

    Returns:
        List[Dict]: A list of dictionaries formatted for Label Studio, each containing the original text and
        the extracted entities with their spans and labels.
    """

    logger.debug("convert_to_label_studio - START")
    wandb.log({"convert_to_label_studio": "START"})

    results: List[Dict] = []
    raw_records: List[Dict] = []
    skipped_count: int = 0
    truncated_count: int = 0
    emitted_span_count: int = 0
    dropped_span_count: int = 0
    dropped_span_samples: List[Dict] = []
    for abstract, output in zip(abstracts.to_dict("records"), outputs):
        completion = output.outputs[0]
        json_str = completion.text

        # Persist the verbatim model output so empty results stay diagnosable without re-running:
        # an empty Label Studio result is otherwise ambiguous (model returned empty lists vs. the
        # span-relocation regex below dropped everything).
        raw_records.append(
            {
                "arxiv_id": abstract.get("arxiv_id", "UNKNOWN"),
                "finish_reason": completion.finish_reason,
                "raw_output": json_str,
            }
        )

        # A "length" finish reason means the JSON was cut off by max_tokens and is likely malformed.
        if completion.finish_reason == "length":
            truncated_count += 1

        try:
            prediction = json.loads(json_str)
        except json.JSONDecodeError as exception:
            # Keep the abstract with an empty prediction rather than dropping it, so every input
            # always produces an output row. The failure is still surfaced via the warning below.
            skipped_count += 1
            logger.warning(
                "arxiv_id=%s — model output is not valid JSON (finish_reason=%s); emitting empty result: %s",
                abstract.get("arxiv_id", "UNKNOWN"),
                completion.finish_reason,
                str(exception),
            )
            prediction = {}

        result: List[Dict] = []
        seen_spans: set = set()
        for label_name, spans in prediction.items():
            clean_label = FIELD_TO_LABEL.get(label_name, label_name.replace("_", " ").title())

            for span in spans:
                # Guard against empty/whitespace-only spans: re.escape("") matches at every
                # character position, which would otherwise flood the output with bogus entities.
                span = span.strip()
                if not span:
                    continue

                # Match the span as a whole token: the negative look-arounds prevent short spans
                # (e.g. the language code "en") from matching inside unrelated words ("sentence").
                emitted_span_count += 1
                pattern = r"(?<![A-Za-z0-9])" + re.escape(span) + r"(?![A-Za-z0-9])"
                matches = list(re.finditer(pattern, abstract["abstract"], flags=re.IGNORECASE))

                # Surface spans that never relocate instead of dropping them silently: these are usually
                # entities the model inferred rather than copied verbatim (e.g. an assumed language), so a
                # high drop rate is a signal of hallucination, not a relocation bug.
                if not matches:
                    dropped_span_count += 1
                    if len(dropped_span_samples) < 50:
                        dropped_span_samples.append(
                            {"arxiv_id": abstract.get("arxiv_id", "UNKNOWN"), "label": clean_label, "span": span}
                        )
                    continue

                for match in matches:
                    start_idx = match.start()
                    end_idx = match.end()

                    # Skip duplicate annotations for the same span/label at the same position.
                    span_key = (start_idx, end_idx, clean_label)
                    if span_key in seen_spans:
                        continue
                    seen_spans.add(span_key)

                    result.append(
                        {
                            "from_name": "label",
                            "to_name": "text",
                            "type": "labels",
                            "value": {
                                "start": start_idx,
                                "end": end_idx,
                                "text": abstract["abstract"][start_idx:end_idx],
                                "labels": [clean_label],
                            },
                        }
                    )

        results.append(
            {
                "data": {"arxiv_id": abstract["arxiv_id"], "text": abstract["abstract"]},
                "predictions": [{"model_version": config.LM_MODEL_NAME, "result": result}],
            }
        )

    output_path = resolve_path(os.path.dirname(__file__), f"{model_name}-annotation.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Sibling diagnostic file holding the raw model outputs (e.g. lmner_annotation.raw.json).
    raw_output_path = f"{os.path.splitext(output_path)[0]}.raw.json"
    with open(raw_output_path, "w", encoding="utf-8") as f:
        for record in raw_records:
            f.write(json.dumps(record) + "\n")

    logger.info("Extraction complete. Saved to %s (raw outputs: %s)", output_path, raw_output_path)
    wandb.log({"lmner_annotation_file_saved": output_path, "lmner_raw_output_file_saved": raw_output_path})

    if emitted_span_count:
        drop_rate = dropped_span_count / emitted_span_count
        logger.warning(
            "Span relocation: %d of %d model spans (%.1f%%) could not be found verbatim in their abstract "
            "and were dropped — typically inferred/hallucinated entities. Sample: %s",
            dropped_span_count,
            emitted_span_count,
            drop_rate * 100,
            dropped_span_samples[:10],
        )
        wandb.log(
            {
                "lmner_spans_emitted": emitted_span_count,
                "lmner_spans_dropped": dropped_span_count,
                "lmner_span_drop_rate": drop_rate,
            }
        )

    if skipped_count or truncated_count:
        logger.warning(
            "Processed all %d abstracts — %d had unparseable JSON and got an empty result "
            "(%d hit the max_tokens limit). Consider raising max_tokens if truncation is frequent.",
            len(results),
            skipped_count,
            truncated_count,
        )
        wandb.log({"lmner_json_decode_failures": skipped_count, "lmner_truncated_outputs": truncated_count})

    logger.debug("convert_to_label_studio - END")
    wandb.log({"convert_to_label_studio": "END"})

    return results


# pylint: enable=too-many-branches, too-many-locals, too-many-statements


def extract_entities(abstracts: DataFrame) -> List[RequestOutput]:
    """
    Extract named entities from a list of abstracts using the LM.

    Args:
        abstracts (DataFrame): A DataFrame containing the `abstracts` and `arxiv_ids`to process.

    Returns:
        list[RequestOutput]: A list of LLM outputs containing the extracted entities in JSON format.
    """

    logger.debug("extract_entities - START")
    wandb.log({"extract_entities": "START"})

    # We primarily rely on using NVIDIA Ada Lovelace 2x L4 GPUs (48 GB total VRAM)
    llm = LLM(
        model=config.LM_MODEL_NAME,
        tensor_parallel_size=config.VLLM_TENSOR_PARALLEL_SIZE,
        max_model_len=config.MODEL_MAX_LEN,
    )

    sampling_params = SamplingParams(
        temperature=config.MODEL_TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
        structured_outputs=StructuredOutputsParams(json=ArXivEntities.model_json_schema()),
    )

    system_prompt = (
        "You are an expert NLP researcher acting as a highly precise Named Entity Recognition system. "
        "Your task is to extract structured entities from the provided academic abstract and populate the "
        "corresponding JSON schema.\n\n"
        "STRICT EXTRACTION RULES:\n"
        "1. Exact Spans: Extract each entity exactly as it appears in the text, copied verbatim. Every item you emit"
        " MUST be a literal substring of the abstract — if you cannot copy it character-for-character from the text,"
        " do not emit it.\n"
        "2. No Hallucination: Do not infer, guess, or use outside knowledge. If a category is not explicitly mentioned,"
        " return an empty list []. In particular, never assume a default language: only populate language_dialect with"
        " a language or dialect literally named in the abstract, and never emit qualifiers such as"
        " '(assumed from context)'.\n"
        "3. Acronyms vs. Expansions: When an entity is written together with its abbreviation — e.g. 'Bidirectional"
        " Encoder Representations from Transformers (BERT)' or 'word error rate (WER)' — emit the acronym and the full"
        " name as TWO SEPARATE list items ('BERT' and 'Bidirectional Encoder Representations from Transformers')."
        " Never store the combined 'Full Name (ACRONYM)' string as a single entity.\n"
        "4. Specific Entities Only: Extract only concrete, named entities. Skip generic umbrella terms that name a"
        " category rather than a specific instance (e.g. 'neural network', 'deep learning', 'a novel dataset',"
        " 'benchmark datasets', 'downstream tasks', 'performance', 'NLP').\n"
        "5. Category Precision: Carefully distinguish between the categories. In particular, a TASK (what is being"
        " solved, e.g. 'Question Answering', 'Text Classification') belongs in target_nlp_task, NOT in"
        " application_domain, which is reserved for subject areas/industries (e.g. 'Biomedical', 'Legal'). Likewise"
        " separate the model architecture from the training method.\n"
        "6. No Empty Values: Never output empty strings or whitespace-only items. If you are unsure, omit the item.\n"
        "7. Output: Generate only the requested JSON."
    )

    conversations: List[List[Dict[str, str]]] = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Abstract:\n{abstract}"},
        ]
        for abstract in abstracts["abstract"]
    ]

    logger.debug("extract_entities - END")
    wandb.log({"extract_entities": "END"})

    return llm.chat(conversations, sampling_params)


def main() -> None:
    """
    Entry point for the LM-based NER extraction process.
    """

    model_name = config.LM_MODEL_NAME.split("/")[-1].lower()

    # Initialize Weights & Biases for experiment tracking.
    wandb.init(
        project=config.WANDB_PROJECT_NAME,
        name=f"{model_name}-ner-extraction",
        config={
            "max_tokens": config.MAX_TOKENS,
            "model": config.LM_MODEL_NAME,
            "model_max_len": config.MODEL_MAX_LEN,
            "model_temperature": config.MODEL_TEMPERATURE,
            "tensor_parallel_size": config.VLLM_TENSOR_PARALLEL_SIZE,
        },
    )

    # Initialize the carbon emissions tracker to monitor the environmental impact of our inference process.
    tracker = EmissionsTracker(
        experiment_name=f"{model_name}-ner-extraction",
        project_name=config.WANDB_PROJECT_NAME,
        save_to_logger=True,
        logging_logger=LoggerOutput(logger),
    )

    # Load abstracts only from the cleaned dataset.
    current_dir: str = os.path.dirname(__file__)
    dataset_path: str = resolve_path(current_dir, config.CLEANED_DATASET_FILE)
    abstracts: DataFrame = pd.read_json(dataset_path, lines=True)[["arxiv_id", "abstract"]]

    # Start tracking.
    tracker.start()
    start_time: float = time.time()

    outputs: List[RequestOutput] = extract_entities(abstracts)

    inference_duration: float = time.time() - start_time
    emissions_kg_co2: float | None = tracker.stop()
    energy_kwh: float = tracker.final_emissions_data.energy_consumed

    results: List[Dict] = convert_to_label_studio(abstracts, model_name, outputs)

    wandb.log(
        {
            "performance/inference_duration_seconds": inference_duration,
            "performance/abstracts_processed": len(results),
            "performance/throughput_abstracts_per_sec": (
                len(results) / inference_duration if inference_duration > 0 else 0
            ),
            "carbon/emissions_kgCO2eq": emissions_kg_co2,
            "carbon/energy_consumed_kWh": energy_kwh,
        }
    )

    # Save WandB artifact.
    output_path = resolve_path(current_dir, f"{model_name}-annotation.json")
    raw_output_path = f"{os.path.splitext(output_path)[0]}.raw.json"
    artifact = wandb.Artifact(
        name=f"{model_name}-annotations",
        type="dataset",
        description="JSON file containing structured NER entities extracted from ArXiv abstracts using an LM.",
        metadata={"model": config.LM_MODEL_NAME, "instances_count": len(results), "hardware": "2x L4"},
    )
    artifact.add_file(output_path)
    # Bundle the raw model outputs alongside the converted annotations so empties stay diagnosable.
    artifact.add_file(raw_output_path)
    wandb.log_artifact(artifact)

    wandb.finish()


if __name__ == "__main__":
    main()
