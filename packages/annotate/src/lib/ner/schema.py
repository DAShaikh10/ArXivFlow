"""
Schema definitions for NER module.

`@author`: DAShaikh10
"""

from typing import Dict, List, TypedDict

from pydantic import BaseModel, ConfigDict, Field

NER_LABELS: List[str] = [
    "Application Domain",
    "Dataset or Benchmark Name",
    "Evaluation Metric",
    "Language or Dialect",
    "Machine Learning Architecture",
    "Target NLP Task",
    "Training or Fine-tuning Method",
]

FIELD_TO_LABEL: Dict[str, str] = {
    "target_nlp_task": "Target NLP Task",
    "machine_learning_architecture": "Machine Learning Architecture",
    "training_method": "Training or Fine-tuning Method",
    "dataset_name": "Dataset or Benchmark Name",
    "application_domain": "Application Domain",
    "evaluation_metric": "Evaluation Metric",
    "language_dialect": "Language or Dialect",
}


class Entity(TypedDict):
    """
    Entity schema representing a named entity found in the paper abstract.
    """

    text: str
    label: str
    start: int
    end: int


class Annotation(TypedDict):
    """
    Annotation schema representing the final output of the NER process for a single paper.
    """

    id: str
    entities: List[Entity]


# pylint: disable=too-few-public-methods


class ArXivEntities(BaseModel):
    """
    Comprehensive schema for extracting and categorizing structured Natural Language Processing (NLP)
    and Machine Learning (ML) entities from academic research paper abstracts.
    Only include entities explicitly mentioned in the text.
    """

    # Emit `additionalProperties: false` so guided decoding refuses to generate any key outside this
    # schema (the model was inventing extra, non-list fields that crashed the converter).
    model_config = ConfigDict(extra="forbid")

    target_nlp_task: List[str] = Field(
        default_factory=list,
        max_length=10,
        description="The NLP problem or objective the work aims to solve (the what), not how it is solved & not the "
        + "subject area (e.g., Machine Translation, Named Entity Recognition, Question Answering, Text Classification, "
        + "Natural Language Inference). Subject areas like Biomedical or Legal belong in application_domain.",
    )
    machine_learning_architecture: List[str] = Field(
        default_factory=list,
        max_length=10,
        description="Specific, named model architectures, neural network families, or structural backbones, not "
        + "training procedures (e.g., Transformer, Mamba, CNN, BERT). When an architecture is given with its "
        + "abbreviation, list the acronym and the full name as two separate items. Exclude generic terms such as "
        + "'neural network' or 'deep learning'.",
    )
    training_method: List[str] = Field(
        default_factory=list,
        max_length=10,
        description="Optimization strategies, learning paradigms, or model adaptation techniques that update weights "
        + "(e.g., LoRA, RLHF, Fine-Tuning)",
    )
    dataset_name: List[str] = Field(
        default_factory=list,
        max_length=10,
        description="Specific, proper-named datasets, corpora, benchmarks, evaluation suites, or leaderboards "
        + "(e.g., SQuAD, MMLU, ImageNet). Exclude generic phrases such as 'a novel dataset', 'benchmark datasets', "
        + "or 'two datasets'.",
    )
    application_domain: List[str] = Field(
        default_factory=list,
        max_length=10,
        description="The real-world subject area, industry vertical, or specialized text type the work targets "
        + "(e.g., Biomedical, Legal, Finance, Social Media). This is NOT an NLP task — tasks such as Question "
        + "Answering or Text Classification belong in target_nlp_task.",
    )
    evaluation_metric: List[str] = Field(
        default_factory=list,
        max_length=10,
        description="Established, proper-named quantitative metrics actually used to report or compare scores "
        + "(e.g., BLEU, ROUGE-L, METEOR, F1-Score, Precision, Recall, Accuracy, WER, CER, Perplexity, MAP, MRR, "
        + "AUC, BERTScore). The term MUST be a recognized metric name AND denote a number computed to score a "
        + "system's output. If it is not a named metric, omit it. In particular, EXCLUDE: (a) vague outcome or "
        + "quality phrases ('performance', 'state-of-the-art results', 'comparable results', 'translation quality', "
        + "'diversity', 'stability', 'empirically more robust', 'essential features'); (b) similarity or distance "
        + "functions and math operations unless explicitly reported as the scoring metric ('cosine', 'Euclidean', "
        + "'softmax', 'distance metric'); (c) tasks, methods, or analyses ('back-translation', 'k-shot learning', "
        + "'cross-domain analysis', 'missing word prediction', 'classifiers'); and (d) datasets or benchmarks "
        + "('ground truth dataset', 'benchmark', 'publicly available dataset').",
    )
    language_dialect: List[str] = Field(
        default_factory=list,
        max_length=10,
        description="Specific natural languages or dialects only (e.g., English, Mandarin, Egyptian Arabic, Hindi). "
        + "Strictly exclude generic linguistic terms (e.g., 'text', 'sentence', 'natural language', 'word embeddings') "
        + "and method or model names.",
    )


# pylint: enable=too-few-public-methods


class Paper(TypedDict):
    """
    Paper schema representing the essential information of a research paper for NER processing.
    """

    id: str
    abstract: str
