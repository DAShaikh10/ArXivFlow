"""
Curated few-shot examples for LM-based NER on ArXiv abstracts.

These four arXiv IDs are members of the human-annotation IAA set; exclude
them when scoring few-shot runs in `quality/iaa.py` to avoid grading the models on
abstracts whose answers they were shown.

`@author`: DAShaikh10
"""

import json
from typing import Dict, List, TypedDict


class FewShotExample(TypedDict):
    """
    A single curated abstract -> entities demonstration.
    """

    arxiv_id: str
    teaches: str  # Human-readable note on the rule(s) this example illustrates.
    abstract: str
    entities: Dict[str, List[str]]


FEW_SHOT_EXAMPLES: List[FewShotExample] = [
    {
        "arxiv_id": "1810.13091v2",
        "teaches": (
            "Task vs. application_domain ('speech recognition' is the task, 'Code-switching' the domain); "
            "acronym splitting (ASR, MER); compound-language splitting (Mandarin-English -> Mandarin, English)."
        ),
        "abstract": (
            "Code-switching speech recognition has attracted an increasing interest recently, but the need for "
            "expert linguistic knowledge has always been a big issue. End-to-end automatic speech recognition (ASR) "
            "simplifies the building of ASR systems considerably by predicting graphemes or characters directly from "
            "acoustic input. In the mean time, the need of expert linguistic knowledge is also eliminated, which "
            "makes it an attractive choice for code-switching ASR. This paper presents a hybrid CTC-Attention based "
            "end-to-end Mandarin-English code-switching (CS) speech recognition system and studies the effect of "
            "hybrid CTC-Attention based models, different modeling units, the inclusion of language identification "
            "and different decoding strategies on the task of code-switching ASR. On the SEAME corpus, our system "
            "achieves a mixed error rate (MER) of 34.24%."
        ),
        "entities": {
            "target_nlp_task": ["speech recognition", "ASR", "automatic speech recognition"],
            "machine_learning_architecture": ["hybrid CTC-Attention"],
            "training_method": [],
            "dataset_name": ["SEAME"],
            "application_domain": ["Code-switching"],
            "evaluation_metric": ["mixed error rate", "MER"],
            "language_dialect": ["Mandarin-English", "Mandarin", "English"],
        },
    },
    {
        "arxiv_id": "2008.07772v2",
        "teaches": (
            "Restraint — emit the named metric 'BLEU' but skip vague outcome phrases ('state-of-the-art benchmark "
            "results', 'baseline'); named-not-rare ('Transformer' extracted from compounds like 'Transformer-based "
            "models'); dataset trimmed to 'WMT14', not 'WMT14 English-French'; acronym split (NMT)."
        ),
        "abstract": (
            "We explore the application of very deep Transformer models for Neural Machine Translation (NMT). Using "
            "a simple yet effective initialization technique that stabilizes training, we show that it is feasible to "
            "build standard Transformer-based models with up to 60 encoder layers and 12 decoder layers. These deep "
            "models outperform their baseline 6-layer counterparts by as much as 2.5 BLEU, and achieve new "
            "state-of-the-art benchmark results on WMT14 English-French (BLEU and BLEU with back-translation) "
            "and WMT14 English-German (30.1 BLEU).The code and trained models will be publicly available at: "
            "https://github.com/namisan/exdeep-nmt."
        ),
        "entities": {
            "target_nlp_task": ["Neural Machine Translation", "NMT"],
            "machine_learning_architecture": ["Transformer"],
            "training_method": [],
            "dataset_name": ["WMT14"],
            "application_domain": [],
            "evaluation_metric": ["BLEU"],
            "language_dialect": ["English", "French", "German"],
        },
    },
    {
        "arxiv_id": "2012.07527v1",
        "teaches": (
            "Architecture vs. training_method — backbones ('Recurrent Neural Networks', 'BiLSTM-CRF', 'RNN') "
            "vs. the techniques that adapt them ('Input Mixup', 'Manifold Mixup', 'sequence mixup'); acronym split "
            "(RNN); empty categories left as []."
        ),
        "abstract": (
            "In this paper, we extend a class of celebrated regularization techniques originally proposed for "
            "feed-forward neural networks, namely Input Mixup (Zhang et al., 2017) and Manifold Mixup (Verma et al., "
            "2018), to the realm of Recurrent Neural Networks (RNN). Our proposed methods are easy to implement and "
            "have a low computational complexity, while leverage the performance of simple neural architectures in a "
            "variety of tasks. We have validated our claims through several experiments on real-world datasets, and "
            "also provide an asymptotic theoretical analysis to further investigate the properties and potential "
            "impacts of our proposed techniques. Applying sequence mixup to BiLSTM-CRF model (Huang et al., 2015) to "
            "Named Entity Recognition task on CoNLL-2003 data (Sang and De Meulder, 2003) has improved the F-1 score "
            "on the test stage and reduced the loss, considerably."
        ),
        "entities": {
            "target_nlp_task": ["Named Entity Recognition"],
            "machine_learning_architecture": ["Recurrent Neural Networks", "BiLSTM-CRF", "RNN"],
            "training_method": ["Input Mixup", "Manifold Mixup", "sequence mixup"],
            "dataset_name": ["CoNLL-2003"],
            "application_domain": [],
            "evaluation_metric": ["F-1"],
            "language_dialect": [],
        },
    },
    {
        "arxiv_id": "2010.10111v1",
        "teaches": (
            "Named metrics ARE extracted ('precision', 'recall', 'F1') — the flip side of the restraint example; "
            "application_domain 'social media'; a non-English language ('Tamil'); no dataset present -> [] for it."
        ),
        "abstract": (
            "Sentiment analysis has been an active area of research in the past two decades and recently, with the "
            "advent of social media, there has been an increasing demand for sentiment analysis on social media "
            "texts. Since the social media texts are not in one language and are largely code-mixed in nature, the "
            "traditional sentiment classification models fail to produce acceptable results. This paper tries to "
            "solve this very research problem and uses bi-directional LSTMs along with language tagging, to "
            "facilitate sentiment tagging of code-mixed Tamil texts that have been extracted from social media. The "
            "presented algorithm, when evaluated on the test data, garnered precision, recall, and F1 scores of "
            "0.59, 0.66, and 0.58 respectively."
        ),
        "entities": {
            "target_nlp_task": ["sentiment tagging", "Sentiment analysis"],
            "machine_learning_architecture": ["bi-directional LSTMs"],
            "training_method": [],
            "dataset_name": [],
            "application_domain": ["social media"],
            "evaluation_metric": ["precision", "recall", "F1"],
            "language_dialect": ["Tamil"],
        },
    },
]

# arXiv IDs shown to the model as few-shot demos — exclude from the IAA eval set.
FEW_SHOT_ARXIV_IDS: List[str] = [example["arxiv_id"] for example in FEW_SHOT_EXAMPLES]


def build_fewshot_messages() -> List[Dict[str, str]]:
    """
    Render the curated examples as alternating user/assistant turns.

    The user turn mirrors the real query format (`"Abstract:\\n..."`) and the assistant turn is the
    entities object serialized to JSON, so the demonstrations are indistinguishable in shape from the
    turn the model must produce.

    Returns:
        List[Dict[str, str]]: A flat list of message dicts to splice between the system prompt and the
        target abstract's user turn.
    """

    messages: List[Dict[str, str]] = []
    for example in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": f"Abstract:\n{example['abstract']}"})
        messages.append({"role": "assistant", "content": json.dumps(example["entities"], ensure_ascii=False, indent=2)})

    return messages
