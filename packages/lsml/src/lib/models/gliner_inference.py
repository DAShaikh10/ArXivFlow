"""
GLiNER model integration with Label Studio ML backend.

`@author`: DAShaikh10
"""

import os
from typing import Any, Dict, List, Optional

import torch
from gliner import GLiNER
from label_studio_sdk.label_interface.objects import PredictionValue

import wandb

from src.schema import ModelResponse
from src.utils import logger

from .model import LabelStudioMLBase


class LSMLGLiNER(LabelStudioMLBase):
    """
    GLiNER model integration for Label Studio ML backend.
    This class initializes the GLiNER model and provides a predict method for inference.
    """

    def __init__(self, project_id: str, label_config, **kwargs) -> None:
        super().__init__(project_id, label_config, **kwargs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hf_model_name = os.getenv("GLINER_MODEL_NAME")
        self.label_studio_url = os.getenv("LABEL_STUDIO_URL")
        self.label_studio_api = os.getenv("LABEL_STUDIO_API_KEY")
        self.model = None
        self.threshold = float(os.getenv("THRESHOLD"))

    def convert_to_ls_annotation(
        self, prediction: List[Dict[str, Any]], from_name: str, to_name: str
    ) -> List[PredictionValue]:
        """
        Convert from GLiNER output format to Label Studio annotastion format.

        Args:
            prediction: The prediction output from GLiNER
            from_name: The name of the source tag
            to_name: The name of the target tag

        Returns:
            A list of PredictionValue objects in the format expected by Label Studio
        """

        results = []
        sent_preds = []
        for ent in prediction:
            label = [ent["label"]]
            if label:
                score = ent["score"]
                sent_preds.append(
                    {
                        "from_name": from_name,
                        "to_name": to_name,
                        "type": "labels",
                        "value": {"start": ent["start"], "end": ent["end"], "text": ent["text"], "labels": label},
                        "score": round(score, 4),
                    }
                )

        # Add minimum of certaincy scores of entities in sentence for active learning use.
        score = min(p["score"] for p in sent_preds) if sent_preds else 2.0
        results.append(PredictionValue(result=sent_preds, score=score, model_version=self.hf_model_name))

        return results

    def load(self) -> None:
        """
        Load the GLiNER model.
        """

        # Download (once per app lifespan) and load the GLiNER model from Hugging Face,
        # mapping it to the appropriate device.
        self.model = GLiNER.from_pretrained(self.hf_model_name, map_location=str(self.device))

        # Log device type on which the model was loaded.
        device_type = str(next(self.model.parameters()).device).upper()
        logger.info("Loaded GLiNER model '%s' on device: %s", self.hf_model_name, device_type)
        wandb.log({"model_name": self.hf_model_name, "model_device": device_type})

    def predict(self, tasks: List[Dict], context: Optional[Dict]) -> ModelResponse:
        # pylint: disable=line-too-long
        """
        Inference logic.

        Args:
            tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)

        Returns:
            ModelResponse(predictions=predictions) with
            predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        # pylint: enable=line-too-long

        logger.info(
            """\
            Run prediction on %s
            Received context: %s
            Project ID: %s
            Label config: %s
            Parsed JSON Label config: %s
            Extra params: %s""",
            tasks,
            context,
            self.project_id,
            self.label_config,
            self.parsed_label_config,
            self.extra_params,
        )
        wandb.log(
            {
                "Received context": context,
                "Project ID": self.project_id,
                "Parsed JSON Label config": self.parsed_label_config,
                "Extra params": self.extra_params,
            }
        )

        from_name, to_name, value = self.label_interface.get_first_tag_occurence("Labels", "Text")

        # Get labels from the labeling configuration.
        labels = sorted(self.label_interface.get_tag(from_name).labels)

        predictions = []
        for text in [task["data"][value] for task in tasks]:
            entities: List[Dict[str, Any]] = self.model.predict_entities(text, labels, threshold=self.threshold)
            pred: List[PredictionValue] = self.convert_to_ls_annotation(entities, from_name, to_name)
            predictions.extend(pred)

        return ModelResponse(predictions=predictions)
