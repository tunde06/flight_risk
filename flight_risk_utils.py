# Cell 34 — Create flight_risk_utils.py
#
# This shared module becomes the single source of prediction and SHAP
# behavior for the notebook, FastAPI, retention agent, and Streamlit.


import base64
import json
import uuid
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


DEFAULT_ARTIFACT_DIR = Path("model_artifacts")


def load_artifacts(
    artifact_dir=DEFAULT_ARTIFACT_DIR,
):
    artifact_dir = Path(artifact_dir)

    model_path = (
        artifact_dir
        / "employee_flight_risk_smote_lightgbm_pipeline.pkl"
    )

    feature_path = (
        artifact_dir
        / "model_features.json"
    )

    metadata_path = (
        artifact_dir
        / "model_metadata.json"
    )

    model = joblib.load(model_path)

    with open(
        feature_path,
        "r",
        encoding="utf-8",
    ) as file:
        feature_columns = json.load(file)

    with open(
        metadata_path,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    return (
        model,
        feature_columns,
        metadata,
    )


def prepare_input(
    input_data,
    feature_columns,
):
    if isinstance(input_data, dict):
        input_df = pd.DataFrame(
            [input_data]
        )

    elif isinstance(
        input_data,
        pd.DataFrame,
    ):
        input_df = input_data.copy()

    else:
        raise TypeError(
            "input_data must be a dictionary "
            "or pandas DataFrame."
        )

    for column in feature_columns:
        if column not in input_df.columns:
            input_df[column] = np.nan

    return input_df[
        feature_columns
    ].copy()


def get_model_space(
    model,
    input_df,
    feature_columns,
):
    if not hasattr(
        model,
        "named_steps",
    ):
        raise TypeError(
            "Expected a fitted pipeline "
            "with named_steps."
        )

    imputer = model.named_steps[
        "imputer"
    ]

    classifier = model.named_steps[
        "classifier"
    ]

    transformed = imputer.transform(
        input_df
    )

    scaler = model.named_steps.get(
        "scaler"
    )

    if scaler is not None:
        transformed = scaler.transform(
            transformed
        )

    model_space = pd.DataFrame(
        transformed,
        columns=feature_columns,
        index=input_df.index,
    )

    return (
        model_space,
        classifier,
    )


def extract_leave_shap(
    explainer,
    model_space,
):
    shap_values = explainer.shap_values(
        model_space
    )

    if isinstance(shap_values, list):
        leave_values = np.asarray(
            shap_values[1]
        )
    else:
        shap_array = np.asarray(
            shap_values
        )

        if shap_array.ndim == 3:
            leave_values = (
                shap_array[:, :, -1]
            )
        else:
            leave_values = shap_array

    expected_value = (
        explainer.expected_value
    )

    if isinstance(
        expected_value,
        (list, tuple, np.ndarray),
    ):
        expected_array = np.asarray(
            expected_value
        ).reshape(-1)

        base_value = (
            expected_array[1]
            if len(expected_array) > 1
            else expected_array[0]
        )

    else:
        base_value = expected_value

    return (
        leave_values,
        float(base_value),
    )


def _python_value(value):
    if pd.isna(value):
        return None

    if isinstance(
        value,
        (np.integer,),
    ):
        return int(value)

    if isinstance(
        value,
        (np.floating,),
    ):
        return float(value)

    return value


def encode_image_to_base64(
    image_path,
):
    image_path = Path(image_path)

    with open(
        image_path,
        "rb",
    ) as image_file:
        return base64.b64encode(
            image_file.read()
        ).decode("utf-8")


def predict_with_shap_explanation(
    input_data,
    model,
    feature_columns,
    threshold=0.50,
    top_n=10,
    output_dir=(
        "model_artifacts/shap_outputs"
    ),
    filename_prefix=(
        "employee_prediction"
    ),
    return_shap_image_base64=False,
):
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_df = prepare_input(
        input_data=input_data,
        feature_columns=feature_columns,
    )

    probability_leave = (
        model.predict_proba(
            input_df
        )[:, 1]
    )

    prediction_value = (
        probability_leave >= threshold
    ).astype(int)

    first_probability = float(
        probability_leave[0]
    )

    first_prediction = int(
        prediction_value[0]
    )

    prediction_label = (
        "Leave"
        if first_prediction == 1
        else "Stay"
    )

    prediction = {
        "prediction_label": (
            prediction_label
        ),
        "prediction_value": (
            first_prediction
        ),
        "probability_leave": round(
            first_probability,
            4,
        ),
        "flight_risk_score": round(
            first_probability * 100,
            2,
        ),
        "threshold_used": round(
            float(threshold),
            4,
        ),
    }

    (
        model_space,
        classifier,
    ) = get_model_space(
        model=model,
        input_df=input_df,
        feature_columns=feature_columns,
    )

    explainer = shap.TreeExplainer(
        classifier
    )

    (
        leave_shap_values,
        base_value,
    ) = extract_leave_shap(
        explainer=explainer,
        model_space=model_space,
    )

    shap_row = np.asarray(
        leave_shap_values
    )[0]

    shap_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "feature_value": [
                _python_value(value)
                for value
                in input_df.iloc[0].values
            ],
            "shap_impact": shap_row,
        }
    )

    shap_df["abs_impact"] = (
        shap_df["shap_impact"].abs()
    )

    shap_df["effect"] = np.where(
        shap_df["shap_impact"] > 0,
        "Increases leave risk",
        "Reduces leave risk",
    )

    top_features = (
        shap_df
        .sort_values(
            "abs_impact",
            ascending=False,
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    explanation = shap.Explanation(
        values=shap_row,
        base_values=base_value,
        data=model_space.iloc[0],
        feature_names=feature_columns,
    )

    prediction_id = str(
        uuid.uuid4()
    )[:8]

    image_path = (
        output_dir
        / (
            f"{filename_prefix}_"
            f"{prediction_id}_"
            "shap_waterfall.png"
        )
    )

    plt.figure()

    shap.plots.waterfall(
        explanation,
        max_display=top_n,
        show=False,
    )

    plt.savefig(
        image_path,
        bbox_inches="tight",
        dpi=150,
    )

    plt.close()

    top_records = []

    for record in (
        top_features[
            [
                "feature",
                "feature_value",
                "shap_impact",
                "effect",
            ]
        ]
        .to_dict(
            orient="records"
        )
    ):
        record["feature_value"] = (
            _python_value(
                record[
                    "feature_value"
                ]
            )
        )

        record["shap_impact"] = round(
            float(
                record[
                    "shap_impact"
                ]
            ),
            6,
        )

        top_records.append(record)

    risk_increasers = (
        top_features[
            top_features[
                "shap_impact"
            ] > 0
        ]["feature"]
        .head(5)
        .tolist()
    )

    risk_reducers = (
        top_features[
            top_features[
                "shap_impact"
            ] < 0
        ]["feature"]
        .head(5)
        .tolist()
    )

    explanation_text = {
        "summary": (
            f"The model predicts "
            f"'{prediction_label}' with a "
            f"{first_probability * 100:.2f}% "
            "probability of leaving within "
            "the next 3 months."
        ),
        "top_risk_increasing_features": (
            risk_increasers
        ),
        "top_risk_reducing_features": (
            risk_reducers
        ),
    }

    result = {
        "prediction": prediction,
        "explanation_text": (
            explanation_text
        ),
        "top_shap_features": (
            top_records
        ),
        "shap_image_path": str(
            image_path
        ),
    }

    if return_shap_image_base64:
        result["shap_image_base64"] = (
            encode_image_to_base64(
                image_path
            )
        )

    return result
