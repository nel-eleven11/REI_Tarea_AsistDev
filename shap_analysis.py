"""Generate image SHAP explanations for the committed vehicle CNN.

The script is intentionally self-contained: it loads the four sample images
that were exported from the original notebook, explains the three samples
used in the paper, and writes both machine-readable results and publication
figures to ``figures/shap``.

Usage
-----
    .venv/bin/python shap_analysis.py

The default of 300 model evaluations per image is a compromise between
reproducibility and runtime on CPU.  Increase it for a smoother estimate:

    .venv/bin/python shap_analysis.py --max-evals 500
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import zipfile
from pathlib import Path

# Keep the run quiet and deterministic before importing TensorFlow/Matplotlib.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-rei-shap")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import shap
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers


SEED = 42
IMAGE_SIZE = (128, 128)
CLASS_NAMES = [
    "Auto Rickshaws",
    "Bikes",
    "Cars",
    "Motorcycles",
    "Planes",
    "Ships",
    "Trains",
]

# The labels come from the validation output already stored in the source
# notebook.  The PNGs are committed so this analysis can be rerun without
# downloading the full Vehicles/ directory.
SAMPLES = [
    {
        "file": "train_misclassified.png",
        "true_label": "Trains",
        "short": "Tren clasificado como automóvil",
    },
    {
        "file": "ship_misclassified.png",
        "true_label": "Ships",
        "short": "Barco clasificado como avión",
    },
    {
        "file": "bike_correct.png",
        "true_label": "Bikes",
        "short": "Bicicleta clasificada correctamente",
    },
]


def build_notebook_model() -> keras.Model:
    """Recreate the architecture in vehicle-image-classification-using-cnn."""

    augmentation = keras.Sequential(
        [
            layers.Input(shape=(*IMAGE_SIZE, 3)),
            layers.RandomFlip("horizontal", seed=SEED),
            layers.RandomRotation(0.06, seed=SEED),
            layers.RandomZoom(0.08, seed=SEED),
        ]
    )
    model = keras.Sequential(
        [
            layers.Input(shape=(*IMAGE_SIZE, 3)),
            augmentation,
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.30),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.25),
            layers.Dense(len(CLASS_NAMES), activation="softmax"),
        ]
    )
    return model


def load_model_compat(model_path: Path) -> tuple[keras.Model, str]:
    """Load the model, falling back to topology + weights for newer Keras.

    The committed artifact was saved with Keras 3.15.1.  Older environments
    can reject its newer initializer config even though the layer topology and
    weights are compatible.  The fallback only reconstructs the known
    notebook architecture; it does not alter the committed model file.
    """

    try:
        return keras.models.load_model(model_path, compile=False), "direct"
    except (TypeError, ValueError, OSError):
        if not zipfile.is_zipfile(model_path):
            raise
        with tempfile.TemporaryDirectory(prefix="rei-shap-") as tmp:
            weights_path = Path(tmp) / "model.weights.h5"
            with zipfile.ZipFile(model_path) as archive:
                weights_path.write_bytes(archive.read("model.weights.h5"))
            model = build_notebook_model()
            model.load_weights(weights_path)
            return model, "topology+weights fallback"


def load_samples(input_dir: Path) -> tuple[np.ndarray, list[dict]]:
    records = []
    images = []
    for sample in SAMPLES:
        path = input_dir / sample["file"]
        if not path.is_file():
            raise FileNotFoundError(f"No existe la imagen de prueba: {path}")
        image = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
        images.append(np.asarray(image, dtype=np.float32))
        records.append({**sample, "path": str(path)})
    return np.stack(images), records


def region_name(row: int, col: int, grid_size: int = 4) -> str:
    vertical = ["superior", "medio-superior", "medio-inferior", "inferior"][row]
    horizontal = ["izquierda", "centro-izquierda", "centro-derecha", "derecha"][col]
    return f"{vertical}-{horizontal}"


def region_scores(heatmap: np.ndarray, grid_size: int = 4) -> np.ndarray:
    """Sum absolute pixel attribution into a small spatial grid."""

    height, width = heatmap.shape
    rows = np.array_split(np.arange(height), grid_size)
    cols = np.array_split(np.arange(width), grid_size)
    scores = np.zeros((grid_size, grid_size), dtype=np.float64)
    for row, row_ids in enumerate(rows):
        for col, col_ids in enumerate(cols):
            scores[row, col] = np.abs(heatmap[np.ix_(row_ids, col_ids)]).sum()
    return scores


def normalize_for_display(heatmap: np.ndarray, limit: float) -> tuple[np.ndarray, np.ndarray]:
    """Return a diverging normalization and an alpha map for overlays."""

    limit = max(float(limit), np.finfo(float).eps)
    normalized = np.clip(heatmap / limit, -1, 1)
    alpha = np.clip(np.abs(normalized) * 0.90, 0.10, 0.90)
    return normalized, alpha


def make_overview(
    images: np.ndarray,
    heatmaps: np.ndarray,
    records: list[dict],
    probabilities: np.ndarray,
    target_names: list[str],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(
        len(records), 2, figsize=(8.2, 3.1 * len(records)), constrained_layout=True
    )
    if len(records) == 1:
        axes = np.asarray([axes])
    for row, (image, heatmap, record, probability, target) in enumerate(
        zip(images, heatmaps, records, probabilities, target_names)
    ):
        axes[row, 0].imshow(image.astype(np.uint8))
        axes[row, 0].set_title(
            f"{record['true_label']} → {target} ({probability:.1%})", fontsize=10
        )
        axes[row, 0].axis("off")

        # Normalize each sample independently so a high-confidence example
        # does not make a lower-magnitude explanation look blank.
        sample_limit = float(np.percentile(np.abs(heatmap), 99))
        normalized, alpha = normalize_for_display(heatmap, sample_limit)
        axes[row, 1].imshow(image.astype(np.uint8))
        plot = axes[row, 1].imshow(
            normalized,
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            alpha=alpha,
        )
        axes[row, 1].set_title(
            "SHAP: rojo favorece / azul contradice\n" + target, fontsize=10
        )
        axes[row, 1].axis("off")
    fig.colorbar(plot, ax=axes[:, 1].tolist(), fraction=0.025, pad=0.02, label="signo e intensidad")
    fig.suptitle("Explicaciones locales del clasificador de vehículos", fontsize=14, y=0.995)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_sample_figure(
    image: np.ndarray,
    heatmap: np.ndarray,
    record: dict,
    probability: float,
    target: str,
    output_path: Path,
) -> None:
    scores = region_scores(heatmap)
    shares = scores / max(scores.sum(), np.finfo(float).eps)
    sample_limit = float(np.percentile(np.abs(heatmap), 99))
    normalized, alpha = normalize_for_display(heatmap, sample_limit)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1, 1, 1.05]})

    axes[0].imshow(image.astype(np.uint8))
    axes[0].set_title(f"Entrada\nreal: {record['true_label']}", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(image.astype(np.uint8))
    plot = axes[1].imshow(normalized, cmap="RdBu_r", vmin=-1, vmax=1, alpha=alpha)
    axes[1].set_title(f"SHAP para {target}\nP = {probability:.1%}", fontsize=10)
    axes[1].axis("off")
    fig.colorbar(plot, ax=axes[1], fraction=0.046, pad=0.04, label="contribución relativa")

    axes[2].imshow(np.zeros_like(image, dtype=np.uint8) + 245)
    for row in range(4):
        for col in range(4):
            left, right = col * 32, (col + 1) * 32
            top, bottom = row * 32, (row + 1) * 32
            axes[2].add_patch(
                plt.Rectangle(
                    (left, top), 32, 32,
                    fill=False,
                    linewidth=1.0,
                    edgecolor="black",
                )
            )
            axes[2].text(
                left + 16,
                top + 16,
                f"{shares[row, col]:.0%}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
    axes[2].set_xlim(0, 128)
    axes[2].set_ylim(128, 0)
    axes[2].set_title("Peso espacial\n|SHAP| por región 4×4", fontsize=10)
    axes[2].axis("off")

    fig.suptitle(record["short"], fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_global_region_figure(global_scores: np.ndarray, output_path: Path) -> None:
    shares = global_scores / max(global_scores.sum(), np.finfo(float).eps)
    labels = [region_name(row, col) for row in range(4) for col in range(4)]
    values = shares.reshape(-1) * 100
    order = np.argsort(values)[::-1]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(np.arange(len(order)), values[order], color="#377eb8")
    ax.set_xticks(np.arange(len(order)), [labels[index] for index in order], rotation=55, ha="right")
    ax.set_ylabel("Porcentaje del |SHAP| agregado")
    ax.set_title("Ranking agregado de regiones espaciales (tres imágenes)")
    ax.grid(axis="y", alpha=0.25)
    for x, index in enumerate(order[:3]):
        ax.text(x, values[index] + 0.2, f"{values[index]:.1f}%", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    model, load_mode = load_model_compat(args.model)
    images, records = load_samples(args.input_dir)

    probabilities = model.predict(images, verbose=0)
    predicted_ids = probabilities.argmax(axis=1)
    predicted_names = [CLASS_NAMES[index] for index in predicted_ids]

    def predict_fn(batch: np.ndarray) -> np.ndarray:
        return model(batch, training=False).numpy()

    masker = shap.maskers.Image("blur(32,32)", images[0].shape)
    explainer = shap.Explainer(
        predict_fn,
        masker,
        output_names=CLASS_NAMES,
        seed=SEED,
    )
    # Selecting the top output per image is the image-classification pattern
    # documented by SHAP.  The chosen output is the model's own prediction.
    explanation = explainer(
        images,
        max_evals=args.max_evals,
        batch_size=args.batch_size,
        outputs=shap.Explanation.argsort.flip[:1],
    )

    values = np.asarray(explanation.values)[..., 0]
    # RGB channels are summed because the paper discusses spatial regions;
    # the raw per-channel values remain in shap_values.npz.
    heatmaps = values.sum(axis=-1)
    target_names = [str(names[0]) for names in explanation.output_names]
    target_probabilities = probabilities[np.arange(len(images)), predicted_ids]
    base_values = np.asarray(explanation.base_values).reshape(len(images), -1)[:, 0]
    total_attribution = values.reshape(len(images), -1).sum(axis=1)
    additivity_error = base_values + total_attribution - target_probabilities
    if target_names != predicted_names:
        raise RuntimeError(
            "SHAP seleccionó una salida distinta a la predicción del modelo: "
            f"{target_names} != {predicted_names}"
        )
    if not np.all(np.abs(additivity_error) < 1e-4):
        raise RuntimeError(f"Falló la comprobación de aditividad: {additivity_error}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    make_overview(
        images,
        heatmaps,
        records,
        target_probabilities,
        target_names,
        args.output_dir / "shap_overview.png",
    )

    global_scores = np.zeros((4, 4), dtype=np.float64)
    rows = []
    for index, (image, heatmap, record) in enumerate(zip(images, heatmaps, records)):
        scores = region_scores(heatmap)
        global_scores += scores
        shares = scores / max(scores.sum(), np.finfo(float).eps)
        flat_order = np.argsort(scores.reshape(-1))[::-1]
        top_abs = int(flat_order[0])
        top_positive = np.unravel_index(np.argmax(np.maximum(heatmap, 0)), heatmap.shape)
        top_negative = np.unravel_index(np.argmin(np.minimum(heatmap, 0)), heatmap.shape)
        top_abs_row, top_abs_col = np.unravel_index(top_abs, scores.shape)
        rows.append(
            {
                "sample": record["file"],
                "true_label": record["true_label"],
                "predicted_label": predicted_names[index],
                "confidence": f"{target_probabilities[index]:.8f}",
                "base_probability": f"{base_values[index]:.8f}",
                "positive_mass": f"{np.maximum(heatmap, 0).sum():.8e}",
                "negative_mass": f"{np.minimum(heatmap, 0).sum():.8e}",
                "top_abs_region": region_name(top_abs_row, top_abs_col),
                "top_abs_share": f"{shares[top_abs_row, top_abs_col]:.8f}",
                "top_positive_pixel": f"fila {top_positive[0] + 1}, columna {top_positive[1] + 1}",
                "top_negative_pixel": (
                    "ninguna"
                    if not np.any(heatmap < 0)
                    else f"fila {top_negative[0] + 1}, columna {top_negative[1] + 1}"
                ),
                "additivity_error": f"{additivity_error[index]:.8e}",
            }
        )
        make_sample_figure(
            image,
            heatmap,
            record,
            target_probabilities[index],
            target_names[index],
            args.output_dir / record["file"].replace(".png", "_shap.png"),
        )

    make_global_region_figure(global_scores, args.output_dir / "global_region_importance.png")

    with (args.output_dir / "shap_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(
        args.output_dir / "shap_values.npz",
        images=images,
        shap_values=values,
        base_values=base_values,
        probabilities=probabilities,
        predicted_ids=predicted_ids,
        class_names=np.asarray(CLASS_NAMES),
    )
    (args.output_dir / "results.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "image_size": IMAGE_SIZE,
                "max_evals": args.max_evals,
                "batch_size": args.batch_size,
                "masker": "blur(32,32)",
                "explainer": "shap.Explainer (Partition for Image masker)",
                "model_load_mode": load_mode,
                "tensorflow_version": tf.__version__,
                "shap_version": shap.__version__,
                "samples": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Modelo: {load_mode}; TensorFlow {tf.__version__}; SHAP {shap.__version__}")
    print(f"Muestras explicadas: {len(records)}; max_evals={args.max_evals}")
    for row in rows:
        print(
            f"{row['sample']}: {row['true_label']} -> {row['predicted_label']} "
            f"({float(row['confidence']):.2%}); error aditivo={float(row['additivity_error']):.2e}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("vehicle_classifier.keras"))
    parser.add_argument("--input-dir", type=Path, default=Path("figures/inputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures/shap"))
    parser.add_argument("--max-evals", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
