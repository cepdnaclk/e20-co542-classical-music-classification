import argparse
import json
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

import sagemaker
import tensorflow as tf
from keras.layers import Dense as KerasDense
from sagemaker.tensorflow import TensorFlowModel


class DenseCompat(KerasDense):
    """Backwards-compatible Dense layer that ignores unknown quantization args."""

    def __init__(self, *args, quantization_config=None, **kwargs):
        super().__init__(*args, **kwargs)


def _strip_quantization_config(node):
    if isinstance(node, dict):
        node.pop("quantization_config", None)
        for value in node.values():
            _strip_quantization_config(value)
    elif isinstance(node, list):
        for value in node:
            _strip_quantization_config(value)


def _sanitize_keras_archive(src: Path) -> Path:
    """Return a temporary .keras copy with incompatible keys removed."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="keras_sanitize_"))
    unpack_dir = tmp_dir / "unpacked"
    unpack_dir.mkdir(parents=True, exist_ok=True)
    sanitized = tmp_dir / src.name

    with zipfile.ZipFile(src, "r") as zf:
        zf.extractall(unpack_dir)

    config_path = unpack_dir / "config.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    _strip_quantization_config(cfg)
    config_path.write_text(json.dumps(cfg), encoding="utf-8")

    with zipfile.ZipFile(sanitized, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in unpack_dir.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(unpack_dir))
    return sanitized


def _load_keras_model(model_path: Path):
    try:
        return tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects={"Dense": DenseCompat},
        )
    except Exception:
        sanitized = _sanitize_keras_archive(model_path)
        return tf.keras.models.load_model(
            sanitized,
            compile=False,
            custom_objects={"Dense": DenseCompat},
        )


def _prepare_saved_model(staging: Path, models_dir: Path) -> None:
    """Create SageMaker TF Serving layout: model version directory `1/`."""
    saved_model_src = models_dir / "saved_model" / "1"
    if saved_model_src.exists():
        shutil.copytree(saved_model_src, staging / "1", dirs_exist_ok=True)
        return

    keras_model_path = models_dir / "gtzan_cnn.keras"
    h5_model_path = models_dir / "gtzan_cnn.h5"

    if keras_model_path.exists():
        model = _load_keras_model(keras_model_path)
    elif h5_model_path.exists():
        model = tf.keras.models.load_model(h5_model_path, compile=False)
    else:
        raise FileNotFoundError(
            "Missing model file. Expected one of: "
            f"{saved_model_src}, {keras_model_path}, {h5_model_path}"
        )

    tf.saved_model.save(model, str(staging / "1"))


def create_model_tar(project_root: Path, output_tar: Path) -> None:
    models_dir = project_root / "models"
    processed_dir = project_root / "data" / "processed"

    required = [
        models_dir / "mfcc_mean.npy",
        models_dir / "mfcc_std.npy",
        processed_dir / "classes.npy",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required model artifacts: {missing}")

    staging = project_root / "code" / "final codes" / "model"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    _prepare_saved_model(staging, models_dir)
    (staging / "mfcc_mean.npy").write_bytes((models_dir / "mfcc_mean.npy").read_bytes())
    (staging / "mfcc_std.npy").write_bytes((models_dir / "mfcc_std.npy").read_bytes())
    (staging / "classes.npy").write_bytes((processed_dir / "classes.npy").read_bytes())

    if output_tar.exists():
        output_tar.unlink()
    with tarfile.open(output_tar, "w:gz") as tar:
        tar.add(staging, arcname=".")


def deploy(
    role_arn: str,
    instance_type: str,
    instance_count: int,
    endpoint_name: str | None,
    key_prefix: str,
) -> str:
    project_root = Path(__file__).resolve().parents[2]
    source_dir = Path(__file__).resolve().parent
    model_tar = source_dir / "model.tar.gz"

    create_model_tar(project_root, model_tar)

    session = sagemaker.Session()
    bucket = session.default_bucket()
    model_s3_uri = session.upload_data(str(model_tar), bucket=bucket, key_prefix=key_prefix)

    tf_model = TensorFlowModel(
        model_data=model_s3_uri,
        role=role_arn,
        framework_version="2.13",
        entry_point="inference.py",
        source_dir=str(source_dir),
        sagemaker_session=session,
    )

    predictor = tf_model.deploy(
        initial_instance_count=instance_count,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
    )
    return predictor.endpoint_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy GTZAN CNN to SageMaker real-time endpoint")
    parser.add_argument("--role-arn", required=True, help="IAM role ARN used by SageMaker")
    parser.add_argument("--instance-type", default="ml.m5.large")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--endpoint-name", default=None)
    parser.add_argument("--key-prefix", default="gtzan-cnn")

    args = parser.parse_args()

    endpoint = deploy(
        role_arn=args.role_arn,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        endpoint_name=args.endpoint_name,
        key_prefix=args.key_prefix,
    )
    print(f"Endpoint deployed: {endpoint}")


if __name__ == "__main__":
    main()
