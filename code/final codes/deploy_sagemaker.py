from pathlib import Path
import tarfile
import argparse

import sagemaker
from sagemaker.tensorflow import TensorFlowModel


def create_model_tar(project_root: Path, output_tar: Path) -> None:
    models_dir = project_root / "models"
    processed_dir = project_root / "data" / "processed"

    required = [
        models_dir / "gtzan_cnn.h5",
        models_dir / "mfcc_mean.npy",
        models_dir / "mfcc_std.npy",
        processed_dir / "classes.npy",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required model artifacts: {missing}")

    staging = project_root / "code" / "final codes" / "model"
    staging.mkdir(parents=True, exist_ok=True)

    (staging / "gtzan_cnn.h5").write_bytes((models_dir / "gtzan_cnn.h5").read_bytes())
    (staging / "mfcc_mean.npy").write_bytes((models_dir / "mfcc_mean.npy").read_bytes())
    (staging / "mfcc_std.npy").write_bytes((models_dir / "mfcc_std.npy").read_bytes())
    (staging / "classes.npy").write_bytes((processed_dir / "classes.npy").read_bytes())

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
        py_version="py310",
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
