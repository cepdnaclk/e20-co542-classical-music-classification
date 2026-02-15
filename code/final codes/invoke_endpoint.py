import argparse
import boto3


def main() -> None:
    parser = argparse.ArgumentParser(description="Invoke SageMaker endpoint with WAV payload")
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--audio-path", required=True)
    args = parser.parse_args()

    runtime = boto3.client("sagemaker-runtime")
    with open(args.audio_path, "rb") as f:
        payload = f.read()

    resp = runtime.invoke_endpoint(
        EndpointName=args.endpoint_name,
        ContentType="audio/wav",
        Body=payload,
    )

    print(resp["Body"].read().decode("utf-8"))


if __name__ == "__main__":
    main()
