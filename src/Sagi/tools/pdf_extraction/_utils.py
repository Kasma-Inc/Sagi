import argparse
import logging
import os
import sys
from typing import Any, Dict

import boto3
import requests
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


def upload_file_to_s3(input_path: str, s3_path: str) -> bool:
    if os.getenv("AWS_ACCESS_KEY_ID", None) is None:
        raise ValueError("AWS_ACCESS_KEY_ID is not set")
    s3_client = boto3.client("s3")

    aws_bucket_name = os.getenv("AWS_BUCKET_NAME")
    if aws_bucket_name is None:
        raise ValueError("AWS_BUCKET_NAME is not set")

    try:
        s3_client.upload_file(input_path, aws_bucket_name, s3_path)
        print(f"✅ Successfully uploaded {input_path} to {s3_path}")
    except ClientError as e:
        logger.error(e)
        return False
    return True


def list_files_in_s3(prefix: str) -> bool:
    if os.getenv("AWS_ACCESS_KEY_ID", None) is None:
        raise ValueError("AWS_ACCESS_KEY_ID is not set")
    s3_client = boto3.client("s3")

    aws_bucket_name = os.getenv("AWS_BUCKET_NAME")
    if aws_bucket_name is None:
        raise ValueError("AWS_BUCKET_NAME is not set")

    try:
        response = s3_client.list_objects_v2(Bucket=aws_bucket_name, Prefix=prefix)

        if "Contents" in response:
            print(f"========== S3 File List ({prefix}) ==========")
            for idx, item in enumerate(response["Contents"]):
                print(f"{idx+1}. {item['Key']}")
            print(f"========== End of S3 File List ({prefix}) ==========")
            return True
        else:
            print(f"No files found in {prefix}")
            return False
    except ClientError as e:
        logger.error(e)
        return False


def cnt_files_in_s3(prefix: str) -> int:
    if os.getenv("AWS_ACCESS_KEY_ID", None) is None:
        raise ValueError("AWS_ACCESS_KEY_ID is not set")
    s3_client = boto3.client("s3")

    aws_bucket_name = os.getenv("AWS_BUCKET_NAME")
    if aws_bucket_name is None:
        raise ValueError("AWS_BUCKET_NAME is not set")

    try:
        response = s3_client.list_objects_v2(Bucket=aws_bucket_name, Prefix=prefix)
        return len(response["Contents"]) if "Contents" in response else 0
    except ClientError as e:
        logger.error(e)
        return 0


def download_file_from_s3(s3_path: str, download_path: str) -> bool:
    if os.getenv("AWS_ACCESS_KEY_ID", None) is None:
        raise ValueError("AWS_ACCESS_KEY_ID is not set")
    s3_client = boto3.client("s3")

    aws_bucket_name = os.getenv("AWS_BUCKET_NAME")
    if aws_bucket_name is None:
        raise ValueError("AWS_BUCKET_NAME is not set")

    try:
        s3_client.download_file(aws_bucket_name, s3_path, download_path)
        print(f"✅ Successfully downloaded {s3_path} to {download_path}")
        return True
    except ClientError as e:
        logger.error(e)
        return False


def delete_file_from_s3(s3_path: str) -> bool:
    if os.getenv("AWS_ACCESS_KEY_ID", None) is None:
        raise ValueError("AWS_ACCESS_KEY_ID is not set")
    s3_client = boto3.client("s3")

    aws_bucket_name = os.getenv("AWS_BUCKET_NAME")
    if aws_bucket_name is None:
        raise ValueError("AWS_BUCKET_NAME is not set")

    try:
        s3_client.delete_object(Bucket=aws_bucket_name, Key=s3_path)
        print(f"✅ Successfully deleted {s3_path} from bucket {aws_bucket_name}")
        return True
    except ClientError as e:
        logger.error(f"Error deleting {s3_path}: {e}")
        return False


def delete_dir_from_s3(s3_path: str) -> bool:
    if os.getenv("AWS_ACCESS_KEY_ID", None) is None:
        raise ValueError("AWS_ACCESS_KEY_ID is not set")
    s3_client = boto3.client("s3")

    aws_bucket_name = os.getenv("AWS_BUCKET_NAME")
    if aws_bucket_name is None:
        raise ValueError("AWS_BUCKET_NAME is not set")

    if not s3_path.endswith("/"):
        s3_path += "/"

    try:
        response = s3_client.list_objects_v2(Bucket=aws_bucket_name, Prefix=s3_path)
        if "Contents" in response:
            for obj in response["Contents"]:
                s3_client.delete_object(Bucket=aws_bucket_name, Key=obj["Key"])
        else:
            print(f"No files found in {s3_path}")
        return True
    except ClientError as e:
        logger.error(e)
        return False


def ocr_parse(input_s3_path: str, output_s3_dir: str) -> Dict[str, Any]:
    api_url: str = os.getenv("OCR_BASE_URL", None)
    model_name: str = os.getenv("OCR_MODEL_NAME", None)
    auth_token: str = os.getenv("OCR_AUTH_TOKEN", None)
    entry_point: str = os.getenv("OCR_ENTRY_POINT", "/parse")
    timeout: int = int(os.getenv("OCR_TIMEOUT", 120))

    if not api_url:
        raise ValueError("OCR_BASE_URL must be set")
    if not model_name:
        raise ValueError("OCR_MODEL_NAME must be set")
    if not auth_token:
        raise ValueError("OCR_AUTH_TOKEN must be set")

    # Prepare headers
    headers = {
        "Model-Name": model_name,
        "Entry-Point": entry_point,
        "Authorization": f"Bearer {auth_token}",
    }

    aws_bucket_name = os.getenv("AWS_BUCKET_NAME")
    if aws_bucket_name is None:
        raise ValueError("AWS_BUCKET_NAME is not set")

    # Prepare data payload
    data = {
        "input_s3_path": f"s3://{aws_bucket_name}/{input_s3_path}",
        "output_s3_path": f"s3://{aws_bucket_name}/{output_s3_dir}",
    }

    try:
        logger.info(f"Sending OCR request for {input_s3_path}")

        # Make the API request
        response = requests.post(api_url, headers=headers, data=data, timeout=timeout)

        # Raise an exception for bad status codes
        response.raise_for_status()

        logger.info(f"OCR request successful. Output saved to {output_s3_dir}")

        # Try to parse JSON response, fallback to text if not JSON
        try:
            return response.json()
        except ValueError:
            return {"response": response.text, "status_code": response.status_code}

    except requests.exceptions.Timeout:
        logger.error(f"Request timeout after {timeout} seconds")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"OCR API request failed: {str(e)}")
        raise


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def main():
    """Main command-line interface."""
    import dotenv

    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(
        description="PDF Extraction S3 Utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Add verbose flag
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    # S3 operations
    parser.add_argument(
        "--list_files",
        metavar="PREFIX",
        type=str,
        nargs="?",
        const="",
        default=None,
        help="List files in S3 with the given prefix (optional, defaults to empty string to list all files)",
    )

    parser.add_argument(
        "--count_files",
        metavar="PREFIX",
        type=str,
        help="Count files in S3 with the given prefix",
    )

    parser.add_argument(
        "--upload",
        nargs=2,
        metavar=("LOCAL_PATH", "S3_PATH"),
        help="Upload a file to S3 (local_path s3_path)",
    )

    parser.add_argument(
        "--download",
        nargs=2,
        metavar=("S3_PATH", "LOCAL_PATH"),
        help="Download a file from S3 (s3_path local_path)",
    )

    parser.add_argument(
        "--delete", metavar="S3_PATH", type=str, help="Delete a file from S3"
    )

    parser.add_argument(
        "--delete_dir",
        metavar="S3_PATH",
        type=str,
        help="Delete a directory (and all its contents) from S3",
    )

    parser.add_argument(
        "--ocr_parse",
        nargs=2,
        metavar=("INPUT_S3_PATH", "OUTPUT_S3_DIR"),
        help="Parse a document using OCR API (input_s3_path output_s3_dir)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return

    try:
        # Execute the requested operation
        if args.list_files is not None:
            success = list_files_in_s3(args.list_files)
            sys.exit(0 if success else 1)

        elif args.count_files:
            count = cnt_files_in_s3(args.count_files)
            print(f"Found {count} files with prefix '{args.count_files}'")
            sys.exit(0)

        elif args.upload:
            local_path, s3_path = args.upload
            success = upload_file_to_s3(local_path, s3_path)
            sys.exit(0 if success else 1)

        elif args.download:
            s3_path, local_path = args.download
            success = download_file_from_s3(s3_path, local_path)
            sys.exit(0 if success else 1)

        elif args.delete:
            success = delete_file_from_s3(args.delete)
            sys.exit(0 if success else 1)

        elif args.delete_dir:
            success = delete_dir_from_s3(args.delete_dir)
            sys.exit(0 if success else 1)

        elif args.ocr_parse:
            input_path, output_dir = args.ocr_parse
            result = ocr_parse(input_path, output_dir)
            print(f"OCR parsing completed. Result: {result}")
            sys.exit(0)

        else:
            parser.print_help()
            sys.exit(1)

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
