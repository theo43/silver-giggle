import os
import boto3


def upload_folder_to_s3(
    local_folder_path: str,
    s3_bucket_name: str,
    path_on_s3: str
):
    # Create s3 client
    s3_client = boto3.client('s3')

    # Enumerate local files recursively
    for root, dirs, files in os.walk(local_folder_path):
        for filename in files:
            # Construct the full local path
            local_path = os.path.join(root, filename)

            # Construct the full Dropbox path
            relative_path = os.path.relpath(local_path, local_folder_path)
            s3_path = os.path.join(path_on_s3, relative_path)
            print(f'Searching {s3_path} in {s3_bucket_name}')

            try:
                s3_client.head_object(Bucket=s3_bucket_name, Key=s3_path)
                print(f'Path found on S3! Skipping {s3_path}...')
            except:
                print(f'Uploading {s3_path}...')
                s3_client.upload_file(local_path, s3_bucket_name, s3_path)
