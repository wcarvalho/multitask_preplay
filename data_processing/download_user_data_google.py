"""
This script was used to download the user data from Google Cloud Storage.

Call from root directory with:
python data_processing/download_user_data_google.py
"""

import os
from google.cloud import storage
import fnmatch
import data_configs


##############################
# User Data
##############################
def initialize_storage_client(bucket_name: str):
  storage_client = storage.Client.from_service_account_json(
    data_configs.GOOGLE_CREDENTIALS
  )
  bucket = storage_client.bucket(bucket_name)
  return bucket


def download_user_files(
  bucket_name: str, pattern: str, destination_folder: str, prefix: str = "data/"
):
  # Create a client
  bucket = initialize_storage_client(bucket_name)

  # List all blobs in the bucket with the given prefix
  blobs = bucket.list_blobs(prefix=prefix)

  # Create the destination folder if it doesn't exist
  os.makedirs(destination_folder, exist_ok=True)

  # Download matching files
  for blob in blobs:
    if fnmatch.fnmatch(blob.name, pattern):
      destination_file = os.path.join(destination_folder, os.path.basename(blob.name))
      # Check if file already exists
      if os.path.exists(destination_file):
        print(f"File already exists: \n\t {destination_file}")
        continue
      blob.download_to_filename(destination_file)
      print(f"Downloaded: \n\t from: {blob.name} \n\t to: {destination_file}")


if __name__ == "__main__":
  import data_configs
  from glob import glob

  # housemaze
  bucket_name = data_configs.JAXMAZE_BUCKET
  human_data_pattern = data_configs.JAXMAZE_HUMAN_DATA_PATTERN
  download_user_files(
    bucket_name=bucket_name,
    pattern=human_data_pattern,
    destination_folder=f"{data_configs.JAXMAZE_USER_DIR}",
  )
  jaxmaze_files = f"{data_configs.JAXMAZE_USER_DIR}/*{human_data_pattern}"
  jaxmaze_files = list(set(glob(jaxmaze_files)))

  # craftax
  bucket_name = data_configs.CRAFTAX_BUCKET
  human_data_pattern = data_configs.CRAFTAX_HUMAN_DATA_PATTERN
  download_user_files(
    bucket_name=bucket_name,
    pattern=human_data_pattern,
    destination_folder=f"{data_configs.CRAFTAX_USER_DIR}",
  )
  craftax_files = f"{data_configs.CRAFTAX_USER_DIR}/*{human_data_pattern}"
  craftax_files = list(set(glob(craftax_files)))
  print(f"{len(jaxmaze_files)} JaxMaze files")
  print(f"{len(craftax_files)} Craftax files")
