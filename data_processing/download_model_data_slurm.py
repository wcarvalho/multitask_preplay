"""
This script was used to download the model data from the Harvard Kempner SLURM cluster.

Call from root directory with:
python data_processing/download_model_data_slurm.py
"""

import os

import subprocess
import shutil
import data_configs


def run_command(command, dir_path):
  """Execute a shell command and print the result"""
  print(f"\nExecuting: {command}")
  result = subprocess.run(command, shell=True, capture_output=True, text=True)
  if result.returncode != 0:
    print(f"\nError occurred: {result.stderr}")
  else:
    print(f"\nDownload {dir_path} successfully.")


def reorganize_files(source_dir, target_dir_name, file_prefix):
  """
  Reorganize files from seed directories into a flat structure.

  Args:
      source_dir (str): Path to the source directory containing seed folders
      target_dir_name (str): Name of the target directory to create
      file_prefix (str): Prefix to use for renamed files
  """
  # Get absolute paths
  source_dir = os.path.abspath(source_dir)

  # Create the target directory if it doesn't exist
  target_dir = os.path.join(os.path.dirname(source_dir), target_dir_name)
  os.makedirs(target_dir, exist_ok=True)

  # Change to source directory
  orig_dir = os.getcwd()
  os.chdir(source_dir)

  # Loop through all seed directories
  for item in os.listdir():
    if item.startswith("seed=") and os.path.isdir(item):
      # Extract the seed number
      seed_num = item.split("=")[1]

      # Process config files
      config_src = os.path.join(item, f"{file_prefix}.config")
      config_dst = os.path.join(target_dir, f"{file_prefix}_seed={seed_num}.config")

      # Process safetensors files
      tensor_src = os.path.join(item, f"{file_prefix}.safetensors")
      tensor_dst = os.path.join(
        target_dir, f"{file_prefix}_seed={seed_num}.safetensors"
      )

      # Copy files if they exist
      if os.path.isfile(config_src):
        shutil.move(config_src, config_dst)

      if os.path.isfile(tensor_src):
        shutil.move(tensor_src, tensor_dst)

  # Return to original directory
  os.chdir(orig_dir)

  print(f"Files have been reorganized from {source_dir} to {target_dir}")


def download_model_files(base_server_dir, base_local_dir, model_dirs, model_names):
  """Download model files using rsync

  Args:
      base_server_dir: Base server directory path
      base_local_dir: Base local directory path
      model_dirs: Dictionary of {model_name: dir_path}
      model_names: Dictionary of {model_name: local_model_name}
  """
  # SSH connection details
  hostname = "rcfas_login"  # Using the SSH config alias

  # Common rsync options
  rsync_options = "-avz --prune-empty-dirs --exclude='*wandb*'"

  for model, dir_path in model_dirs.items():
    print("=" * 50)
    server_dir = f"{base_server_dir}/{dir_path}"

    # First download the file
    original_local_dir = os.path.join(base_local_dir, "temp", dir_path)
    os.makedirs(original_local_dir, exist_ok=True)

    run_command(
      f"rsync {rsync_options} {hostname}:{server_dir}/ {original_local_dir}/",
      original_local_dir,
    )

    final_local_dir = os.path.join(base_local_dir, model_names[model])
    if os.path.exists(final_local_dir):
      if os.path.islink(final_local_dir):
        # If it's a symlink, use os.unlink to remove it
        os.unlink(final_local_dir)
        print(f"\nRemoved existing symlink: {final_local_dir}")
      else:
        # If it's a regular directory, use rmtree
        shutil.rmtree(final_local_dir, ignore_errors=True)
        print(f"\nDeleted existing directory: {final_local_dir}")
    shutil.move(original_local_dir, final_local_dir)


if __name__ == "__main__":
  IS_FINAL = True
  model_names = {
    "qlearning": "qlearning",
    "sf": "usfa",
    "dyna": "dyna",
    "preplay": "preplay",
  }

  ############################################################
  # Housemaze
  ############################################################
  print("Downloading Housemaze model data...")
  server_dir = (
    "/n/holylfs06/LABS/kempner_fellow_wcarvalho/jax_rl_results/housemaze_trainer"
  )
  download_model_files(
    base_server_dir=server_dir,
    base_local_dir=data_configs.JAXMAZE_DATA_DIR,
    model_dirs={
      "qlearning": "ql-final/save_data/ql-final-run-6/exp=exp2",
      "sf": "usfa-final/save_data/usfa-final-4/exp=exp2",
      "dyna": "dyna-final/save_data/dyna-final-run-6/alg=dyna,exp=exp2",
      "preplay": "preplay-old-final/save_data/preplay-old-final-run-8/alg=dynaq_shared,exp=exp2",
    },
    model_names=model_names,
  )

  ############################################################
  # Craftax
  ############################################################
  print("Downloading Craftax model data...")
  server_dir = "/n/holylfs06/LABS/kempner_fellow_wcarvalho/jax_rl_results/craftax_multigoal_trainer"
  download_model_files(
    base_server_dir=server_dir,
    base_local_dir=data_configs.CRAFTAX_DATA_DIR,
    model_dirs={
      "qlearning": "ql-final/save_data/ql-final-1/alg=qlearning",
      "sf": "usfa-final/save_data/usfa-final-5/alg=usfa",
      "dyna": "dyna-final/save_data/dyna-final-2/alg=dyna",
      "preplay": "preplay-final/save_data/preplay-final-1/alg=preplay",
    },
    model_names=model_names,
  )
