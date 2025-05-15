import subprocess
import argparse


def launch_experiment(name, environment, env_vars, memory=16, scale: int = 4 * 12):
  # Construct the flyctl launch command
  launch_cmd = [
    "flyctl",
    "launch",
    "--dockerfile",
    f"Dockerfile_{environment}",
    "--name",
    f"human-dyna-{environment}-{name}",
    "--config",
    f"configs/human-dyna-{environment}-{name}.toml",
    "--vm-size",
    "performance-8x",
    "--vm-memory",
    str(1024 * memory),
    "--wait-timeout",
    "20m0s",
    "--yes",
  ]
  launch_cmd.extend(env_vars)
  launch_cmd.extend(
    [
      "--env",
      "GIVE_INSTRUCTIONS=1",
      "--env",
      "LOGGER_DISPLAY_TIME=0",
    ]
  )

  # Run the flyctl launch command
  subprocess.run(launch_cmd, check=True)

  # Deploy the website
  deploy_cmd = [
    "flyctl",
    "deploy",
    "--config",
    f"configs/human-dyna-{environment}-{name}.toml",
  ]
  subprocess.run(deploy_cmd, check=True)

  # Scale regionally application
  if scale:
    scale_cmd = [
      "flyctl",
      "scale",
      "count",
      str(scale),
      "--config",
      f"configs/human-dyna-{environment}-{name}.toml",
      "--region",
      "iad,sea,lax,den",
      "--yes",
    ]
    subprocess.run(scale_cmd, check=True)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Launch a Fly.io experiment")
  parser.add_argument("name", help="Name of the experiment")
  parser.add_argument(
    "--environment", default="housemaze", help="Name of the environment"
  )
  parser.add_argument(
    "--env", action="append", help="Environment variables in the format KEY=VALUE"
  )

  parser.add_argument(
    "--scale",
    type=int,
    default=28,
    help="Number of instances to scale the application to",
  )
  args = parser.parse_args()

  env_vars = ["--name", args.name]
  env_vars = ["--env", f"NAME={args.environment}_{args.name}"]
  if args.env:
    for env in args.env:
      env_vars.extend(["--env", env])

  launch_experiment(args.name, args.environment, env_vars, scale=args.scale)
