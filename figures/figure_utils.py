import os
import matplotlib.pyplot as plt
import data_configs


def save_figure(fig, filename, directory=None):
  directory = directory or f"{data_configs.DIRECTORY}/figures"
  filename = os.path.join(directory, f"{filename}.pdf")
  base_dir = os.path.dirname(filename)
  os.makedirs(base_dir, exist_ok=True)
  # plt.savefig(os.path.join(directory, f"{filename}.png"), bbox_inches='tight', dpi=300)
  plt.savefig(filename, bbox_inches="tight", dpi=300)
  print(f"Saved figure to {filename}")
  plt.close()
