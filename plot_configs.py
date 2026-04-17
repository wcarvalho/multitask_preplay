default_colors = {
  "reddish purple": (204 / 255, 121 / 255, 167 / 255),  # unused
  "yellow": (240 / 255, 228 / 255, 66 / 255),  # EVAL2_COLOR
  "orange": (230 / 255, 159 / 255, 0.0),  # human
  "vermillion": (213 / 255, 94 / 255, 0.0),  # preplay
  "sky blue": (86 / 255, 180 / 255, 233 / 255),  # dfs, EVAL_COLOR
  "bluish green": (0.0, 158 / 255, 115 / 255),  # reuse
  "blue": (0.0, 114 / 255, 178 / 255),  # unused
  "black": "#2f2f2e",  # unused
  "dark gray": "#666666",  # dynaq_shared, dyna
  "light gray": "#999999",  # unused
  "purple": "#CC79A7",  # qlearning
  "red": "#FD0000",  # TRAIN_COLOR
  "nice purple": "#9B80E6",  # usfa, new_path
  "pretty blue": "#679FE5",  # bfs
  "google blue": "#186CED",  # unused
  "google orange": "#FFB700",  # unused
  "white": "#FFFFFF",  # unused
}

default_colors["new_path"] = default_colors["nice purple"]
default_colors["reuse"] = default_colors["bluish green"]

TRAIN_COLOR = default_colors["sky blue"]
EVAL_COLOR = "red"
EVAL2_COLOR = "yellow"

model_colors = {
  #'human_success': '#0072B2',
  "human": default_colors["orange"],
  #'human_terminate': '#D55E00',
  "usfa": default_colors["nice purple"],
  "qlearning": default_colors["light gray"],
  # "dynaq_shared": default_colors["dark gray"],
  # "dyna": default_colors["dark gray"],
  "her": default_colors["reddish purple"],
  "dyna": default_colors["bluish green"],
  "preplay": default_colors["vermillion"],
  "bfs": default_colors["pretty blue"],
  "dfs": default_colors["sky blue"],
  "new_path": default_colors["new_path"],
  "reuse": default_colors["reuse"],
}

model_names = {
  "human": "Human",
  "qlearning": "Universal Value Function",
  "usfa": "Universal Landmark Successor Features",
  "dyna": "Dyna",
  "her": "Hindsight Experience Replay",
  "dynaq_shared": "Multitask preplay",
  "preplay": "Multitask Preplay",
  "bfs": "Breadth-first search",
  "dfs": "Depth-first search",
}


model_order = [
  "human",
  "human_success",
  "human_terminate",
  "bfs",
  "dfs",
  "preplay",
  "dyna",
  "qlearning",
  "usfa",
  "her",
  # "dynaq_shared",
]


measures = [
  "success",
  "path_length",
  "termination",
  "first_log_rt",
  "avg_log_rt",
  "total_log_rt",
  "log_avg_post_rt",
  "max_log_rt",
  "log_max_post_rt",
  "log_max_init_post_rt",
  "log_max_end_rt",
  "log_max_final_rt",
]
measure_to_title = {
  "success": "Success Rate",
  "path_length": "Path Length",
  "termination": "Task Completion Rate",
  "first_log_rt": "First Response Time",
  "avg_log_rt": "Average Response Time",
  "total_log_rt": "Total Response Time",
  "max_log_rt": "Max Response Time",
}

measure_to_ylabel = {
  "success": "Success Rate (%)",
  "path_length": "Number of Steps",
  "termination": "Completion Rate (%)",
  "first_log_rt": "Log Response Time",
  "avg_log_rt": "Log Response Time",
  "total_log_rt": "Log Response Time",
  "log_avg_post_rt": "Log Response Time",
  "max_log_rt": "Log Response Time",
  "log_max_post_rt": "Log Response Time",
  "log_max_init_post_rt": "Log Response Time",
  "log_max_end_rt": "Log Response Time",
  "log_max_final_rt": "Log Response Time",
  "first_rt": "seconds",
  "avg_rt": "seconds",
  "total_rt": "seconds",
  "avg_post_rt": "seconds",
  "max_rt": "seconds",
  "max_post_rt": "seconds",
  "max_init_post_rt": "seconds",
  "max_end_rt": "seconds",
  "max_final_rt": "seconds",
}


experiment_to_name = {
  "paths": "Two Paths",
  "shortcut": "Shortcut",
  "shortcut": "Shortcut",
}
