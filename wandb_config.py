ENTITY = "wcarvalho92"

# Default metric keys (for craftax-multigoal)
DEFAULT_KEYS = {
  "train_key": "actor_performance/0.episode_return",
  "test_key": "evaluator_performance/0.episode_return",
  "train_step_key": "actor_performance/num_actor_steps",
  "test_step_key": "evaluator_performance/num_actor_steps",
}

# Housemaze metric keys (multiple test keys for different environments)
HOUSEMAZE_KEYS = {
  "train_key": "actor_performance/0.0 avg_episode_return",
  "test_keys": [
    ("evaluator_performance/0.1 0.test \n two_paths - AvgReturn", "Two Paths"),
    ("evaluator_performance/0.1 0.test \n shortcut - AvgReturn", "Shortcut"),
  ],
  "train_step_key": "actor_performance/num_actor_steps",
  "test_step_key": "evaluator_performance/num_actor_steps",
}

# Preset configurations for different experiments
jaxmaze = {
  "main": {
      "qlearning": "Group: ql-final-epsilon-1",
      "usfa": "Group: usfa-final-epsilon-1",
      "dyna": "Group: dyna-final-epsilon-1",
      "her": "Group: her-pnas-revision-2",
      "preplay": "Group: preplay-pnas-revision-5",
  },
  "preplay-all-goals-abalation": {
    "preplay": "Group: preplay-pnas-revision-5",
    'peng': "Group: preplay-peng-ablation-pnas-revision-2",
    'cql': "Group: preplay-cql-ablation-pnas-revision-2",
    'all_goals': "Group: preplay-all-goals-ablation-pnas-revision-2",
  },
  "her-all-goals-abalation": {
    "with": "Name: alg=her,all_=1.0,tota=20000000,exp=her_test_big",
    'without': "Name: alg=her,all_=0.0,tota=20000000,exp=her_test_big",
    "keys": DEFAULT_KEYS,
  },
}