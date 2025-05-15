# Multitask Preplay (AI Simulations)


## Install

### For re-running results

### For Development

Adding jaxneurorl in case you want to change library
```
git submodule init
git submodule update
```

```
mamba create -n preplay-ai python=3.10 pip wheel -y

# in case a mamba env is already active
mamba deactivate
mamba activate preplay-ai
mamba env update -f conda_envs/dev.yaml
pip install -e libraries/jaxneurorl
```

VSCODE: add `libraries/jaxneurorl` to `python.autoComplete.extraPaths`
