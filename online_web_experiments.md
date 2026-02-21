

# JaxMaze

We show an example using the first experiment from the paper

```bash
# create configuration file
flyctl launch \
--dockerfile Dockerfile_jaxmaze \
--name jaxmaze-preplay \
--config flyio_configs/jaxmaze-preplay.toml \
--vm-size 'shared-cpu-4x' \
--wait-timeout "20m0s" \
--env MAN="paths" \
--yes

# deploy
flyctl deploy --config flyio_configs/jaxmaze-preplay.toml

# scale to N=10 machines across USA
flyctl scale count 10 --config flyio_configs/jaxmaze-preplay.toml --region "iad,sea,lax,den" --yes

# to see logs of run
flyctl logs --config flyio_configs/jaxmaze-preplay.toml
```

# Craftax

We show an example where the participant will not know the evaluation goal. You just need to change that environment variable if you want a different experiment.

Before running experiments, run `python experiments/craftax/extract_craftax_cache.py` to copy caches into a local directory. When uploading with Docker, we will copy the caches to avoid a computing the caches online (this takes a lot of time and memory).

```bash
# create configuration file
flyctl launch \
--dockerfile Dockerfile_craftax \
--name craftax-preplay \
--config flyio_configs/craftax-preplay.toml \
--vm-size 'performance-8x' \
--wait-timeout "20m0s" \
--env SAY_REUSE=0 \
--yes

# deploy
flyctl deploy --config flyio_configs/craftax-preplay.toml

# scale to N=10 machines across USA
flyctl scale count 10 --config flyio_configs/craftax-preplay.toml --region "iad,sea,lax,den" --yes

# to see logs of run
flyctl logs --config flyio_configs/craftax-preplay.toml
```


# Killing fly.io instances

To kill any fly.io instances, we provide a utility script`experiments/kill_flyio_apps.sh`. If it doesn't get an argument, it will kill all instances.
```bash
# kill any with name craftax
bash experiments/kill_flyio_apps.sh craftax

# kill any with name craftax or jaxmaze
bash experiments/kill_flyio_apps.sh craftax jaxmaze
```