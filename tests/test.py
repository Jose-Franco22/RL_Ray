# -------------------------
# IMPORTS
# -------------------------
import os                               # For environment variables
import ray                              # Ray core library (used to initialize Ray runtime)
from ray import tune                     # Ray Tune API (experiment management & hyperparameter search)
from ray.tune.schedulers import ASHAScheduler  # ASHA scheduler for early stopping
import gymnasium as gym                  # Gymnasium (environment API, successor to OpenAI Gym)
from stable_baselines3 import PPO        # Stable-Baselines3 implementation of PPO algorithm

# Suppress Ray warnings
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"  # Suppress metrics exporter warning
os.environ["RAY_DEDUP_LOGS"] = "0"                      # Disable log deduplication

# -------------------------
# OPTIONAL: initialize Ray and limit resource usage
# -------------------------
# This starts a local Ray instance. If omitted, tune.run will auto-start Ray.
# Setting num_cpus limits how many CPU cores Ray can use for parallel trials.
ray.init(num_cpus=4, ignore_reinit_error=True)

# -------------------------
# TRAINING FUNCTION (executed by each Ray trial)
# -------------------------
def train_ppo(config):
    """
    This function will be executed once per trial by Ray Tune.
    'config' is a dict containing the hyperparameters for this trial.
    """
    # Create the LunarLander environment instance
    # env = gym.make("BipedalWalker-v3")
    # env = gym.make("LunarLander-v3")
    env = gym.make("InvertedDoublePendulum-v4", render_mode=None)

    # Instantiate PPO with the hyperparameters passed in 'config'
    # "MlpPolicy" = a simple fully-connected (MLP) policy network suitable for CartPole
    # verbose=0 silences Stable-Baselines3 logging for cleaner Tune output
    model = PPO(
        "MlpPolicy",              # policy architecture to use (MLP for small state spaces)
        env,                      # the Gymnasium/Env instance for training
        learning_rate=config["learning_rate"],  # optimizer learning rate
        n_steps=config["n_steps"],              # steps of experience to collect per update
        batch_size=config["batch_size"],        # minibatch size when doing gradient updates
        gamma=config["gamma"],                  # discount factor for future rewards
        gae_lambda=config["gae_lambda"],        # lambda for generalized advantage estimation (GAE)
        clip_range=config["clip_range"],        # PPO clip parameter for policy update
        ent_coef=config["ent_coef"],            # coefficient for entropy regularization
        verbose=0                               # 0=no SB3 logs, 1=info
    )

    # Train the agent for a fixed number of environment timesteps.
    # This is the "budget" each trial gets (short for tuning, long for final training).
    model.learn(total_timesteps=int(config.get("total_timesteps", 100_000)))

    # -------------------------
    # EVALUATION: run a few episodes and compute mean reward
    # -------------------------
    # We'll evaluate deterministically to get stable behavior from the learned policy.
    eval_episodes = 5                 # number of episodes to average over for evaluation
    total_reward = 0.0                # accumulator for episode rewards

    for _ in range(eval_episodes):
        # Reset the environment to get the initial observation (Gymnasium returns (obs, info))
        obs, _ = env.reset()
        done = False                   # loop control flag for episode termination
        ep_reward = 0.0                # reward accumulator for this episode

        # Step through a single episode until it terminates or is truncated
        while not done:
            # Predict action using the current policy (deterministic=True chooses best action)
            action, _ = model.predict(obs, deterministic=True)
            # Step the environment with the chosen action; Gymnasium returns many values
            obs, reward, terminated, truncated, _ = env.step(action)
            # Either 'terminated' (env ended normally) or 'truncated' (time limit) ends episode
            done = terminated or truncated
            ep_reward += float(reward)  # accumulate reward for this episode

        total_reward += ep_reward       # add this episode's reward to the total

    # Compute mean reward across the evaluation episodes
    mean_reward = total_reward / eval_episodes

    # Report the metric back to Ray Tune so it can compare trials
    # The key name "mean_reward" is what's used later in tune.run(..., metric=..., mode=...)
    # Report results using tune.report
    tune.report(metrics={"mean_reward": mean_reward})

    # Close the environment to free resources
    env.close()


# -------------------------
# HYPERPARAMETER SEARCH SPACE
# -------------------------
# We define distributions / choices for each hyperparameter we want to tune.
# Ray Tune will sample from these when launching trials.
search_space = {
    # learning_rate: log-uniform sampling between 1e-5 and 1e-3
    "learning_rate": tune.loguniform(1e-5, 1e-3),

    # n_steps: LunarLander benefits from longer rollouts
    "n_steps": tune.choice([512, 1024, 2048]),

    # batch_size: slightly larger batches for more stable learning
    "batch_size": tune.choice([64, 128, 256]),

    # gamma: LunarLander needs good long-term reward consideration
    "gamma": tune.uniform(0.98, 0.999),

    # gae_lambda: GAE parameter for advantage estimation
    "gae_lambda": tune.uniform(0.9, 0.99),

    # clip_range: standard PPO clipping range
    "clip_range": tune.uniform(0.1, 0.3),

    # ent_coef: slightly higher entropy for exploration
    "ent_coef": tune.loguniform(1e-5, 1e-3),

    # More timesteps needed for LunarLander
    "total_timesteps": 500_000
}

# -------------------------
# SCHEDULER (early stopping) - ASHA
# -------------------------
# ASHA (Asynchronous Successive Halving) reduces computation by stopping poor trials early.
scheduler = ASHAScheduler(
    # max_t should match the largest resource unit your trials will need; use timesteps here
    max_t=search_space["total_timesteps"],
    # grace_period = minimum number of training timesteps before a trial is eligible to be stopped
    grace_period=10_000,
    # reduction_factor: how aggressively to cut the number of trials at each decision point
    reduction_factor=2
)

# -------------------------
# RUN THE TUNING JOB
# -------------------------
# tune.run launches trials: each trial executes train_ppo(config) with different sampled config.
analysis = tune.run(
    train_ppo,                       # the training function to call in each trial
    config=search_space,             # the hyperparameter search space
    num_samples=12,                  # total number of trials to try (random sampling by default)
    scheduler=scheduler,             # scheduler to stop bad trials early (saves time)
    metric="mean_reward",            # metric reported by train_ppo that Tune will optimize
    mode="max",                      # we want to maximize mean_reward
    resources_per_trial={"cpu": 1},  # allocate 1 CPU per trial (change if you want parallelism)
    storage_path="file:///Users/frank/Param_Tune/ray_results",  # absolute path with file:// scheme
    verbose=1                        # verbosity level for Tune's console output
)

# -------------------------
# REPORT BEST CONFIG AND METRICS
# -------------------------
# analysis.best_config returns the hyperparameter dictionary for the best trial (by metric).
print("Best hyperparameters found:")
print(analysis.best_config)                 # show the best config (dict)



# Get the best result from the analysis
best_result = analysis.best_result
print("Best mean reward:", best_result["mean_reward"])

# -------------------------
# OPTIONAL: load & save the best model (if you saved checkpoints in train function)
# -------------------------
# Note: The simple train_ppo above did not save model checkpoints. If you want to checkpoint,
# modify train_ppo to save model files to tune.get_trial_dir() or use Tune's Checkpoint API.
#
# Example (not executed here):
# best_checkpoint = best_result.checkpoint
# model = PPO.load(path_to_checkpoint)   # would require you to save model state during training
#
# -------------------------
# CLEANUP
# -------------------------
# When you are done with experiments, shut down the Ray instance to free resources.
ray.shutdown()
