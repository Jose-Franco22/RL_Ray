import gymnasium as gym
from stable_baselines3 import PPO
import os
import imageio
import numpy as np

# Create training environment (no rendering)
env = gym.make("LunarLander-v3", render_mode=None)

# Create directory for GIFs if it doesn't exist
os.makedirs("eval_gifs", exist_ok=True)

# Create separate environment for rendering evaluation episodes
eval_env = gym.make("LunarLander-v3", render_mode="rgb_array")

# Initialize the model with your best hyperparameters

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.000130435,  
    n_steps=2048,
    batch_size=64,
    gamma=0.985554,
    gae_lambda=0.915363,
    clip_range=0.111322,
    ent_coef=2.5569e-05,
    verbose=1
)

TIMESTEPS = 100000  # We'll evaluate and render every this many steps
TOTAL_TIMESTEPS = 500000

# Training loop with visualization
try:
    for i in range(0, TOTAL_TIMESTEPS, TIMESTEPS):
        # Train for some timesteps
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, progress_bar=True)
        
        try:
            # Evaluate and render 3 episodes
            print("\nEvaluating and rendering 3 episodes...")
            for episode in range(3):
                frames = []
                obs, _ = eval_env.reset()
                episode_reward = 0
                done = truncated = False
                
                while not (done or truncated):
                    action, _ = model.predict(obs)
                    obs, reward, done, truncated, _ = eval_env.step(action)
                    frames.append(eval_env.render())
                    episode_reward += reward
                
                # Save the episode as a GIF
                imageio.mimsave(
                    f"eval_gifs/episode_{i+TIMESTEPS}_{episode+1}.gif",
                    frames,
                    fps=30
                )
                print(f"Episode {episode + 1} reward: {episode_reward}")
                print(f"Saved GIF: eval_gifs/episode_goodtrain_{i+TIMESTEPS}_{episode+1}.gif")
            
            # Save the model after successful evaluation
            model.save(f"lunarlander_model_{i+TIMESTEPS}")
            print(f"\nSaved model checkpoint: lunarlander_model_{i+TIMESTEPS}\n")
        
        except Exception as e:
            print(f"\nEvaluation interrupted: {e}")
            break

except KeyboardInterrupt:
    print("\nTraining interrupted by user")
except Exception as e:
    print(f"\nAn error occurred: {e}")
finally:
    # Always clean up the environments
    print("\nCleaning up...")
    try:
        env.close()
    except:
        pass
    try:
        eval_env.close()
    except:
        pass
    print("Done!")


# ╭────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
# │ Trial name              status         learning_rate     n_steps     batch_size      gamma     gae_lambda     clip_range      ent_coef     iter     total time (s)     mean_reward │
# ├────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
# │ train_ppo_3e4e6_00007   TERMINATED       2.81315e-05        1024            128   0.992152       0.952828       0.217669   0.000884828        1           107.487       -230.09    │
# │ train_ppo_3e4e6_00008   TERMINATED       0.000130435        2048             64   0.985554       0.915363       0.111322   2.5569e-05         1           131.015        184.807   │  
# ╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯


# good train

# model = PPO(
#     "MlpPolicy",
#     env,
#     learning_rate=0.000130435,  
#     n_steps=2048,
#     batch_size=64,
#     gamma=0.985554,
#     gae_lambda=0.915363,
#     clip_range=0.111322,
#     ent_coef=2.5569e-05,
#     verbose=1
# )

# bad train

# model = PPO(
#     "MlpPolicy",
#     env,
#     learning_rate=2.81315e-05,  
#     n_steps=1024,
#     batch_size=128,
#     gamma=0.992152,
#     gae_lambda=0.952828,
#     clip_range=0.217669,
#     ent_coef=0.000884828,
#     verbose=1
# )