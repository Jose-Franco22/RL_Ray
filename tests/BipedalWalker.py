import gymnasium as gym
from stable_baselines3 import PPO
import os
import imageio
import numpy as np

# Create training environment (no rendering)
env = gym.make("BipedalWalker-v3", render_mode=None)

# Create directory for GIFs if it doesn't exist
os.makedirs("eval_gifs", exist_ok=True)

# Create separate environment for rendering evaluation episodes
eval_env = gym.make("BipedalWalker-v3", render_mode="rgb_array")

# Initialize the model with your best hyperparameters

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=4.72936e-05,  
    n_steps=512,
    batch_size=128,
    gamma=0.988612,
    gae_lambda=0.962573,
    clip_range=0.128221,
    ent_coef=7.86519e-05,
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
            # Evaluate episodes (only render and save GIFs on final evaluation)
            is_final_eval = (i + TIMESTEPS) == TOTAL_TIMESTEPS
            print("\nEvaluating episodes..." + (" (saving GIFs)" if is_final_eval else ""))
            for episode in range(3):
                frames = [] if is_final_eval else None
                obs, _ = eval_env.reset()
                episode_reward = 0
                done = truncated = False

                while not (done or truncated):
                    action, _ = model.predict(obs)
                    obs, reward, done, truncated, _ = eval_env.step(action)
                    # Only collect frames for the GIF on the final evaluation
                    if is_final_eval:
                        frame = eval_env.render()
                        if frame is not None:
                            frames.append(frame)
                    episode_reward += reward

                # Save GIF only for final evaluation
                if is_final_eval and frames:
                    imageio.mimsave(
                        f"eval_gifs/BW_Bad_episode_{i+TIMESTEPS}_{episode+1}.gif",
                        frames,
                        fps=30
                    )
                    print(f"Saved GIF: eval_gifs/BW_episode_{i+TIMESTEPS}_{episode+1}.gif")

                print(f"Episode {episode + 1} reward: {episode_reward}")

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

# │ Trial name              status         learning_rate     n_steps     batch_size      gamma     gae_lambda     clip_range      ent_coef     iter     total time (s)     mean_reward │
# ├────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
# │ train_ppo_a3280_00001   TERMINATED       4.72936e-05         512            128   0.988612       0.962573       0.128221   7.86519e-05        1            140.742      -125.123   │
# │ train_ppo_a3280_00008   TERMINATED       0.000103201        2048             64   0.985347       0.971848       0.176941   9.9616e-05         1            165.047       275.409   │


# bad train

# model = PPO(
#     "MlpPolicy",
#     env,
#     learning_rate=4.72936e-05,  
#     n_steps=512,
#     batch_size=128,
#     gamma=0.988612,
#     gae_lambda=0.962573,
#     clip_range=0.128221,
#     ent_coef=7.86519e-05,
#     verbose=1
# )


# good train

# model = PPO(
#     "MlpPolicy",
#     env,
#     batch_size = 64,
#     clip_range= 0.17694099064829574,
#     ent_coef= 9.961601842755075e-05,
#     gae_lambda= 0.971848371585926,
#     gamma= 0.9853471809710711,
#     learning_rate= 0.00010320116617031678,
#     n_steps= 2048,
#     total_timesteps= 500000
# )

