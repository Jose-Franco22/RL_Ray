import gymnasium as gym
from stable_baselines3 import PPO
import os
import imageio
import numpy as np

# --------------------------
# Environment Setup
# --------------------------
# Create training environment (no rendering)
env = gym.make("InvertedDoublePendulum-v5", render_mode=None)

# Create directory for GIFs if it doesn't exist
os.makedirs("eval_gifs", exist_ok=True)

# Create separate environment for rendering evaluation episodes
eval_env = gym.make("InvertedDoublePendulum-v5", render_mode="rgb_array")

# --------------------------
# PPO Model Setup
# --------------------------
# These are good baseline PPO hyperparameters for MuJoCo tasks
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.0006391158186508456,
    n_steps=512,
    batch_size=64,
    gamma=0.9944558114973807,
    gae_lambda=0.9683476045485316,
    clip_range=0.2088467139930176,
    ent_coef=0.0005697693027318373,
    verbose=1,
)

# --------------------------
# Training Parameters
# --------------------------
TIMESTEPS = 100000       # Evaluate every 50k steps
TOTAL_TIMESTEPS = 500000  # Total training steps

# --------------------------
# Training Loop with Evaluation
# --------------------------
try:
    for i in range(0, TOTAL_TIMESTEPS, TIMESTEPS):
        # Train for some timesteps
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, progress_bar=True)

        try:
            # Evaluate episodes
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

                    # Collect frames for GIF only at the end
                    if is_final_eval:
                        frame = eval_env.render()
                        if frame is not None:
                            frames.append(frame)
                    episode_reward += reward

                # Save GIF for the final evaluation
                if is_final_eval and frames:
                    gif_path = f"eval_gifs/IDP_Bad_episode_{i+TIMESTEPS}_{episode+1}.gif"
                    imageio.mimsave(gif_path, frames, fps=30)
                    print(f"Saved GIF: {gif_path}")

                print(f"Episode {episode + 1} reward: {episode_reward}")

        except Exception as e:
            print(f"\nEvaluation interrupted: {e}")
            break

except KeyboardInterrupt:
    print("\nTraining interrupted by user")
except Exception as e:
    print(f"\nAn error occurred: {e}")
finally:
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


# model = PPO(
#     "MlpPolicy",
#     env,
#     learning_rate=0.0006391158186508456,
#     n_steps=512,
#     batch_size=64,
#     gamma=0.9944558114973807,
#     gae_lambda=0.9683476045485316,
#     clip_range=0.2088467139930176,
#     ent_coef=0.0005697693027318373,
#     verbose=1,
# )

# model = PPO(
#     "MlpPolicy",
#     env,
#     learning_rate=0.0001462619256600089,
#     n_steps=512,
#     batch_size=256,
#     gamma=0.991042510701005,
#     gae_lambda=0.9770318514041544,
#     clip_range=0.22229709913373819,
#     ent_coef=2.409072209176715e-05,
#     verbose=1,
# )