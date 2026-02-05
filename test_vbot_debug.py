
from motrix_envs import registry
import motrix_envs.navigation.vbot.vbot_section001_np  # Ensure it is registered
import numpy as np
import traceback

def test_env():
    print("Creating environment...")
    # Initialize the environment
    try:
        env = registry.make("vbot_navigation_section001", num_envs=2)
    except Exception as e:
        print(f"Failed to create env: {e}")
        traceback.print_exc()
        return

    print("Resetting environment...")
    try:
        # NpEnv uses init_state() for the first reset, which returns NpEnvState
        state = env.init_state()
        obs = state.obs
        info = state.info
        print("Reset successful")
    except Exception as e:
        print(f"Reset failed: {e}")
        traceback.print_exc()
        return
    
    print("Stepping environment...")
    try:
        for i in range(10):
            # Create random actions
            actions = np.zeros((env.num_envs, 12), dtype=np.float32)
            
            # Step the environment
            # NpEnv.step returns NpEnvState, not the standard tuple
            state = env.step(actions)
            obs = state.obs
            reward = state.reward
            terminated = state.terminated
            truncated = state.truncated
            info = state.info
            
            print(f"Step {i+1} completed. Reward: mean={np.mean(reward):.2f}, min={np.min(reward):.2f}, max={np.max(reward):.2f}")
            print(f"Terminated: {terminated}")
    except Exception as e:
        print(f"Step failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_env()
