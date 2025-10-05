# rl_alpha_optimizer.py

import numpy as np
import random

# --- RL Parameters ---
MAX_EPISODES = 200        # Number of optimization trials
INITIAL_ALPHA = 0.1       # Starting guess for the embedding strength
LEARNING_RATE = 0.01      # How aggressively to update the alpha value
ALPHA_BOUNDS = (0.01, 0.3) # Min/Max physical constraints for alpha

def simulate_watermarking_performance(alpha):
    """
    Mocks the execution of the full watermarking scheme and attack simulation.
    In a real system, this function calls the DWT-SVD-CNN code and runs attacks.
    
    Args:
        alpha (float): The current embedding strength being tested.
        
    Returns:
        tuple: (PSNR value, Normalized Correlation (NC) value)
    """
    # 1. Imperceptibility (PSNR): Decreases as alpha increases
    # Simulated PSNR is high for low alpha, drops for high alpha
    psnr = 50.0 - (alpha * 70) 
    
    # 2. Robustness (NC): Increases as alpha increases (more signal, more robust)
    # Simulated NC is low for low alpha, increases for high alpha
    nc = 0.6 + (alpha * 1.5) - random.uniform(0.01, 0.05) # Add small random attack noise
    
    # Clamp results to realistic bounds
    psnr = max(30.0, min(50.0, psnr))
    nc = max(0.5, min(1.0, nc))
    
    return psnr, nc

def calculate_reward(psnr, nc):
    """
    The Reward function: A weighted score that balances Imperceptibility (PSNR) and Robustness (NC).
    
    Goal: Maximize the weighted sum.
    """
    # Weighting: 40% Imperceptibility, 60% Robustness (often robustness is prioritized)
    reward = (0.4 * psnr) + (0.6 * (nc * 100)) # Scale NC by 100 to match PSNR magnitude
    return reward

def rl_optimize_alpha():
    """
    The main Reinforcement Learning (simplified Q-learning/gradient) loop.
    """
    current_alpha = INITIAL_ALPHA
    best_reward = -np.inf
    best_alpha = current_alpha

    for episode in range(MAX_EPISODES):
        # 1. Action: Choose a new alpha (Exploration vs. Exploitation)
        # Here, we use a simple epsilon-greedy or random walk approach:
        action = random.uniform(-0.05, 0.05)
        new_alpha = np.clip(current_alpha + action, *ALPHA_BOUNDS) 
        
        # 2. Environment Step: Simulate the performance of the new alpha
        psnr, nc = simulate_watermarking_performance(new_alpha)
        reward = calculate_reward(psnr, nc)
        
        # 3. Update (Learning Step - Simplified Gradient Update)
        if reward > best_reward:
            # If the new alpha is better, shift 'current_alpha' towards 'new_alpha'
            delta_alpha = LEARNING_RATE * (new_alpha - current_alpha) 
            current_alpha += delta_alpha
            best_reward = reward
            best_alpha = new_alpha
        else:
            # If the new alpha was worse, slightly revert the change (damping)
            current_alpha -= LEARNING_RATE * (current_alpha - new_alpha)
            
        # Ensure alpha stays strictly within bounds after learning
        current_alpha = np.clip(current_alpha, *ALPHA_BOUNDS)

        if episode % 20 == 0:
            print(f"Episode {episode:3d} | Current α: {current_alpha:.4f} | PSNR: {psnr:.2f} | NC: {nc:.3f} | Reward: {reward:.2f} | Best α: {best_alpha:.4f}")

    print("\n" + "="*50)
    print(f"FINAL OPTIMAL ALPHA (α) determined by RL: {best_alpha:.4f}")
    print("="*50)
    return best_alpha

# Example Execution
# optimal_alpha = rl_optimize_alpha()
