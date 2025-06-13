import numpy as np
import random

def value_iteration(env, gamma=0.9, theta=1e-4):
    """
    Value Iteration algorithm that computes optimal policy for all states.
    
    Args:
        env: GridWorld environment
        gamma: Discount factor
        theta: Convergence threshold
    
    Returns:
        V: Value function (dict mapping states to values)
        policy: Optimal policy (dict mapping states to actions)
    """
    # Initialize value function for ALL cells
    V = {}
    for x in range(env.size):
        for y in range(env.size):
            V[(x, y)] = 0.0
    
    actions = ['up', 'down', 'left', 'right']
    
    iteration = 0
    while True:
        delta = 0
        
        # Update value for ALL cells
        for x in range(env.size):
            for y in range(env.size):
                state = (x, y)
                v = V[state]
                
                # Calculate Q-values for all actions from ALL states
                q_values = []
                for action in actions:
                    q_value = 0.0
                    
                    # Get transitions for this state-action pair
                    transitions = get_transitions_for_all_states(env, state, action)
                    
                    for next_state, prob, reward in transitions:
                        q_value += prob * (reward + gamma * V[next_state])
                    
                    q_values.append(q_value)
                
                # Bellman update for ALL states
                V[state] = max(q_values)
                delta = max(delta, abs(v - V[state]))
        
        iteration += 1
        if delta < theta:
            print(f"Value Iteration converged in {iteration} iterations")
            break
        if iteration > 1000:
            print(f"Value Iteration stopped at {iteration} iterations (max reached)")
            break
    
    # Extract optimal policy for ALL states
    policy = {}
    for x in range(env.size):
        for y in range(env.size):
            state = (x, y)
            
            # Find best action for this state
            best_action = None
            best_value = float('-inf')
            
            for action in actions:
                q_value = 0.0
                transitions = get_transitions_for_all_states(env, state, action)
                
                for next_state, prob, reward in transitions:
                    q_value += prob * (reward + gamma * V[next_state])
                
                if q_value > best_value:
                    best_value = q_value
                    best_action = action
            
            # Store policy for this cell
            policy[state] = best_action
    
    # Verify policies for empty and slippery cells specifically
    empty_cells = []
    slippery_cells = []
    missing_policies = []
    
    for x in range(env.size):
        for y in range(env.size):
            state = (x, y)
            
            # Check if it's an empty cell (not obstacle, not goal)
            if state not in env.obstacles and state not in env.gems:
                if state in env.slippery_cells:
                    slippery_cells.append(state)
                else:
                    empty_cells.append(state)
                
                # Check if policy exists for this cell
                if state not in policy or policy[state] is None:
                    missing_policies.append(state)
    
    if missing_policies:
        print(f"ERROR: Missing policies for cells: {missing_policies}")
        raise ValueError(f"Missing policies for {len(missing_policies)} cells")
    
    print(f"✅ Policy computed for {len(empty_cells)} empty cells and {len(slippery_cells)} slippery cells")
    return V, policy

def sarsa(env, alpha=0.1, gamma=0.9, epsilon=0.1, episodes=1000, max_steps=200, callback=None):
    """
    SARSA (State-Action-Reward-State-Action) algorithm for model-free learning.
    
    Args:
        env: GridWorld environment
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Initial exploration rate
        episodes: Number of episodes to train
        max_steps: Maximum steps per episode
        callback: Optional callback function for progress updates
    
    Returns:
        Q: Q-table (state-action values)
        policy: Learned policy
        training_stats: Training statistics
    """
    actions = ['up', 'down', 'left', 'right']
    
    # Initialize Q-table for all state-action pairs
    Q = {}
    for x in range(env.size):
        for y in range(env.size):
            for action in actions:
                Q[((x, y), action)] = 0.0
    
    # Training statistics
    episode_rewards = []
    episode_steps = []
    epsilon_values = []
    
    print(f"🎓 Starting SARSA training...")
    print(f"   Alpha (learning rate): {alpha}")
    print(f"   Gamma (discount): {gamma}")
    print(f"   Initial epsilon: {epsilon}")
    print(f"   Episodes: {episodes}")
    print(f"   Max steps per episode: {max_steps}")
    
    initial_epsilon = epsilon
    min_epsilon = 0.01
    epsilon_decay = 0.995
    
    for episode in range(episodes):
        # Reset environment for new episode
        state = env.reset()
        total_reward = 0
        steps = 0
        
        # Choose initial action using epsilon-greedy
        action = epsilon_greedy_action(Q, state, epsilon, actions)
        
        while steps < max_steps:
            # Take action and observe result
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                # Terminal state - update Q-value with no next action
                Q[(state, action)] += alpha * (reward - Q[(state, action)])
                break
            else:
                # Choose next action using epsilon-greedy
                next_action = epsilon_greedy_action(Q, next_state, epsilon, actions)
                
                # SARSA update: Q(s,a) += alpha * [r + gamma * Q(s',a') - Q(s,a)]
                Q[(state, action)] += alpha * (
                    reward + gamma * Q[(next_state, next_action)] - Q[(state, action)]
                )
                
                # Move to next state and action
                state = next_state
                action = next_action
        
        # Record episode statistics
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        epsilon_values.append(epsilon)
        
        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        # Progress reporting
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_steps = np.mean(episode_steps[-50:])
            print(f"   Episode {episode + 1}/{episodes}: Avg reward = {avg_reward:.2f}, Avg steps = {avg_steps:.1f}, ε = {epsilon:.3f}")
        
        # Call progress callback if provided
        if callback:
            callback(episode + 1, total_reward, steps, epsilon)
    
    # Extract policy from Q-table
    policy = {}
    for x in range(env.size):
        for y in range(env.size):
            state = (x, y)
            best_action = None
            best_value = float('-inf')
            
            for action in actions:
                q_value = Q.get((state, action), 0.0)
                if q_value > best_value:
                    best_value = q_value
                    best_action = action
            
            policy[state] = best_action
    
    # Training statistics
    training_stats = {
        'episode_rewards': episode_rewards,
        'episode_steps': episode_steps,
        'epsilon_values': epsilon_values,
        'final_epsilon': epsilon,
        'total_episodes': episodes
    }
    
    print(f"✅ SARSA training completed!")
    print(f"   Final epsilon: {epsilon:.3f}")
    print(f"   Average reward (last 50 episodes): {np.mean(episode_rewards[-50:]):.2f}")
    print(f"   Average steps (last 50 episodes): {np.mean(episode_steps[-50:]):.1f}")
    
    return Q, policy, training_stats

def epsilon_greedy_action(Q, state, epsilon, actions):
    """
    Choose action using epsilon-greedy strategy.
    
    Args:
        Q: Q-table
        state: Current state
        epsilon: Exploration probability
        actions: Available actions
    
    Returns:
        action: Selected action
    """
    if random.random() < epsilon:
        # Explore: choose random action
        return random.choice(actions)
    else:
        # Exploit: choose best action
        best_action = None
        best_value = float('-inf')
        
        for action in actions:
            q_value = Q.get((state, action), 0.0)
            if q_value > best_value:
                best_value = q_value
                best_action = action
        
        return best_action if best_action is not None else random.choice(actions)

def get_transitions_for_all_states(env, state, action):
    """
    Get transitions from ANY state including obstacles, goals, slippery cells.
    """
    x, y = state
    
    # Goal states - terminal but still have transitions for policy computation
    if state in env.gems:
        return [(state, 1.0, 0)]  # Stay in goal with 0 reward
    
    # Obstacle states - heavily penalized but still have transitions
    if state in env.obstacles:
        next_state, reward = compute_next_state_and_reward(env, state, action, from_obstacle=True)
        return [(next_state, 1.0, reward)]
    
    # Slippery cells - stochastic transitions
    if state in env.slippery_cells:
        intended_prob = 0.6
        slip_prob = 0.1
        
        transitions = []
        actions = ['up', 'down', 'left', 'right']
        for actual_action in actions:
            prob = intended_prob if actual_action == action else slip_prob
            next_state, reward = compute_next_state_and_reward(env, state, actual_action)
            transitions.append((next_state, prob, reward))
        return transitions
    
    # Regular empty cells - deterministic transitions
    next_state, reward = compute_next_state_and_reward(env, state, action)
    return [(next_state, 1.0, reward)]

def compute_next_state_and_reward(env, state, action, from_obstacle=False):
    """Compute next state and reward for any state-action pair."""
    x, y = state
    new_x, new_y = x, y
    
    # Calculate intended next position
    if action == 'up' and y > 0:
        new_y = y - 1
    elif action == 'down' and y < env.size - 1:
        new_y = y + 1
    elif action == 'left' and x > 0:
        new_x = x - 1
    elif action == 'right' and x < env.size - 1:
        new_x = x + 1
    
    next_state = (new_x, new_y)
    
    # Check collision with obstacles
    if next_state in env.obstacles:
        next_state = state  # Stay in place
    
    # Calculate rewards
    if from_obstacle:
        reward = -100  # Heavy penalty for obstacle states
    elif state in env.gems:
        reward = 0     # No reward for staying in goal
    else:
        reward = -1    # Step cost
        if state in env.slippery_cells:
            reward -= 2  # Slip penalty
        if next_state in env.gems:
            reward = 100  # Goal reward
    
    return next_state, reward

def q_learning(env, alpha, gamma, epsilon, episodes, max_steps, callback=None):
    # Q-Learning implementation
    pass

def dqn(env, gamma=0.9, epsilon=0.1, episodes=1000, max_steps=1000, callback=None):
    # DQN implementation (use PyTorch or TensorFlow)
    pass
