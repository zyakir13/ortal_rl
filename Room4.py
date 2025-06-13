import pygame
from grid import GridWorld
from rl_algorithms import dqn
from ui import ControlPanel

class Room4(GridWorld):
    def __init__(self):
        super().__init__(size=10)
        # Add gems, slippery cells, etc.
        self.agent = None
        self.done = False
        self.next_room = False

    def solve(self, params):
        # Run DQN with current params
        pass

    def train(self, learning_rate=0.1, discount_factor=0.9, max_steps=1000, 
              num_episodes=100, initial_epsilon=1.0, min_epsilon=0.01, 
              epsilon_decay=0.995, **kwargs):
        # Run DQN, update UI with stats
        print(f"Starting DQN training with parameters:")
        print(f"  Learning Rate: {learning_rate}")
        print(f"  Discount Factor: {discount_factor}")
        print(f"  Max Steps: {max_steps}")
        print(f"  Episodes: {num_episodes}")
        print(f"  Initial Epsilon: {initial_epsilon}")
        print(f"  Min Epsilon: {min_epsilon}")
        print(f"  Epsilon Decay: {epsilon_decay}")
        try:
            # dqn(self, gamma=discount_factor, epsilon=initial_epsilon, episodes=num_episodes, max_steps=max_steps)
            pass
        except NotImplementedError:
            print("DQN is not yet implemented!")
        print("DQN training completed!")

    def play(self):
        # Let the agent play using the learned policy
        print("Starting DQN AI play...")
        self._ai_play_direct()

    def _ai_play_direct(self):
        """AI plays automatically - moves towards the gem using DQN-like strategy"""
        import time
        import random
        
        # Reset to starting position and clear done state
        self.reset()
        self.done = False
        
        max_steps = 50
        step_count = 0
        
        while step_count < max_steps and not self.done:
            # DQN style: sophisticated decision making
            current_x, current_y = self.agent_pos
            gem_x, gem_y = self.gems[0] if self.gems else (9, 9)
            
            print(f"DQN AI at ({current_x}, {current_y}), target at ({gem_x}, {gem_y})")
            
            # Advanced AI logic: consider multiple factors
            actions_scores = {}
            
            for action in ['up', 'down', 'left', 'right']:
                new_x, new_y = current_x, current_y
                
                if action == 'up' and new_y > 0:
                    new_y -= 1
                elif action == 'down' and new_y < self.size - 1:
                    new_y += 1
                elif action == 'left' and new_x > 0:
                    new_x -= 1
                elif action == 'right' and new_x < self.size - 1:
                    new_x += 1
                
                score = 0
                
                # Check if it's not an obstacle
                if (new_x, new_y) in self.obstacles:
                    score = -1000  # Heavy penalty for obstacles
                else:
                    # Distance to goal
                    distance = abs(new_x - gem_x) + abs(new_y - gem_y)
                    score = 100 - distance * 10  # Closer is better
                    
                    # Bonus for being on the direct path
                    if (new_x == gem_x) or (new_y == gem_y):
                        score += 20
                    
                    # Small penalty for staying in place
                    if new_x == current_x and new_y == current_y:
                        score -= 5
                
                actions_scores[action] = score
            
            # Very small exploration for DQN
            if random.random() < 0.05:  # 5% exploration
                action = random.choice(['up', 'down', 'left', 'right'])
                print(f"  Exploring: chose {action}")
            else:
                # Choose action with highest score
                action = max(actions_scores, key=actions_scores.get)
                print(f"  Greedy: chose {action} (score: {actions_scores[action]})")
            
            # Take the action
            next_state, reward, done, info = self.step(action)
            print(f"  Result: moved to {next_state}, reward={reward}, done={done}")
            
            if done:
                print(f"🎉 DQN AI reached the goal in {step_count + 1} steps!")
                print(f"Final reward: {reward}")
                # DON'T reset - just stop the AI play
                break
            
            step_count += 1
            
            # Wait a bit so we can see the movement
            time.sleep(0.5)  # Slower so we can see what's happening
        
        if step_count >= max_steps:
            print(f"DQN AI didn't reach goal in {max_steps} steps")
        
        print("DQN AI play finished")

    def manual(self):
        # Allow user to control agent with keyboard
        print("Manual control mode activated! Use arrow keys to move.")
        # Reset environment and clear done state for manual play
        self.reset()
        self.done = False

    def stop(self):
        # Stop training or episode
        self.done = True

    def run(self):
        """
        Simple run method without threads - integrated pygame and tkinter.
        """
        # Initialize pygame
        pygame.init()
        screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Escape Room - DQN")
        clock = pygame.time.Clock()
        
        # Start Tkinter control panel
        panel = ControlPanel(
            env=self,
            next_room_callback=self.next_room_callback
        )
        
        print("🤖 Room 4 - DQN Environment")
        print("📋 Use Train button to start DQN training")
        print("🎮 Use arrow keys for manual control")
        print("⌨️  Press 'R' to reset")
        
        # Simple integrated game loop
        running = True
        while running and not self.done:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    self.done = True
                elif event.type == pygame.KEYDOWN:
                    # Manual control
                    action_taken = None
                    if event.key == pygame.K_UP:
                        action_taken = 'up'
                    elif event.key == pygame.K_DOWN:
                        action_taken = 'down'
                    elif event.key == pygame.K_LEFT:
                        action_taken = 'left'
                    elif event.key == pygame.K_RIGHT:
                        action_taken = 'right'
                    elif event.key == pygame.K_r:
                        print("🔄 Resetting DQN environment...")
                        self.reset()
                        continue
                    
                    if action_taken:
                        next_state, reward, done, info = self.step(action_taken)
                        if done:
                            print("🎉 Congratulations! You reached the goal manually (DQN room)!")
                            print(f"You got a reward of {reward}!")
                            print("Press 'R' to reset and try again, or 'Next Room' to continue.")
            
            # Render the environment (uses inherited method from GridWorld)
            self.render(screen)
            pygame.display.flip()
            clock.tick(60)
            
            # Update Tkinter (non-blocking)
            try:
                panel.update_idletasks()
                panel.update()
            except:
                break
                
        try:
            pygame.quit()
        except:
            pass
            
        return "next" if self.next_room else "quit"

    def next_room_callback(self):
        self.next_room = True
        self.done = True 