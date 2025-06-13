import pygame
import numpy as np
import tkinter as tk
from grid import GridWorld
from rl_algorithms import sarsa
from ui import ControlPanel

class Room2(GridWorld):
    def __init__(self):
        super().__init__(size=10)
        
        print("🏠 Room 2 - SARSA Environment Starting...")
        
        # Design Room2 environment for SARSA learning - ROOM 2 SPECIFIC
        self._setup_sarsa_environment()
        
        # SARSA learning attributes
        self.Q_table = None
        self.policy = None
        self.training_stats = None
        self.show_policy_arrows = False
        self.is_training = False
        
        # Agent and control
        self.agent = None
        self.done = False
        self.next_room = False
        
        # AI play attributes
        self.ai_playing = False
        self.ai_step_count = 0
        self.ai_max_steps = 200
        self.ai_total_slips = 0
        self.ai_step_timer = 0
        self.ai_step_delay = 30  # frames (about 0.5 seconds at 60 FPS)
        
        # Distance-based reward system settings
        self.debug_rewards = False  # Set to True to see detailed reward breakdown
        
        print(f"🧠 Room 2 initialized for SARSA learning:")
        print(f"   Environment: 10x10 grid with slippery cells")
        print(f"   Goal position: {self.gems[0]} (ROOM 2 SPECIFIC)")
        print(f"   Slippery cells: {self.slippery_cells}")
        print(f"   Obstacles: {self.obstacles}")
        print(f"   🆓 Movement: Always FREE (consistent training and playing experience)")
        print(f"   🎯 NEW: Enhanced reward system with distance-based incentives!")
        print(f"      • -1 per step (minimize total steps)")
        print(f"      • +0.5 per distance unit closer to goal (encourage progress)")
        print(f"      • -0.2 per distance unit away from goal (discourage retreat)")
        print(f"      • +100 for reaching goal")
        print(f"   🎓 Ready for model-free learning!")

    def _setup_sarsa_environment(self):
        """
        ROOM 2 SPECIFIC GRID DESIGN
        Set up the environment for SARSA learning with obstacles and slippery cells.
        """
        print("🔧 Setting up Room 2 specific grid configuration...")
        
        # ROOM 2: Goal position - DIFFERENT FROM ROOM 1
        self.gems = [(8, 8)]
        
        # ROOM 2: Strategic slippery cell placement for interesting learning dynamics
        self.slippery_cells = [
            (3, 3), (4, 3), (5, 3),  # Horizontal slippery corridor
            (2, 6), (3, 6),          # Slippery area near path
            (6, 2), (7, 2),          # Upper slippery zone
            (1, 8), (2, 8),          # Lower slippery area
            (6, 6),                  # Single slippery cell
        ]
        
        # ROOM 2: Obstacles that create interesting path choices
        self.obstacles = [
            (2, 2), (3, 2), (4, 2),  # Horizontal wall
            (5, 4), (5, 5), (5, 6),  # Vertical wall
            (7, 4), (8, 4),          # Partial barrier
            (1, 6), (1, 7),          # Lower barrier
            (7, 7), (8, 7),          # Near-goal obstacle
        ]
        
        # Ensure starting position (0,0) is not blocked
        if (0, 0) in self.obstacles:
            self.obstacles.remove((0, 0))
        
        # Ensure goal position is not blocked
        if (8, 8) in self.obstacles:
            self.obstacles.remove((8, 8))
        
        # Update remaining gems list after changing gems
        self.remaining_gems = list(self.gems)
        self.total_gems = len(self.gems)
        
        print(f"   ✅ Room 2 grid setup complete - Goal at {self.gems[0]}")

    def solve(self, params):
        # SARSA doesn't "solve" - it learns through interaction
        pass

    def train(self, learning_rate=0.1, discount_factor=0.9, max_steps=200, 
              num_episodes=500, initial_epsilon=1.0, min_epsilon=0.01, 
              epsilon_decay=0.995, **kwargs):
        """
        Train the agent using SARSA algorithm with model-free learning.
        This method is called by the UI Train button.
        """
        print(f"🎓 Starting ROOM 2 SARSA training...")
        print(f"   Room 2 Goal: {self.gems[0]}")
        print(f"   Episodes: {num_episodes}, Learning Rate: {learning_rate}")
        print(f"   Discount Factor: {discount_factor}, Max Steps: {max_steps}")
        print(f"   Initial Epsilon: {initial_epsilon}, Min Epsilon: {min_epsilon}")
        print(f"   Epsilon Decay: {epsilon_decay}")
        
        self.is_training = True
        
        # Reset arrow logging flag for new training session
        self._arrows_drawn_logged = False
        
        # Progress callback for UI updates
        def progress_callback(episode, reward, steps, epsilon):
            if episode % 10 == 0:  # Update every 10 episodes
                print(f"   Room 2 Episode {episode}: Reward={reward:.2f}, Steps={steps}, ε={epsilon:.3f}")
        
        try:
            # Run SARSA algorithm
            print("🔄 Running SARSA algorithm for Room 2...")
            self.Q_table, self.policy, self.training_stats = sarsa(
                env=self,
                alpha=learning_rate,
                gamma=discount_factor,
                epsilon=initial_epsilon,
                episodes=num_episodes,
                max_steps=max_steps,
                callback=progress_callback
            )
            
            self.is_training = False
            print("✅ Room 2 SARSA algorithm completed!")
            
            # Extract complete policy ensuring all empty/slippery cells have actions
            print("🔍 Extracting complete Room 2 policy...")
            self._extract_complete_policy()
            
            # Policy arrows are optional - use 'P' key to toggle if desired
            self.show_policy_arrows = False
            print("💡 Room 2 Policy learned! Press 'P' to toggle policy arrows if needed")
            
            # Analyze learned policy
            print("📊 Analyzing Room 2 learned policy...")
            self._analyze_learned_policy()
            
            # Debug: Print complete policy grid
            print("🗒️  Printing Room 2 debug policy grid...")
            self._debug_print_policy_grid()
            
            print("✅ Room 2 SARSA training completed successfully!")
            
        except Exception as e:
            self.is_training = False
            print(f"❌ Room 2 Training failed with error: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            raise

    def _extract_complete_policy(self):
        """
        Extract policy ensuring ALL empty and slippery cells have actions.
        Handle border cases and ensure no None actions.
        """
        if self.Q_table is None:
            print("❌ No Q-table available for Room 2 policy extraction")
            return
        
        actions = ['up', 'down', 'left', 'right']
        self.policy = {}
        
        print("🔍 Extracting complete Room 2 policy for all empty and slippery cells...")
        
        missing_actions = []
        border_fixes = []
        
        for x in range(self.size):
            for y in range(self.size):
                state = (x, y)
                
                # Skip obstacles and goals - they don't need policy actions
                if state in self.obstacles or state in self.gems:
                    continue
                
                # For all other cells (empty + slippery), ensure they have actions
                best_action = None
                best_value = float('-inf')
                
                # Get valid actions for this cell (not going outside borders)
                valid_actions = self._get_valid_actions(x, y)
                
                if not valid_actions:
                    print(f"⚠️  WARNING: No valid actions for Room 2 cell {state}")
                    continue
                
                # Find best action among valid ones
                for action in valid_actions:
                    q_value = self.Q_table.get((state, action), 0.0)
                    if q_value > best_value:
                        best_value = q_value
                        best_action = action
                
                # If all Q-values are zero or no action found, choose default valid action
                if best_action is None:
                    best_action = valid_actions[0]  # Default to first valid action
                    missing_actions.append(state)
                
                # Check if we had to avoid border
                all_actions_q = [self.Q_table.get((state, a), 0.0) for a in actions]
                if len(valid_actions) < len(actions) and len(all_actions_q) > 0 and max(all_actions_q) > best_value:
                    border_fixes.append(state)
                
                self.policy[state] = best_action
        
        # Report results
        if missing_actions:
            print(f"📝 Room 2: {len(missing_actions)} cells had zero Q-values, assigned default actions")
        if border_fixes:
            print(f"🚧 Room 2: {len(border_fixes)} cells had border restrictions applied")
        
        # Verify all empty/slippery cells have actions
        self._verify_policy_completeness()

    def _get_valid_actions(self, x, y):
        """
        Get valid actions for a cell that don't go outside grid borders.
        """
        valid_actions = []
        
        if y > 0:  # Can go up
            valid_actions.append('up')
        if y < self.size - 1:  # Can go down
            valid_actions.append('down')
        if x > 0:  # Can go left
            valid_actions.append('left')
        if x < self.size - 1:  # Can go right
            valid_actions.append('right')
        
        return valid_actions

    def _verify_policy_completeness(self):
        """
        Verify that all empty and slippery cells have policy actions.
        """
        missing_policies = []
        empty_count = 0
        slippery_count = 0
        
        for x in range(self.size):
            for y in range(self.size):
                state = (x, y)
                
                # Check if it's an empty or slippery cell
                if state not in self.obstacles and state not in self.gems:
                    if state in self.slippery_cells:
                        slippery_count += 1
                    else:
                        empty_count += 1
                    
                    # Check if policy exists
                    if state not in self.policy or self.policy[state] is None:
                        missing_policies.append(state)
                        print(f"⚠️  WARNING: Missing Room 2 policy for cell {state}")
        
        if missing_policies:
            print(f"❌ CRITICAL: Room 2 - {len(missing_policies)} cells missing policies: {missing_policies}")
        else:
            print(f"✅ Room 2 Policy verification complete:")
            print(f"   • Empty cells with policy: {empty_count}")
            print(f"   • Slippery cells with policy: {slippery_count}")
            print(f"   • Total learnable cells: {empty_count + slippery_count}")

    def _debug_print_policy_grid(self):
        """
        Print the complete policy grid for debugging.
        """
        print("\n🔍 Room 2 Complete Policy Grid:")
        print("   Legend: ↑=up, ↓=down, ←=left, →=right, #=obstacle, G=goal, .=empty")
        
        action_symbols = {
            'up': '↑',
            'down': '↓', 
            'left': '←',
            'right': '→'
        }
        
        for y in range(self.size):
            row = ""
            for x in range(self.size):
                state = (x, y)
                if state in self.obstacles:
                    row += "#"
                elif state in self.gems:
                    row += "G"
                elif state in self.policy:
                    action = self.policy[state]
                    row += action_symbols.get(action, "?")
                else:
                    row += "."
            print(f"   {row}")

    def _analyze_learned_policy(self):
        """
        Analyze the learned policy and provide insights.
        """
        if not self.policy or not self.training_stats:
            return
        
        print("\n📊 Room 2 SARSA Learning Analysis:")
        
        # Training performance
        episode_rewards = self.training_stats['episode_rewards']
        episode_steps = self.training_stats['episode_steps']
        
        print(f"   Room 2 Training Performance:")
        print(f"   • Total episodes: {len(episode_rewards)}")
        print(f"   • Final average reward: {np.mean(episode_rewards[-50:]):.2f}")
        print(f"   • Final average steps: {np.mean(episode_steps[-50:]):.1f}")
        print(f"   • Final exploration rate: {self.training_stats['final_epsilon']:.3f}")
        
        # Policy coverage
        states_with_policy = len(self.policy)
        total_learnable_states = 0
        
        for x in range(self.size):
            for y in range(self.size):
                state = (x, y)
                if state not in self.obstacles and state not in self.gems:
                    total_learnable_states += 1
        
        coverage = (states_with_policy / total_learnable_states) * 100 if total_learnable_states > 0 else 0
        print(f"   Room 2 Policy Coverage: {coverage:.1f}% ({states_with_policy}/{total_learnable_states} states)")

    def play(self):
        """
        Let the AI play with free movement (like during training).
        No longer requires a trained policy - AI moves freely.
        """
        print("🤖 Room 2 AI will play with free movement...")
        print("   AI uses goal-directed movement with exploration (like during training)")
        print("   No longer restricted to policy arrows - consistent with training experience!")
        
        # Initialize AI play state
        self.ai_playing = True
        self.ai_step_count = 0
        self.ai_max_steps = 200  # Maximum steps before giving up
        self.ai_total_slips = 0
        self.ai_step_timer = 0
        self.ai_step_delay = 30  # frames (about 0.5 seconds at 60 FPS)
        
        # Reset to starting position for AI play
        self.reset()

    def _ai_play_step(self):
        """Execute one step of AI play with free movement (like during training)."""
        if not self.ai_playing:
            return
        
        try:
            current_state = self.agent_pos
            
            # AI plays with free movement - chooses actions freely like during training
            # This ensures consistency between training and playing environments
            available_actions = ['up', 'down', 'left', 'right']
            
            # Simple strategy: move toward goal with some randomness (like epsilon-greedy)
            goal_pos = self.gems[0] if self.gems else (8, 8)
            current_x, current_y = current_state
            goal_x, goal_y = goal_pos
            
            # Calculate best directions toward goal
            best_actions = []
            if goal_x > current_x and current_x < self.size - 1:  # Can go right
                best_actions.append('right')
            elif goal_x < current_x and current_x > 0:  # Can go left
                best_actions.append('left')
            
            if goal_y > current_y and current_y < self.size - 1:  # Can go down
                best_actions.append('down')
            elif goal_y < current_y and current_y > 0:  # Can go up
                best_actions.append('up')
            
            # If no best actions (already at goal row/column), try any valid action
            if not best_actions:
                if current_x > 0:
                    best_actions.append('left')
                if current_x < self.size - 1:
                    best_actions.append('right')
                if current_y > 0:
                    best_actions.append('up')
                if current_y < self.size - 1:
                    best_actions.append('down')
            
            # Choose action with some exploration (like during training)
            import random
            if random.random() < 0.3:  # 30% exploration
                action = random.choice(available_actions)
                print(f"🎲 Room 2 AI at {current_state} exploring: {action}")
            else:
                action = random.choice(best_actions) if best_actions else random.choice(available_actions)
                print(f"🎯 Room 2 AI at {current_state} moving toward goal: {action}")
            
            # Take the chosen action (free movement)
            next_state, reward, done, info = self.step(action)
            
            # Check if action slipped
            if info.get('slipped', False):
                self.ai_total_slips += 1
                actual_action = info.get('actual_action', action)
                print(f"   💫 SLIPPED! Intended {action}, actually moved {actual_action}")
                print(f"   📍 Ended up at {next_state}, reward: {reward}")
            else:
                print(f"   ✅ Moved successfully to {next_state}, reward: {reward}")
            
            if done:
                if next_state in self.gems:
                    print(f"🎉 Room 2 AI reached the goal in {self.ai_step_count + 1} steps!")
                    print(f"   Total slips during episode: {self.ai_total_slips}")
                    print(f"   Final reward: {reward}")
                self.ai_playing = False
                return
            
            self.ai_step_count += 1
            
            if self.ai_step_count >= self.ai_max_steps:
                print(f"⏰ Room 2 AI didn't reach goal in {self.ai_max_steps} steps")
                print(f"   Total slips during episode: {self.ai_total_slips}")
                self.ai_playing = False
                return
                
        except Exception as e:
            print(f"❌ Room 2 AI play error: {e}")
            self.ai_playing = False

    def manual(self):
        """
        Enable manual control mode.
        """
        print("🕹️  Room 2 Manual control activated!")
        print("   Use arrow keys to explore and learn the Room 2 environment")
        print("   🆓 Full free movement enabled - arrows show learned policy for reference")
            
        self.reset()
        self.done = False
        self.ai_playing = False

    def stop(self):
        """Stop training or episode - but keep the environment open."""
        print("🛑 Stopping Room 2 current operation...")
        if self.is_training:
            print("   Stopping training process...")
            self.is_training = False
        elif self.ai_playing:
            print("   Stopping AI play...")
            self.ai_playing = False
        else:
            print("   No active training or AI play to stop.")
        # Don't set self.done = True - this would close the entire room!
        # Keep the environment open so user can interact with trained agent

    def step(self, action):
        """
        Enhanced step method with distance-based reward system.
        
        Room 2 Reward System:
        - Base step penalty: -1 per step
        - Slipping penalty: additional -2 
        - Goal reward: +100 for reaching destination
        - NEW: Distance-based progress reward: +0.5 for getting closer, -0.2 for moving away
        
        This prevents the agent from staying in place to avoid negative rewards.
        """
        # Store current position and distance to goal before moving
        current_state = self.agent_pos
        goal_position = self.gems[0] if self.gems else (8, 8)  # Room 2 goal at (8,8)
        current_distance = self._calculate_manhattan_distance(current_state, goal_position)
        
        # Room 2: Always allow free movement (no policy restrictions)
        # This ensures consistency between training and playing environments
        
        # Execute the movement using parent's step method
        next_state, base_reward, done, info = super().step(action)
        
        # Calculate distance-based reward modification
        if not done:  # Only apply distance reward if game isn't over
            new_distance = self._calculate_manhattan_distance(next_state, goal_position)
            distance_change = current_distance - new_distance  # Positive if got closer
            
            # Distance-based reward: encourage progress toward goal
            if distance_change > 0:
                # Got closer to goal - positive reinforcement
                distance_reward = 0.5 * distance_change
                print(f"📈 Room 2: Moved closer to goal! Distance reward: +{distance_reward:.1f}")
            elif distance_change < 0:
                # Moved away from goal - small penalty
                distance_reward = 0.2 * distance_change  # This will be negative
                print(f"📉 Room 2: Moved away from goal. Distance penalty: {distance_reward:.1f}")
            else:
                # Same distance (moved parallel or stayed in place)
                distance_reward = 0
            
            # Combine base reward with distance-based reward
            total_reward = base_reward + distance_reward
            
            # Debug output for reward components
            if hasattr(self, 'debug_rewards') and self.debug_rewards:
                print(f"🧮 Room 2 Reward breakdown:")
                print(f"   Base reward: {base_reward}")
                print(f"   Distance reward: {distance_reward:.1f}")  
                print(f"   Total reward: {total_reward:.1f}")
                print(f"   Distance: {current_distance} → {new_distance}")
        else:
            # Game is done (reached goal or hit obstacle) - use base reward only
            total_reward = base_reward
            if next_state in self.gems:
                print(f"🎯 Room 2: Reached goal! Final reward: {total_reward}")
        
        return next_state, total_reward, done, info
    
    def _calculate_manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def render(self, screen):
        """
        Render the environment without automatic policy arrows.
        """
        try:
            # Draw base environment
            super().render(screen)
            
            # Only draw policy arrows if explicitly enabled by user (P key)
            if self.show_policy_arrows and self.policy is not None:
                self.draw_policy_arrows(screen)
            
            # Draw training indicator
            if self.is_training:
                self._draw_training_indicator(screen)
            
            # Draw AI play indicator
            if self.ai_playing:
                self._draw_ai_play_indicator(screen)
                
        except Exception as e:
            print(f"❌ Room 2 Render error: {e}")

    def draw_policy_arrows(self, screen):
        """
        Draw arrows showing the learned SARSA policy in ALL empty and slippery cells.
        Skip obstacles - they don't need arrows.
        """
        if self.policy is None:
            return
        
        try:
            arrows_drawn = 0
            missing_arrows = []
            
            # Draw arrows for accessible cells only
            for x in range(self.size):
                for y in range(self.size):
                    state = (x, y)
                    
                    # Skip obstacles and goals - no arrows needed
                    if state in self.obstacles or state in self.gems:
                        continue
                    
                    # All other cells (empty + slippery) should have arrows
                    action = self.policy.get(state)
                    
                    if action is None:
                        missing_arrows.append(state)
                        print(f"⚠️  WARNING: Missing arrow for cell {state}")
                        # Draw default arrow as fallback
                        action = 'right'  # Default direction
                    
                    # Draw arrow for this cell
                    self._draw_policy_arrow(screen, x, y, action, state)
                    arrows_drawn += 1
            
            # Debug output for missing arrows (only once)
            if missing_arrows and arrows_drawn == 0:  # Only report on first draw
                print(f"⚠️  {len(missing_arrows)} cells missing arrows: {missing_arrows}")
            
            # Only print arrow count once (not every frame)
            if not hasattr(self, '_arrows_drawn_logged') or not self._arrows_drawn_logged:
                print(f"✅ Drew {arrows_drawn} arrows in Room 2 (skipped {len(self.obstacles)} obstacles + {len(self.gems)} goals)")
                self._arrows_drawn_logged = True
                
        except Exception as e:
            print(f"❌ Room 2 Arrow drawing error: {e}")

    def _draw_policy_arrow(self, screen, x, y, action, state):
        """
        Draw a single policy arrow showing the learned action.
        Ensures no arrows point outside grid boundaries.
        """
        try:
            # Validate that the action doesn't point outside boundaries
            valid_actions = self._get_valid_actions(x, y)
            if action not in valid_actions:
                print(f"⚠️  Fixing invalid arrow: {action} at {state}, using {valid_actions[0] if valid_actions else 'right'}")
                action = valid_actions[0] if valid_actions else 'right'
            
            # Calculate cell center
            center_x = x * self.cell_size + self.cell_size // 2
            center_y = y * self.cell_size + self.cell_size // 2
            
            # Color coding for different cell types
            if state in self.slippery_cells:
                # Slippery cells: Cyan arrows (learned policy in risky areas)
                arrow_color = (0, 255, 255)         # Cyan
                outline_color = (0, 0, 0)           # Black outline
                arrow_size = 16
                thickness = 3
            else:
                # Regular cells: Green arrows (learned policy)
                arrow_color = (0, 200, 0)           # Green
                outline_color = (255, 255, 255)     # White outline
                arrow_size = 14
                thickness = 2
            
            # Calculate arrow geometry
            half_size = arrow_size // 2
            head_size = 5
            
            if action == 'up':
                start_pos = (center_x, center_y + half_size)
                end_pos = (center_x, center_y - half_size)
                head_points = [
                    end_pos,
                    (end_pos[0] - head_size, end_pos[1] + head_size + 1),
                    (end_pos[0] + head_size, end_pos[1] + head_size + 1)
                ]
            elif action == 'down':
                start_pos = (center_x, center_y - half_size)
                end_pos = (center_x, center_y + half_size)
                head_points = [
                    end_pos,
                    (end_pos[0] - head_size, end_pos[1] - head_size - 1),
                    (end_pos[0] + head_size, end_pos[1] - head_size - 1)
                ]
            elif action == 'left':
                start_pos = (center_x + half_size, center_y)
                end_pos = (center_x - half_size, center_y)
                head_points = [
                    end_pos,
                    (end_pos[0] + head_size + 1, end_pos[1] - head_size),
                    (end_pos[0] + head_size + 1, end_pos[1] + head_size)
                ]
            elif action == 'right':
                start_pos = (center_x - half_size, center_y)
                end_pos = (center_x + half_size, center_y)
                head_points = [
                    end_pos,
                    (end_pos[0] - head_size - 1, end_pos[1] - head_size),
                    (end_pos[0] - head_size - 1, end_pos[1] + head_size)
                ]
            else:
                # Invalid action - draw a default right arrow with warning color
                print(f"⚠️  Invalid action '{action}' for cell {state}")
                arrow_color = (255, 0, 0)  # Red for invalid
                start_pos = (center_x - half_size, center_y)
                end_pos = (center_x + half_size, center_y)
                head_points = [
                    end_pos,
                    (end_pos[0] - head_size - 1, end_pos[1] - head_size),
                    (end_pos[0] - head_size - 1, end_pos[1] + head_size)
                ]
            
            # Draw outline for visibility
            pygame.draw.line(screen, outline_color, start_pos, end_pos, thickness + 2)
            pygame.draw.polygon(screen, outline_color, head_points)
            
            # Draw main arrow
            pygame.draw.line(screen, arrow_color, start_pos, end_pos, thickness)
            pygame.draw.polygon(screen, arrow_color, head_points)
            
        except Exception as e:
            print(f"❌ Error drawing arrow at {state}: {e}")

    def _draw_training_indicator(self, screen):
        """Draw indicator when training is in progress."""
        try:
            font = pygame.font.Font(None, 36)
            text = font.render("ROOM 2 TRAINING...", True, (255, 255, 0))
            screen.blit(text, (self.game_width + 10, 50))
        except Exception as e:
            print(f"❌ Error drawing Room 2 training indicator: {e}")

    def _draw_ai_play_indicator(self, screen):
        """Draw indicator when AI is playing."""
        try:
            font = pygame.font.Font(None, 36)
            text = font.render("ROOM 2 AI PLAYING...", True, (0, 255, 0))
            screen.blit(text, (self.game_width + 10, 90))
            
            # Show step count
            step_font = pygame.font.Font(None, 24)
            step_text = step_font.render(f"Step: {self.ai_step_count}/{self.ai_max_steps}", True, (0, 200, 0))
            screen.blit(step_text, (self.game_width + 10, 120))
            
            # Show slip count
            slip_text = step_font.render(f"Slips: {self.ai_total_slips}", True, (0, 200, 0))
            screen.blit(slip_text, (self.game_width + 10, 140))
        except Exception as e:
            print(f"❌ Error drawing Room 2 AI play indicator: {e}")

    def run(self):
        """
        Simple run method without threads - integrated pygame and tkinter.
        """
        print("🚪 Room 2 starting run method...")
        
        try:
            # Initialize pygame
            pygame.init()
            screen = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption("Escape Room - Room 2 - SARSA Learning")
            clock = pygame.time.Clock()
            
            # Start Tkinter control panel
            panel = ControlPanel(
                env=self,
                next_room_callback=self.next_room_callback
            )
            
            print("🧠 Room 2 - SARSA Learning Environment")
            print("📋 Use Train button to start SARSA learning")
            print("🎮 Use arrow keys for manual control (FREE MOVEMENT)")
            print("🤖 Use Play button for AI demonstration (FREE MOVEMENT)")
            print("⌨️  Press 'P' to toggle policy arrows (optional visualization)")
            print("⌨️  Press 'D' to toggle debug rewards (see distance-based rewards)")
            print("🆓 All movement is FREE - training and playing use same mechanics")
            print("🎯 NEW: Distance-based rewards encourage progress toward goal!")
            print("🚪 Use Next Room button when ready to continue")
            
            # Simple integrated game loop
            running = True
            panel_alive = True
            
            while running and not self.done and panel_alive:
                try:
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
                                print("🔄 Resetting Room 2 SARSA environment...")
                                self.reset()
                                continue
                            elif event.key == pygame.K_p:
                                # Toggle policy arrows
                                if self.policy is not None:
                                    self.show_policy_arrows = not self.show_policy_arrows
                                    if self.show_policy_arrows:
                                        print("✨ Room 2 Policy arrows enabled")
                                    else:
                                        print("🚫 Room 2 Policy arrows disabled")
                                else:
                                    print("❌ No Room 2 policy learned! Train first.")
                                continue
                            elif event.key == pygame.K_d:
                                # Toggle debug rewards
                                self.debug_rewards = not self.debug_rewards
                                if self.debug_rewards:
                                    print("🐛 Room 2 Debug rewards enabled - you'll see detailed reward breakdown")
                                else:
                                    print("🔇 Room 2 Debug rewards disabled")
                                continue
                            elif event.key == pygame.K_n:
                                # Debug: Next room with keyboard
                                print("🔑 Keyboard shortcut: Next room")
                                self.next_room_callback()
                                continue
                            
                            if action_taken:
                                print(f"🎮 Room 2 Manual action: {action_taken}")
                                next_state, reward, done, info = self.step(action_taken)
                                if info.get('slipped', False):
                                    print(f"💫 Slipped! Intended {action_taken}, moved {info.get('actual_action')}")
                                if info.get('blocked', False):
                                    print(f"🚫 Move blocked! Policy suggests {info.get('policy_action')}")
                                if done:
                                    print("🎉 Congratulations! Manual goal reached in Room 2!")
                                    print(f"Reward: {reward}")
                                    print("Press 'R' to reset, 'P' to toggle arrows, 'D' to toggle debug rewards, 'N' for next room")
                    
                    # Check if we should move to next room
                    if self.next_room:
                        print("🚪 Next room flag detected!")
                        break
                    
                    # Handle AI play step timing
                    if self.ai_playing:
                        self.ai_step_timer += 1
                        if self.ai_step_timer >= self.ai_step_delay:
                            self.ai_step_timer = 0
                            self._ai_play_step()
                    
                    # Render the environment
                    self.render(screen)
                    pygame.display.flip()
                    clock.tick(60)
                    
                    # Update Tkinter (non-blocking) - CRITICAL FOR BUTTON RESPONSE
                    try:
                        if panel.winfo_exists():
                            panel.update_idletasks()
                            panel.update()
                        else:
                            print("🚪 Panel no longer exists!")
                            panel_alive = False
                            break
                            
                    except tk.TclError:
                        # Panel was destroyed
                        print("🚪 Panel destroyed!")
                        panel_alive = False
                        break
                    except Exception as e:
                        print(f"❌ Panel update error: {e}")
                        panel_alive = False
                        break
                        
                except Exception as e:
                    print(f"❌ Room 2 Game loop error: {e}")
                    break
                    
        except Exception as e:
            print(f"❌ Room 2 Run method error: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
        finally:
            try:
                pygame.quit()
            except:
                pass
                
        print(f"🚪 Room 2 ending - next_room: {self.next_room}, done: {self.done}")
        result = "next" if self.next_room else "quit"
        print(f"🚪 Room 2 returning: {result}")
        return result

    def next_room_callback(self):
        print("🚪 next_room_callback called in Room 2!")
        print("   Room 2 SARSA Learning mission complete!")
        self.next_room = True
        self.done = True
        print(f"   Setting flags - next_room: {self.next_room}, done: {self.done}")