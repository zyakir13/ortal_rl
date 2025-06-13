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
        self.restrict_movement = False  # Optional movement restriction
        
        # Agent and control
        self.agent = None
        self.done = False
        self.next_room = False
        
        print(f"🧠 Room 2 initialized for SARSA learning:")
        print(f"   Environment: 10x10 grid with slippery cells")
        print(f"   Goal position: {self.gems[0]} (ROOM 2 SPECIFIC)")
        print(f"   Slippery cells: {self.slippery_cells}")
        print(f"   Obstacles: {self.obstacles}")
        print(f"   🆓 Movement: FREE (no restrictions until training + 'M' key)")
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
            
            # Automatically show policy arrows after training
            self.show_policy_arrows = True
            print("✨ Room 2 Policy arrows automatically enabled after training!")
            
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
        Let the agent play using the learned SARSA policy.
        This method is called by the UI Play button.
        """
        if self.policy is None:
            print("❌ No Room 2 policy learned! Please train using SARSA first.")
            return
        
        print("🤖 Room 2 AI will play using learned SARSA policy...")
        print("   Watch the agent follow the learned policy arrows in Room 2!")
        
        # Reset to starting position for AI play
        self.reset()

    def manual(self):
        """
        Enable manual control mode.
        """
        print("🕹️  Room 2 Manual control activated!")
        print("   Use arrow keys to explore and learn the Room 2 environment")
        if self.restrict_movement and self.policy:
            print("   🚫 Movement restricted to policy arrows")
        else:
            print("   🆓 Free movement enabled")
            
        self.reset()
        self.done = False

    def stop(self):
        """Stop training or episode."""
        print("🛑 Stopping Room 2 current operation...")
        self.done = True
        self.is_training = False

    def step(self, action):
        """
        Override step to optionally implement policy-restricted movement.
        In Room2, movement is FREE by default - restriction only when explicitly enabled.
        """
        current_state = self.agent_pos
        
        # Only restrict movement if ALL conditions are met:
        # 1. Policy exists (after training)
        # 2. Not currently training
        # 3. Movement restriction is explicitly enabled (user pressed 'M')
        if (self.policy is not None and 
            not self.is_training and 
            hasattr(self, 'restrict_movement') and 
            self.restrict_movement):
            
            policy_action = self.policy.get(current_state)
            if policy_action is not None and action != policy_action:
                print(f"🚫 Room 2 Movement blocked! Policy suggests {policy_action}, attempted {action}")
                return current_state, -0.5, False, {"blocked": True, "policy_action": policy_action}
        
        # Default: Allow free movement (standard SARSA behavior)
        return super().step(action)

    def render(self, screen):
        """
        Enhanced render method with automatic policy visualization.
        """
        try:
            # Draw base environment
            super().render(screen)
            
            # Draw policy arrows if enabled and policy exists
            if self.show_policy_arrows and self.policy is not None:
                self.draw_policy_arrows(screen)
            
            # Draw training indicator
            if self.is_training:
                self._draw_training_indicator(screen)
                
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
            
            print(f"✅ Drew {arrows_drawn} arrows in Room 2 (skipped {len(self.obstacles)} obstacles + {len(self.gems)} goals)")
                
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
            print("⌨️  Press 'P' to toggle policy arrows (after training)")
            print("⌨️  Press 'M' to toggle movement restriction (after training)")
            print("🆓 Movement is FREE until you train and press 'M'")
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
                            elif event.key == pygame.K_m:
                                # Toggle movement restriction
                                if self.policy is not None:
                                    self.restrict_movement = not self.restrict_movement
                                    if self.restrict_movement:
                                        print("🚫 Room 2 Movement restriction enabled - follow arrows only!")
                                    else:
                                        print("🆓 Room 2 Free movement enabled")
                                else:
                                    print("❌ No Room 2 policy learned! Train first.")
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
                                    print("Press 'R' to reset, 'P' to toggle arrows, 'M' to toggle movement restriction, 'N' for next room")
                    
                    # Check if we should move to next room
                    if self.next_room:
                        print("🚪 Next room flag detected!")
                        break
                    
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