import pygame
import math
from grid import GridWorld
from rl_algorithms import value_iteration
from ui import ControlPanel
import tkinter as tk

class Room1(GridWorld):
    def __init__(self):
        super().__init__(size=10)
        
        print("🏠 Room 1 - Dynamic Programming (Value Iteration) Starting...")
        
        # Strategic grid design with multiple path options - ROOM 1 SPECIFIC
        self._design_strategic_grid()
        
        # Value Iteration specific attributes
        self.value_function = None
        self.policy = None
        self.show_policy_arrows = True  # Always show arrows in Room 1
        self.is_training = False
        
        # AI play attributes
        self.ai_playing = False
        self.ai_step_count = 0
        self.ai_max_steps = 100
        self.ai_total_slips = 0
        self.ai_step_timer = 0
        self.ai_step_delay = 48  # frames (about 0.8 seconds at 60 FPS)
        
        # Add gems, slippery cells, etc.
        self.agent = None  # Not needed for value iteration, but keep for interface
        self.done = False
        self.next_room = False
        
        print(f"🎯 Room 1 initialized for Dynamic Programming:")
        print(f"   Environment: 10x10 grid with slippery cells")
        print(f"   Goal position: {self.gems[0]} (ROOM 1 SPECIFIC)")
        print(f"   Slippery cells: {self.slippery_cells}")
        print(f"   Obstacles: {self.obstacles}")
        print(f"   🎓 Model-based learning - Computing optimal policy immediately!")
        
        # Compute policy immediately at initialization - no training button needed
        self._initialize_policy()

    def _design_strategic_grid(self):
        """
        ROOM 1 SPECIFIC GRID DESIGN
        Design the grid with three distinct paths to goal at (8, 5):
        1. Short risky path through slippery cells
        2. Medium path around obstacles  
        3. Long safe path via edges
        """
        print("🔧 Setting up Room 1 specific grid configuration...")
        
        # ROOM 1: Set goal position to (8, 5) - DIFFERENT FROM OTHER ROOMS
        self.gems = [(8, 5)]
        
        # ROOM 1: Short risky path: Slippery cells leading to goal
        self.slippery_cells = [
            (6, 4), (7, 4), (8, 4),  # Short risky path to goal
            (2, 2), (3, 7),          # Additional slippery areas for complexity
            (4, 7), (3, 5)           # More slippery cells for challenge
        ]
        
        # ROOM 1: Strategic obstacle placement to create distinct paths
        self.obstacles = [
            # Main obstacle that forces medium path
            (5, 3),
            
            # Obstacles to channel traffic into the three path types
            (4, 2), (6, 2),          # Forces traffic around or through slippery area
            (3, 4), (4, 4),          # Creates bottleneck for medium path
            (6, 6), (7, 6),          # Blocks some direct routes
            (2, 5), (3, 5),          # Creates interesting choices in mid-grid
            (1, 8), (2, 8),          # Guides toward specific routes
            
            # Corner obstacles to make safe paths longer
            (8, 1), (9, 2),          # Top-right corner obstacles
            (1, 1),                  # Top-left obstacle
        ]
        
        # Ensure starting position (0,0) is not blocked
        if (0, 0) in self.obstacles:
            self.obstacles.remove((0, 0))
        
        # Ensure goal position is not blocked
        if (8, 5) in self.obstacles:
            self.obstacles.remove((8, 5))
        
        # Update remaining gems list after changing gems
        self.remaining_gems = list(self.gems)
        self.total_gems = len(self.gems)
        
        print("🗺️  Room 1 Grid designed with three strategic paths:")
        print("   📍 Path 1 (Short & Risky): (0,0) → ... → (6,4) → (7,4) → (8,4) → (8,5)")
        print("   📍 Path 2 (Medium): (0,0) → around obstacle (5,3) → (8,5)")
        print("   📍 Path 3 (Long & Safe): (0,0) → via top/left edges → (8,5)")
        print(f"   ✅ Room 1 grid setup complete - Goal at {self.gems[0]}")

    def _initialize_policy(self):
        """
        Initialize policy immediately at game start.
        Creates a challenging path that forces movement through slippery cells.
        """
        print("🎯 Computing optimal policy for Room 1 immediately...")
        print("   Using Dynamic Programming (Value Iteration)...")
        
        try:
            # Compute optimal policy using Value Iteration
            self.value_function, self.policy = value_iteration(self, gamma=0.9, theta=1e-4)
            
            # Validate and fix any arrows that point outside the grid
            self._validate_and_fix_arrows()
            
            # Analyze the computed policy
            self._analyze_computed_paths()
            
            print("✅ Room 1 Policy computed and arrows are now visible!")
            print("🎮 Movement is FREE - use arrow keys to explore")
            print("🤖 Use Play button to watch AI follow optimal policy")
            
        except Exception as e:
            print(f"❌ Error computing Room 1 policy: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")

    def solve(self, params):
        # Policy is already computed at initialization
        print("✅ Room 1 policy already computed - no solve needed!")
        pass

    def train(self, learning_rate=0.1, discount_factor=0.9, max_steps=1000, 
              num_episodes=100, initial_epsilon=1.0, min_epsilon=0.01, 
              epsilon_decay=0.995, **kwargs):
        """
        Room 1 doesn't need training - policy is computed immediately.
        This method just confirms the policy is ready.
        """
        print("🎯 Room 1 uses Dynamic Programming - no training needed!")
        print("   Policy is already computed and visible.")
        print("   Use arrow keys for manual control or Play button for AI.")
        
        if self.policy is None:
            print("   Recomputing policy...")
            self._initialize_policy()

    def _validate_and_fix_arrows(self):
        """
        Ensure no arrows point outside the grid boundaries.
        Fix any invalid arrows to point in valid directions.
        """
        try:
            print("🔧 Validating Room 1 arrows...")
            fixed_count = 0
            
            for x in range(self.size):
                for y in range(self.size):
                    state = (x, y)
                    
                    action = self.policy.get(state)
                    if action is None:
                        continue
                    
                    # Check if arrow points outside grid and fix if needed
                    valid_actions = []
                    
                    # Check each direction for validity
                    if y > 0:  # Can go up
                        valid_actions.append('up')
                    if y < self.size - 1:  # Can go down
                        valid_actions.append('down')
                    if x > 0:  # Can go left
                        valid_actions.append('left')
                    if x < self.size - 1:  # Can go right
                        valid_actions.append('right')
                    
                    # If current action is invalid, replace with a valid one
                    if action not in valid_actions and valid_actions:
                        old_action = action
                        # Prefer the original action if possible, otherwise use first valid
                        self.policy[state] = valid_actions[0]
                        fixed_count += 1
                        print(f"🔧 Fixed Room 1 arrow at {state}: {old_action} → {valid_actions[0]}")
            
            if fixed_count > 0:
                print(f"🔧 Fixed {fixed_count} Room 1 arrows that pointed outside grid boundaries")
            else:
                print("✅ All Room 1 arrows point to valid grid positions")
                
        except Exception as e:
            print(f"❌ Error validating Room 1 arrows: {e}")

    def _analyze_computed_paths(self):
        """
        Analyze which path the Value Iteration algorithm chose as optimal for Room 1.
        """
        if not self.policy:
            return
        
        try:
            # Trace the policy from start to see which path it prefers
            print("\n🔍 Analyzing Room 1 computed optimal path...")
            
            current_pos = (0, 0)
            path = [current_pos]
            visited = set()
            
            # Follow the policy arrows to see the preferred path
            for step in range(50):  # Limit to prevent infinite loops
                if current_pos in visited:
                    print(f"   ⚠️  Detected loop at {current_pos}")
                    break
                
                visited.add(current_pos)
                
                if current_pos in self.gems:
                    print(f"   🎯 Reached Room 1 goal at {current_pos}")
                    break
                
                action = self.policy.get(current_pos)
                if not action:
                    print(f"   ❌ No policy action at {current_pos}")
                    break
                
                # Calculate next position
                x, y = current_pos
                if action == 'up' and y > 0:
                    next_pos = (x, y - 1)
                elif action == 'down' and y < self.size - 1:
                    next_pos = (x, y + 1)
                elif action == 'left' and x > 0:
                    next_pos = (x - 1, y)
                elif action == 'right' and x < self.size - 1:
                    next_pos = (x + 1, y)
                else:
                    print(f"   🚫 Invalid action {action} at {current_pos}")
                    break
                
                # Check for obstacle collision
                if next_pos in self.obstacles:
                    next_pos = current_pos  # Stay in place
                
                path.append(next_pos)
                current_pos = next_pos
            
            # Analyze path characteristics
            slippery_encounters = len([pos for pos in path if pos in self.slippery_cells])
            path_length = len(path)
            
            print(f"   📊 Room 1 Optimal path analysis:")
            print(f"      - Path length: {path_length} steps")
            print(f"      - Slippery cells encountered: {slippery_encounters}")
            print(f"      - Path preview: {' → '.join(map(str, path[:8]))}{'...' if len(path) > 8 else ''}")
            
            # Determine which strategy was chosen
            if (6, 4) in path or (7, 4) in path or (8, 4) in path:
                print(f"   🥇 Room 1 Algorithm chose: SHORT RISKY PATH (through slippery cells)")
            elif any(abs(pos[0] - 5) <= 1 and abs(pos[1] - 3) <= 1 for pos in path):
                print(f"   🥈 Room 1 Algorithm chose: MEDIUM PATH (around obstacles)")
            else:
                print(f"   🥉 Room 1 Algorithm chose: LONG SAFE PATH (via edges)")
                
        except Exception as e:
            print(f"❌ Error analyzing Room 1 paths: {e}")

    def play(self):
        """Let the agent play using the computed policy"""
        if self.policy is None:
            print("❌ No Room 1 policy available! Recomputing...")
            self._initialize_policy()
            if self.policy is None:
                return
        
        print("🤖 Starting Room 1 AI play with optimal policy...")
        print("   Watch the agent follow the computed optimal policy arrows in Room 1!")
        
        # Initialize AI play
        self.ai_playing = True
        self.ai_step_count = 0
        self.ai_total_slips = 0
        self.ai_step_timer = 0
        
        # Reset to starting position
        self.reset()

    def _ai_play_step(self):
        """Execute one step of AI play"""
        if not self.ai_playing or self.policy is None:
            return
        
        try:
            current_state = self.agent_pos
            
            # Get optimal action from policy
            optimal_action = self.policy.get(current_state)
            
            if optimal_action is None:
                print(f"❌ No Room 1 policy action for state {current_state}")
                self.ai_playing = False
                return
            
            # Check if currently on slippery cell
            on_slippery = current_state in self.slippery_cells
            if on_slippery:
                print(f"🧊 Room 1 AI on slippery cell at {current_state} - following policy action: {optimal_action}")
            else:
                print(f"🎯 Room 1 AI at {current_state} following policy action: {optimal_action}")
            
            # Take the optimal action
            next_state, reward, done, info = self.step(optimal_action)
            
            # Check if action slipped
            if info.get('slipped', False):
                self.ai_total_slips += 1
                actual_action = info.get('actual_action', optimal_action)
                print(f"   💫 SLIPPED! Policy suggested {optimal_action}, actually moved {actual_action}")
                print(f"   📍 Ended up at {next_state}, reward: {reward}")
            else:
                print(f"   ✅ Moved successfully to {next_state}, reward: {reward}")
            
            if done:
                if next_state in self.gems:
                    print(f"🎉 Room 1 AI reached the goal in {self.ai_step_count + 1} steps!")
                    print(f"   Total slips during episode: {self.ai_total_slips}")
                    print(f"   Final reward: {reward}")
                self.ai_playing = False
                return
            
            self.ai_step_count += 1
            
            if self.ai_step_count >= self.ai_max_steps:
                print(f"⏰ Room 1 AI didn't reach goal in {self.ai_max_steps} steps")
                print(f"   Total slips during episode: {self.ai_total_slips}")
                self.ai_playing = False
                return
                
        except Exception as e:
            print(f"❌ Room 1 AI play error: {e}")
            self.ai_playing = False

    def manual(self):
        """Allow user to control agent with keyboard"""
        print("🕹️  Room 1 Manual control mode...")
        print("   Use arrow keys to explore Room 1 environment")
        print("   Policy arrows show the optimal action for each cell")
        
        self.reset()
        self.done = False
        self.ai_playing = False

    def stop(self):
        """Stop training or episode"""
        print("🛑 Stopping Room 1 current operation...")
        self.done = True
        self.is_training = False
        self.ai_playing = False

    def step(self, action):
        """
        Override step to implement proper reward function for Room 1:
        +10 for reaching goal
        -1 for each step
        -10 for hitting obstacle (though this shouldn't happen in our grid)
        """
        # Store original position
        original_pos = self.agent_pos
        
        # Call parent step method
        next_state, reward, done, info = super().step(action)
        
        # Override reward function for Room 1 specifications
        if done and next_state in self.gems:
            # +10 for reaching goal (override parent's +100)
            reward = 10
        elif next_state in self.obstacles:
            # -10 for hitting obstacle (though shouldn't happen)
            reward = -10
        else:
            # -1 for each step (override parent's -1 or -3 for slipping)
            reward = -1
        
        return next_state, reward, done, info

    def render(self, screen):
        """Enhanced render method with policy arrows"""
        try:
            # Call parent render method
            super().render(screen)
            
            # Always draw policy arrows if policy exists (Room 1 always shows arrows)
            if self.policy is not None:
                self.draw_policy_arrows(screen)
                
            # Draw AI play indicator
            if self.ai_playing:
                self._draw_ai_play_indicator(screen)
                
        except Exception as e:
            print(f"❌ Room 1 Render error: {e}")

    def draw_policy_arrows(self, screen):
        """
        Draw a policy arrow in accessible cells of the Room 1 grid.
        Skip obstacles - they don't need arrows.
        """
        try:
            if self.policy is None:
                return
                
            arrows_drawn = 0
            
            # Draw arrow in accessible cells only
            for x in range(self.size):
                for y in range(self.size):
                    state = (x, y)
                    
                    # Skip obstacles - no arrows needed in walls
                    if state in self.obstacles:
                        continue
                    
                    # Get policy action
                    action = self.policy.get(state)
                    if action is None:
                        # Skip cells without policy
                        continue
                    
                    # Draw arrow
                    self._draw_arrow_in_cell(screen, x, y, action, state)
                    arrows_drawn += 1
                    
        except Exception as e:
            print(f"❌ Room 1 Arrow drawing error: {e}")

    def _draw_arrow_in_cell(self, screen, x, y, action, state):
        """
        Draw a single arrow in the specified cell.
        """
        try:
            # Cell center coordinates
            center_x = x * self.cell_size + self.cell_size // 2
            center_y = y * self.cell_size + self.cell_size // 2
            
            # Arrow styling based on cell type
            if state in self.gems:
                arrow_color = (255, 215, 0)      # Gold for goals
                arrow_size = 18
                thickness = 4
            elif state in self.slippery_cells:
                arrow_color = (100, 200, 255)    # Blue for slippery
                arrow_size = 16
                thickness = 3
            else:
                arrow_color = (60, 60, 60)       # Gray for regular
                arrow_size = 14
                thickness = 2
            
            # Calculate arrow coordinates based on action
            if action == 'up':
                start_pos = (center_x, center_y + arrow_size // 2)
                end_pos = (center_x, center_y - arrow_size // 2)
                head_points = [
                    end_pos,
                    (end_pos[0] - 6, end_pos[1] + 10),
                    (end_pos[0] + 6, end_pos[1] + 10)
                ]
            elif action == 'down':
                start_pos = (center_x, center_y - arrow_size // 2)
                end_pos = (center_x, center_y + arrow_size // 2)
                head_points = [
                    end_pos,
                    (end_pos[0] - 6, end_pos[1] - 10),
                    (end_pos[0] + 6, end_pos[1] - 10)
                ]
            elif action == 'left':
                start_pos = (center_x + arrow_size // 2, center_y)
                end_pos = (center_x - arrow_size // 2, center_y)
                head_points = [
                    end_pos,
                    (end_pos[0] + 10, end_pos[1] - 6),
                    (end_pos[0] + 10, end_pos[1] + 6)
                ]
            elif action == 'right':
                start_pos = (center_x - arrow_size // 2, center_y)
                end_pos = (center_x + arrow_size // 2, center_y)
                head_points = [
                    end_pos,
                    (end_pos[0] - 10, end_pos[1] - 6),
                    (end_pos[0] - 10, end_pos[1] + 6)
                ]
            else:
                # Invalid action - draw a default up arrow
                print(f"WARNING: Invalid action '{action}' for Room 1 cell ({x}, {y}), drawing up arrow")
                start_pos = (center_x, center_y + arrow_size // 2)
                end_pos = (center_x, center_y - arrow_size // 2)
                head_points = [
                    end_pos,
                    (end_pos[0] - 6, end_pos[1] + 10),
                    (end_pos[0] + 6, end_pos[1] + 10)
                ]
            
            # Draw white outline for visibility
            pygame.draw.line(screen, (255, 255, 255), start_pos, end_pos, thickness + 2)
            pygame.draw.polygon(screen, (255, 255, 255), head_points)
            
            # Draw the main arrow
            pygame.draw.line(screen, arrow_color, start_pos, end_pos, thickness)
            pygame.draw.polygon(screen, arrow_color, head_points)
            
        except Exception as e:
            print(f"❌ Error drawing Room 1 arrow at ({x}, {y}): {e}")

    def _draw_ai_play_indicator(self, screen):
        """Draw indicator when AI is playing."""
        try:
            font = pygame.font.Font(None, 36)
            text = font.render("ROOM 1 AI PLAYING...", True, (0, 255, 0))
            screen.blit(text, (self.game_width + 10, 90))
            
            # Show step count
            step_font = pygame.font.Font(None, 24)
            step_text = step_font.render(f"Step: {self.ai_step_count}/{self.ai_max_steps}", True, (0, 200, 0))
            screen.blit(step_text, (self.game_width + 10, 120))
            
            # Show slip count
            slip_text = step_font.render(f"Slips: {self.ai_total_slips}", True, (0, 200, 0))
            screen.blit(slip_text, (self.game_width + 10, 140))
        except Exception as e:
            print(f"❌ Error drawing Room 1 AI play indicator: {e}")

    def run(self):
        """
        Simple run method without threads - integrated pygame and tkinter.
        """
        print("🚪 Room 1 starting run method...")
        
        try:
            # Initialize pygame
            pygame.init()
            screen = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption("Escape Room - Room 1 - Dynamic Programming")
            clock = pygame.time.Clock()
            
            # Start Tkinter control panel
            panel = ControlPanel(
                env=self,
                next_room_callback=self.next_room_callback
            )
            
            print("🎯 Room 1 - Dynamic Programming Environment")
            print("✅ Optimal policy already computed and displayed!")
            print("🎮 Use arrow keys for manual control in Room 1")
            print("⌨️  Press 'P' to toggle policy arrows")
            print("🤖 Use Play button to watch AI follow optimal policy")
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
                                print("🔄 Resetting Room 1 environment...")
                                self.reset()
                                self.ai_playing = False
                                continue
                            elif event.key == pygame.K_p:
                                # Toggle policy arrows
                                self.show_policy_arrows = not self.show_policy_arrows
                                if self.show_policy_arrows:
                                    print("✨ Room 1 Policy arrows enabled")
                                else:
                                    print("🚫 Room 1 Policy arrows disabled")
                                continue
                            elif event.key == pygame.K_n:
                                # Debug: Next room with keyboard
                                print("🔑 Keyboard shortcut: Next room")
                                self.next_room_callback()
                                continue
                            
                            if action_taken and not self.ai_playing:
                                print(f"🎮 Room 1 Manual action: {action_taken}")
                                next_state, reward, done, info = self.step(action_taken)
                                if info.get('slipped', False):
                                    print(f"💫 Slipped! Intended {action_taken}, moved {info.get('actual_action')}")
                                if done:
                                    print("🎉 Congratulations! You reached the Room 1 goal manually!")
                                    print(f"You got a reward of {reward}!")
                                    print("Press 'R' to reset and try again, 'N' for next room, or use Next Room button.")
                    
                    # Handle AI play steps
                    if self.ai_playing:
                        self.ai_step_timer += 1
                        if self.ai_step_timer >= self.ai_step_delay:
                            self.ai_step_timer = 0
                            self._ai_play_step()
                    
                    # Render the environment
                    self.render(screen)
                    pygame.display.flip()
                    clock.tick(60)
                    
                    # Check if we should move to next room
                    if self.next_room:
                        print("🚪 Next room flag detected!")
                        break
                    
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
                    print(f"❌ Room 1 Game loop error: {e}")
                    break
                    
        except Exception as e:
            print(f"❌ Room 1 Run method error: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
        finally:
            try:
                pygame.quit()
            except:
                pass
                
        print(f"🚪 Room 1 ending - next_room: {self.next_room}, done: {self.done}")
        result = "next" if self.next_room else "quit"
        print(f"🚪 Room 1 returning: {result}")
        return result

    def next_room_callback(self):
        print("🚪 next_room_callback called in Room 1!")
        print("   Room 1 Dynamic Programming mission complete!")
        self.next_room = True
        self.done = True
        print(f"   Setting flags - next_room: {self.next_room}, done: {self.done}")