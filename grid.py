import pygame
import tkinter as tk
# import threading  # No longer needed - removed threading
import random
import math

class GridWorld:
    def __init__(self, size=10):
        self.size = size
        self.grid = [[0 for _ in range(size)] for _ in range(size)]
        self.agent_pos = (0, 0)
        self.gems = [(8, 8)]  # Add a gem at position (8,8)
        self.obstacles = [(3, 3), (4, 4), (5, 5)]  # Add some obstacles
        
        # Slippery cells - cells where movement is probabilistic
        self.slippery_cells = [(2, 2), (5, 1), (6, 3), (1, 7), (7, 6)]
        self.slip_probability = 0.4  # 40% chance to slip when on slippery cell
        
        self.cell_size = 50
        
        # Enhanced state tracking
        self.collected_gems = 0
        self.total_gems = len(self.gems)
        self.current_reward = 0
        self.total_reward = 0
        self.episode_steps = 0
        self.is_on_special_cell = False
        self.special_cell_type = ""
        self.remaining_gems = list(self.gems)  # Copy of gems list
        self.last_action_slipped = False  # Track if last action resulted in a slip
        
        # Slip animation state
        self.slip_animation_active = False
        self.slip_animation_timer = 0
        self.slip_animation_duration = 30  # frames (about 0.5 seconds at 60 FPS)
        self.slip_start_pos = None
        self.slip_end_pos = None
        self.slip_direction = None
        
        # Status panel settings
        self.status_width = 250
        self.game_width = self.size * self.cell_size
        self.window_width = self.game_width + self.status_width
        self.window_height = self.size * self.cell_size
        
        self.reset()

    def reset(self):
        self.agent_pos = (0, 0)
        self.collected_gems = 0
        self.current_reward = 0
        self.total_reward = 0
        self.episode_steps = 0
        self.is_on_special_cell = False
        self.special_cell_type = ""
        self.remaining_gems = list(self.gems)  # Reset remaining gems
        self.last_action_slipped = False  # Reset last action slipped flag
        self.slip_animation_active = False
        self.slip_animation_timer = 0
        self.slip_animation_duration = 30  # frames (about 0.5 seconds at 60 FPS)
        self.slip_start_pos = None
        self.slip_end_pos = None
        self.slip_direction = None
        return self.agent_pos

    def step(self, action):
        """
        Execute an action in the environment.
        
        Slippery Cell Movement Logic:
        - On normal cells: deterministic movement (100% chance intended action succeeds)
        - On slippery cells: probabilistic movement
          * 60% chance: intended action works as planned
          * 40% chance: agent "slips" and moves in a random direction
            - Each of the 4 directions (up/down/left/right) has equal 10% probability
        
        The slip probability is designed to simulate icy or wet surfaces where
        the agent loses control and slides in an unintended direction.
        """
        # Move agent, handle collisions, rewards, etc.
        x, y = self.agent_pos
        original_action = action
        
        # Check if current position is slippery before moving
        is_on_slippery = self.agent_pos in self.slippery_cells
        action_slipped = False
        
        # Apply slippery cell logic
        if is_on_slippery and random.random() < self.slip_probability:
            # Agent slips! Choose random direction instead of intended action
            possible_actions = ['up', 'down', 'left', 'right']
            action = random.choice(possible_actions)
            action_slipped = True
            print(f"🧊 SLIP! Intended: {original_action}, Actually moved: {action}")
        
        # Define movement based on (possibly modified) action
        new_x, new_y = x, y
        if action == 'up' and y > 0:
            new_y = y - 1
        elif action == 'down' and y < self.size - 1:
            new_y = y + 1
        elif action == 'left' and x > 0:
            new_x = x - 1
        elif action == 'right' and x < self.size - 1:
            new_x = x + 1
        
        # Check if new position is valid (not an obstacle)
        if (new_x, new_y) not in self.obstacles:
            self.agent_pos = (new_x, new_y)
        else:
            # If slipped into obstacle, stay in place but still count as slipped
            print(f"🚫 Slipped into obstacle at ({new_x}, {new_y}), staying at ({x}, {y})")
        
        # Calculate reward
        reward = -1  # Small negative reward for each step
        done = False
        
        # Additional penalty for slipping
        if action_slipped:
            reward -= 2  # Extra penalty for slipping (total -3 for this step)
        
        # Check what cell the agent is on
        self.is_on_special_cell = False
        self.special_cell_type = ""
        
        # Check if reached gem
        if self.agent_pos in self.remaining_gems:
            reward = 100
            done = True
            self.collected_gems += 1
            self.remaining_gems.remove(self.agent_pos)
            self.is_on_special_cell = True
            self.special_cell_type = "Gem (Goal)"
        elif self.agent_pos in self.obstacles:
            self.is_on_special_cell = True
            self.special_cell_type = "Obstacle"
        elif self.agent_pos in self.slippery_cells:
            self.is_on_special_cell = True
            if action_slipped:
                self.special_cell_type = "Slippery Cell (SLIPPED!)"
            else:
                self.special_cell_type = "Slippery Cell"
        elif self.agent_pos == (0, 0):
            self.is_on_special_cell = True
            self.special_cell_type = "Start Position"
        
        # Update state tracking
        self.current_reward = reward
        self.total_reward += reward
        self.episode_steps += 1
        self.last_action_slipped = action_slipped
        
        # Update slip animation state
        if action_slipped:
            self.slip_animation_active = True
            self.slip_animation_timer = 0
            self.slip_start_pos = (x, y)  # Position before the slip
            self.slip_end_pos = self.agent_pos  # Position after the slip
            self.slip_direction = action  # Actual direction moved (not intended)
        
        return self.agent_pos, reward, done, {"slipped": action_slipped, "intended_action": original_action, "actual_action": action}

    def render(self, screen):
        """Draw the grid world using pygame with status panel"""
        # Clear the entire screen
        screen.fill((240, 240, 240))
        
        # Draw game area background
        game_rect = pygame.Rect(0, 0, self.game_width, self.window_height)
        pygame.draw.rect(screen, (255, 255, 255), game_rect)
        
        # Draw grid lines
        for i in range(self.size + 1):
            # Vertical lines
            pygame.draw.line(screen, (200, 200, 200), 
                           (i * self.cell_size, 0), 
                           (i * self.cell_size, self.size * self.cell_size))
            # Horizontal lines
            pygame.draw.line(screen, (200, 200, 200), 
                           (0, i * self.cell_size), 
                           (self.size * self.cell_size, i * self.cell_size))
        
        # Draw obstacles
        for x, y in self.obstacles:
            rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                             self.cell_size, self.cell_size)
            pygame.draw.rect(screen, (100, 100, 100), rect)
        
        # Draw slippery cells (icy blue background with diagonal stripes and sparkles)
        for x, y in self.slippery_cells:
            rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                             self.cell_size, self.cell_size)
            
            # Base icy blue background (slightly lighter)
            pygame.draw.rect(screen, (200, 230, 255), rect)  # Very light ice blue
            
            # Add diagonal stripe pattern for texture
            stripe_color = (160, 200, 240)  # Slightly darker blue for stripes
            stripe_width = 3
            stripe_spacing = 8
            
            # Draw diagonal stripes from top-left to bottom-right
            for offset in range(-self.cell_size, self.cell_size, stripe_spacing):
                start_x = x * self.cell_size + offset
                start_y = y * self.cell_size
                end_x = x * self.cell_size + offset + self.cell_size
                end_y = y * self.cell_size + self.cell_size
                
                # Clip the line to the cell boundaries
                if start_x < x * self.cell_size:
                    start_y += (x * self.cell_size - start_x)
                    start_x = x * self.cell_size
                if end_x > (x + 1) * self.cell_size:
                    end_y -= (end_x - (x + 1) * self.cell_size)
                    end_x = (x + 1) * self.cell_size
                
                if start_x <= end_x and start_y <= end_y:
                    pygame.draw.line(screen, stripe_color, (start_x, start_y), (end_x, end_y), stripe_width)
            
            # Add subtle sparkle dots for ice crystal effect
            sparkle_color = (255, 255, 255)  # White sparkles
            sparkle_positions = [
                (x * self.cell_size + 12, y * self.cell_size + 15),
                (x * self.cell_size + 35, y * self.cell_size + 10),
                (x * self.cell_size + 25, y * self.cell_size + 30),
                (x * self.cell_size + 8, y * self.cell_size + 38),
                (x * self.cell_size + 40, y * self.cell_size + 35),
            ]
            
            for sparkle_x, sparkle_y in sparkle_positions:
                # Draw small plus-shaped sparkles
                pygame.draw.line(screen, sparkle_color, 
                               (sparkle_x - 2, sparkle_y), (sparkle_x + 2, sparkle_y), 1)
                pygame.draw.line(screen, sparkle_color, 
                               (sparkle_x, sparkle_y - 2), (sparkle_x, sparkle_y + 2), 1)
            
            # Add a subtle border with rounded appearance
            border_color = (120, 160, 220)  # Darker blue border
            pygame.draw.rect(screen, border_color, rect, 2)
            
            # Add corner highlights for a more polished look
            highlight_color = (240, 248, 255)  # Almost white highlight
            corner_size = 6
            
            # Top-left corner highlight
            pygame.draw.line(screen, highlight_color, 
                           (x * self.cell_size + 2, y * self.cell_size + 2), 
                           (x * self.cell_size + corner_size, y * self.cell_size + 2), 2)
            pygame.draw.line(screen, highlight_color, 
                           (x * self.cell_size + 2, y * self.cell_size + 2), 
                           (x * self.cell_size + 2, y * self.cell_size + corner_size), 2)
        
        # Draw remaining gems
        for x, y in self.remaining_gems:
            center = (x * self.cell_size + self.cell_size // 2,
                     y * self.cell_size + self.cell_size // 2)
            pygame.draw.circle(screen, (255, 215, 0), center, self.cell_size // 3)
        
        # Draw collected gems (faded)
        collected_positions = [pos for pos in self.gems if pos not in self.remaining_gems]
        for x, y in collected_positions:
            center = (x * self.cell_size + self.cell_size // 2,
                     y * self.cell_size + self.cell_size // 2)
            pygame.draw.circle(screen, (200, 180, 100), center, self.cell_size // 4)
        
        # Draw agent with slip animation
        self._draw_agent_with_slip_animation(screen)
        
        # Draw status panel
        self._draw_status_panel(screen)

    def _draw_agent_with_slip_animation(self, screen):
        """Draw the agent with cartoon-style slip animation when slipping"""
        # Update animation timer
        if self.slip_animation_active:
            self.slip_animation_timer += 1
            if self.slip_animation_timer >= self.slip_animation_duration:
                self.slip_animation_active = False
                self.slip_animation_timer = 0
        
        x, y = self.agent_pos
        base_center_x = x * self.cell_size + self.cell_size // 2
        base_center_y = y * self.cell_size + self.cell_size // 2
        
        if self.slip_animation_active:
            # Calculate animation progress (0.0 to 1.0)
            progress = self.slip_animation_timer / self.slip_animation_duration
            
            # Create wobble effect (side to side motion)
            wobble_intensity = 8 * (1 - progress)  # Wobble decreases over time
            wobble_offset_x = math.sin(progress * 12) * wobble_intensity
            wobble_offset_y = math.cos(progress * 8) * wobble_intensity * 0.5
            
            # Apply wobble to agent position
            agent_center_x = base_center_x + wobble_offset_x
            agent_center_y = base_center_y + wobble_offset_y
            
            # Draw motion lines behind the agent to show slip direction
            self._draw_slip_motion_lines(screen, base_center_x, base_center_y, progress)
            
            # Agent color changes during slip (gets redder as if embarrassed/dizzy)
            slip_red = min(255, 100 + int(155 * (1 - progress)))
            agent_color = (slip_red, max(0, 100 - int(50 * (1 - progress))), 255)
            
            # Agent size wobbles slightly
            base_radius = self.cell_size // 4
            radius_wobble = math.sin(progress * 20) * 3
            agent_radius = base_radius + radius_wobble
            
        else:
            # Normal agent drawing
            agent_center_x = base_center_x
            agent_center_y = base_center_y
            agent_color = (0, 100, 255)
            agent_radius = self.cell_size // 4
        
        # Draw the agent circle
        pygame.draw.circle(screen, agent_color, 
                         (int(agent_center_x), int(agent_center_y)), 
                         int(agent_radius))
        
        # Add dizzy stars if slipping (classic cartoon effect)
        if self.slip_animation_active and progress < 0.7:
            self._draw_dizzy_stars(screen, agent_center_x, agent_center_y, progress)

    def _draw_slip_motion_lines(self, screen, center_x, center_y, progress):
        """Draw motion lines to show the slip direction"""
        motion_color = (200, 200, 255, int(100 * (1 - progress)))  # Fading blue
        line_length = 20 - int(15 * progress)  # Lines get shorter over time
        
        # Draw several motion lines in the slip direction
        for i in range(3):
            offset = (i + 1) * 8
            line_thickness = 3 - i
            
            # Direction-based motion lines
            if self.slip_direction == 'right':
                start_x = center_x - offset - line_length
                end_x = center_x - offset
                start_y = end_y = center_y + (i - 1) * 4
            elif self.slip_direction == 'left':
                start_x = center_x + offset
                end_x = center_x + offset + line_length
                start_y = end_y = center_y + (i - 1) * 4
            elif self.slip_direction == 'down':
                start_y = center_y - offset - line_length
                end_y = center_y - offset
                start_x = end_x = center_x + (i - 1) * 4
            elif self.slip_direction == 'up':
                start_y = center_y + offset
                end_y = center_y + offset + line_length
                start_x = end_x = center_x + (i - 1) * 4
            else:
                continue
            
            # Draw motion line with alpha blending effect
            pygame.draw.line(screen, (150, 150, 255), 
                           (int(start_x), int(start_y)), 
                           (int(end_x), int(end_y)), 
                           line_thickness)

    def _draw_dizzy_stars(self, screen, center_x, center_y, progress):
        """Draw classic cartoon dizzy stars around the agent's head"""
        star_color = (255, 255, 100)  # Yellow stars
        num_stars = 3
        orbit_radius = 30 + int(5 * math.sin(progress * 15))  # Pulsing orbit
        
        for i in range(num_stars):
            # Stars orbit around the agent
            angle = (progress * 8 + i * (2 * math.pi / num_stars)) % (2 * math.pi)
            star_x = center_x + orbit_radius * math.cos(angle)
            star_y = center_y - 15 + orbit_radius * math.sin(angle) * 0.5  # Elliptical orbit
            
            # Draw star shape (simple 4-pointed star)
            star_size = 4
            # Horizontal line
            pygame.draw.line(screen, star_color, 
                           (star_x - star_size, star_y), 
                           (star_x + star_size, star_y), 2)
            # Vertical line
            pygame.draw.line(screen, star_color, 
                           (star_x, star_y - star_size), 
                           (star_x, star_y + star_size), 2)
            # Diagonal lines
            pygame.draw.line(screen, star_color, 
                           (star_x - star_size*0.7, star_y - star_size*0.7), 
                           (star_x + star_size*0.7, star_y + star_size*0.7), 1)
            pygame.draw.line(screen, star_color, 
                           (star_x - star_size*0.7, star_y + star_size*0.7), 
                           (star_x + star_size*0.7, star_y - star_size*0.7), 1)

    def _draw_status_panel(self, screen):
        """Draw the status panel with real-time information"""
        # Status panel background
        status_rect = pygame.Rect(self.game_width, 0, self.status_width, self.window_height)
        pygame.draw.rect(screen, (50, 50, 50), status_rect)
        
        # Initialize font
        try:
            font_large = pygame.font.Font(None, 24)
            font_medium = pygame.font.Font(None, 20)
            font_small = pygame.font.Font(None, 16)
        except:
            font_large = pygame.font.SysFont('Arial', 24)
            font_medium = pygame.font.SysFont('Arial', 20)
            font_small = pygame.font.SysFont('Arial', 16)
        
        y_offset = 10
        line_height = 25
        
        # Title
        title = font_large.render("STATUS PANEL", True, (255, 255, 255))
        screen.blit(title, (self.game_width + 10, y_offset))
        y_offset += 35
        
        # Agent Position
        pos_text = font_medium.render(f"Agent Position:", True, (255, 255, 255))
        screen.blit(pos_text, (self.game_width + 10, y_offset))
        y_offset += 20
        pos_coords = font_small.render(f"  X: {self.agent_pos[0]}, Y: {self.agent_pos[1]}", True, (200, 255, 200))
        screen.blit(pos_coords, (self.game_width + 10, y_offset))
        y_offset += 30
        
        # Gems Information
        gems_text = font_medium.render(f"Gems:", True, (255, 255, 255))
        screen.blit(gems_text, (self.game_width + 10, y_offset))
        y_offset += 20
        collected_text = font_small.render(f"  Collected: {self.collected_gems}", True, (255, 215, 0))
        screen.blit(collected_text, (self.game_width + 10, y_offset))
        y_offset += 18
        remaining_text = font_small.render(f"  Remaining: {len(self.remaining_gems)}", True, (255, 255, 255))
        screen.blit(remaining_text, (self.game_width + 10, y_offset))
        y_offset += 18
        total_text = font_small.render(f"  Total: {self.total_gems}", True, (200, 200, 200))
        screen.blit(total_text, (self.game_width + 10, y_offset))
        y_offset += 30
        
        # Rewards
        reward_text = font_medium.render(f"Rewards:", True, (255, 255, 255))
        screen.blit(reward_text, (self.game_width + 10, y_offset))
        y_offset += 20
        current_reward_text = font_small.render(f"  Last Action: {self.current_reward}", True, (255, 200, 200))
        screen.blit(current_reward_text, (self.game_width + 10, y_offset))
        y_offset += 18
        total_reward_text = font_small.render(f"  Total: {self.total_reward}", True, (200, 255, 200))
        screen.blit(total_reward_text, (self.game_width + 10, y_offset))
        y_offset += 30
        
        # Episode Information
        episode_text = font_medium.render(f"Episode Info:", True, (255, 255, 255))
        screen.blit(episode_text, (self.game_width + 10, y_offset))
        y_offset += 20
        steps_text = font_small.render(f"  Steps: {self.episode_steps}", True, (255, 255, 200))
        screen.blit(steps_text, (self.game_width + 10, y_offset))
        y_offset += 18
        
        # Slip information
        if self.slip_animation_active:
            slip_text = font_small.render(f"  SLIPPING! 🌟", True, (255, 255, 100))
        elif self.last_action_slipped:
            slip_text = font_small.render(f"  Last Action: SLIPPED!", True, (255, 100, 100))
        else:
            slip_text = font_small.render(f"  Last Action: Normal", True, (200, 200, 200))
        screen.blit(slip_text, (self.game_width + 10, y_offset))
        y_offset += 30
        
        # Special Cell Information
        special_text = font_medium.render(f"Current Cell:", True, (255, 255, 255))
        screen.blit(special_text, (self.game_width + 10, y_offset))
        y_offset += 20
        if self.is_on_special_cell:
            cell_info = font_small.render(f"  {self.special_cell_type}", True, (255, 255, 100))
        else:
            cell_info = font_small.render(f"  Empty Cell", True, (200, 200, 200))
        screen.blit(cell_info, (self.game_width + 10, y_offset))
        y_offset += 30
        
        # Available Actions
        actions_text = font_medium.render(f"Available Items:", True, (255, 255, 255))
        screen.blit(actions_text, (self.game_width + 10, y_offset))
        y_offset += 20
        
        # Check what can be collected
        can_collect = []
        if self.agent_pos in self.remaining_gems:
            can_collect.append("Gold Gem")
        
        if can_collect:
            for item in can_collect:
                item_text = font_small.render(f"  • {item}", True, (100, 255, 100))
                screen.blit(item_text, (self.game_width + 10, y_offset))
                y_offset += 18
        else:
            no_items = font_small.render(f"  None", True, (200, 200, 200))
            screen.blit(no_items, (self.game_width + 10, y_offset))
            y_offset += 20
        
        # Slippery Cells Information
        y_offset += 10
        slippery_title = font_medium.render(f"Slippery Cells:", True, (255, 255, 255))
        screen.blit(slippery_title, (self.game_width + 10, y_offset))
        y_offset += 20
        slippery_count = font_small.render(f"  Total: {len(self.slippery_cells)}", True, (173, 216, 230))
        screen.blit(slippery_count, (self.game_width + 10, y_offset))
        y_offset += 18
        slip_prob = font_small.render(f"  Slip Chance: {int(self.slip_probability * 100)}%", True, (200, 200, 255))
        screen.blit(slip_prob, (self.game_width + 10, y_offset))
        y_offset += 30
        
        # Controls
        y_offset += 20
        controls_text = font_medium.render(f"Controls:", True, (255, 255, 255))
        screen.blit(controls_text, (self.game_width + 10, y_offset))
        y_offset += 20
        arrow_text = font_small.render(f"  Arrows: Move", True, (200, 200, 200))
        screen.blit(arrow_text, (self.game_width + 10, y_offset))
        y_offset += 18
        reset_text = font_small.render(f"  R: Reset", True, (200, 200, 200))
        screen.blit(reset_text, (self.game_width + 10, y_offset))

def value_iteration(env, gamma=0.9, theta=1e-4):
    V = {state: 0 for state in env.states}
    while True:
        delta = 0
        for state in env.states:
            v = V[state]
            V[state] = max(
                sum(
                    prob * (reward + gamma * V[next_state])
                    for prob, next_state, reward, done in env.P[state][action]
                )
                for action in env.actions
            )
            delta = max(delta, abs(v - V[state]))
        if delta < theta:
            break
    return V

def sarsa(env, alpha=0.1, gamma=0.9, epsilon=0.1, episodes=1000):
    Q = {(state, action): 0 for state in env.states for action in env.actions}
    for _ in range(episodes):
        state = env.reset()
        action = epsilon_greedy(Q, state, epsilon, env.actions)
        while True:
            next_state, reward, done = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon, env.actions)
            Q[(state, action)] += alpha * (
                reward + gamma * Q[(next_state, next_action)] - Q[(state, action)]
            )
            state = next_state
            action = next_action
            if done:
                break
    return Q

def q_learning(env, alpha=0.1, gamma=0.9, epsilon=0.1, episodes=1000):
    Q = {(state, action): 0 for state in env.states for action in env.actions}
    for _ in range(episodes):
        state = env.reset()
        while True:
            action = epsilon_greedy(Q, state, epsilon, env.actions)
            next_state, reward, done = env.step(action)
            next_max = max(Q[(next_state, a)] for a in env.actions)
            Q[(state, action)] += alpha * (
                reward + gamma * next_max - Q[(state, action)]
            )
            state = next_state
            if done:
                break
    return Q

def epsilon_greedy(Q, state, epsilon, actions):
    if random.random() < epsilon:
        return random.choice(actions)  # Explore: random action
    else:
        return max(actions, key=lambda a: Q[(state, a)])  # Exploit: best action

def dqn(env, gamma=0.9, epsilon=0.1, episodes=1000):
    # Placeholder for DQN implementation
    # Typically, this would use a neural network to approximate Q-values.
    # For now, raise NotImplementedError to indicate it's not implemented.
    raise NotImplementedError("DQN algorithm is not implemented yet.")

class Environment:
    def __init__(self, size=10):
        self.size = size
        self.grid = [[0 for _ in range(size)] for _ in range(size)]
        self.states = [(x, y) for x in range(size) for y in range(size)]
        self.actions = ['up', 'down', 'left', 'right']
        # Define transitions, rewards, and slippery cells here

    def reset(self):
        # Reset the environment to the initial state
        pass

    def step(self, action):
        # Execute the action and return the next state, reward, and done flag
        pass

def draw_grid(screen, grid):
    for x in range(len(grid)):
        for y in range(len(grid[0])):
            pygame.draw.rect(screen, (255, 255, 255), (x * 50, y * 50, 50, 50), 1)

# --- Tkinter Controls ---
class TrainingControls:
    def __init__(self, master, env, agent):
        self.env = env
        self.agent = agent
        self.window = master
        # Add sliders, entries, and buttons for RL parameters
        # Example:
        self.lr_var = tk.DoubleVar(value=0.1)
        tk.Label(master, text="Learning Rate (α):").pack()
        tk.Entry(master, textvariable=self.lr_var).pack()
        # Add more controls for epsilon, gamma, etc.
        # Add buttons for Train, Stop, Manual Play, etc.

    def train_agent(self):
        # Start training directly (no threading needed)
        self.agent.train(self.lr_var.get())  # Direct call

# --- Pygame Visualization ---
def run_pygame(env):
    pygame.init()
    screen = pygame.display.set_mode((500, 500))
    pygame.display.set_caption("Escape Room RL")
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill((0, 0, 0))
        draw_grid(screen, env.grid)
        pygame.display.flip()
    pygame.quit()

class QLearningAgent:
    def __init__(self, env):
        self.env = env
        # Add more initialization as needed

    def train(self, learning_rate, *args, **kwargs):
        # Implement Q-learning training logic here
        print(f"Training with learning rate: {learning_rate}")
        # You can call your q_learning function here

# --- Main ---
if __name__ == "__main__":
    env = Environment()
    agent = QLearningAgent(env)
    root = tk.Tk()
    controls = TrainingControls(root, env, agent)
    # Start Pygame directly (no threading needed)
    # run_pygame(env)  # Direct call if needed
    root.mainloop()
