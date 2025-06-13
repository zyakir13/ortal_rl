import pygame
from grid import GridWorld 
from rl_algorithms import value_iteration
import tkinter as tk
from tkinter import ttk
import sys

class GameWindow:
    def __init__(self, env):
        pygame.init()
        self.env = env
        self.screen = pygame.display.set_mode((600, 600))
        pygame.display.set_caption("RL Environment")
        self.running = True
        self.run()

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    pygame.quit()
                    sys.exit()
            
            # Clear screen
            self.screen.fill((255, 255, 255))
            
            # Render environment
            self.env.render(self.screen)
            
            # Update display
            pygame.display.flip()
            
            # Cap the frame rate
            pygame.time.Clock().tick(60)

class ControlPanel(tk.Tk):
    def __init__(self, env, next_room_callback):
        super().__init__()
        
        self.env = env
        self.next_room_callback = next_room_callback
        
        self.title("RL Control Panel")
        self.geometry("400x600")
        
        # Create main frame
        main_frame = ttk.Frame(self, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Learning Parameters Section
        learning_frame = ttk.LabelFrame(main_frame, text="Learning Parameters", padding="5")
        learning_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Learning Rate
        ttk.Label(learning_frame, text="Learning Rate (α):").grid(row=0, column=0, sticky=tk.W)
        self.learning_rate = ttk.Entry(learning_frame, width=10)
        self.learning_rate.insert(0, "0.1")
        self.learning_rate.grid(row=0, column=1, padx=5)
        
        # Discount Factor
        ttk.Label(learning_frame, text="Discount Factor (γ):").grid(row=1, column=0, sticky=tk.W)
        self.discount_factor = ttk.Entry(learning_frame, width=10)
        self.discount_factor.insert(0, "0.9")
        self.discount_factor.grid(row=1, column=1, padx=5)
        
        # Max Steps
        ttk.Label(learning_frame, text="Max Steps:").grid(row=2, column=0, sticky=tk.W)
        self.max_steps = ttk.Entry(learning_frame, width=10)
        self.max_steps.insert(0, "1000")
        self.max_steps.grid(row=2, column=1, padx=5)
        
        # Number of Episodes
        ttk.Label(learning_frame, text="Number of Episodes:").grid(row=3, column=0, sticky=tk.W)
        self.num_episodes = ttk.Entry(learning_frame, width=10)
        self.num_episodes.insert(0, "100")
        self.num_episodes.grid(row=3, column=1, padx=5)
        
        # Exploration Parameters Section
        exploration_frame = ttk.LabelFrame(main_frame, text="Exploration Parameters", padding="5")
        exploration_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Initial Epsilon
        ttk.Label(exploration_frame, text="Initial Epsilon:").grid(row=0, column=0, sticky=tk.W)
        self.initial_epsilon = ttk.Entry(exploration_frame, width=10)
        self.initial_epsilon.insert(0, "1.0")
        self.initial_epsilon.grid(row=0, column=1, padx=5)
        
        # Minimum Epsilon
        ttk.Label(exploration_frame, text="Minimum Epsilon:").grid(row=1, column=0, sticky=tk.W)
        self.min_epsilon = ttk.Entry(exploration_frame, width=10)
        self.min_epsilon.insert(0, "0.01")
        self.min_epsilon.grid(row=1, column=1, padx=5)
        
        # Epsilon Decay Rate
        ttk.Label(exploration_frame, text="Epsilon Decay Rate:").grid(row=2, column=0, sticky=tk.W)
        self.epsilon_decay = ttk.Entry(exploration_frame, width=10)
        self.epsilon_decay.insert(0, "0.995")
        self.epsilon_decay.grid(row=2, column=1, padx=5)
        
        # Current Epsilon Display
        ttk.Label(exploration_frame, text="Current Epsilon:").grid(row=3, column=0, sticky=tk.W)
        self.current_epsilon = ttk.Label(exploration_frame, text="1.0")
        self.current_epsilon.grid(row=3, column=1, padx=5)
        
        # Action Buttons Section
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Button(button_frame, text="Train Agent", command=self.train).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Stop Training", command=self.stop).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Watch AI Play", command=self.play).grid(row=1, column=0, padx=5)
        ttk.Button(button_frame, text="Manual Play", command=self.manual).grid(row=1, column=1, padx=5)
        ttk.Button(button_frame, text="Stop Episode", command=self.stop).grid(row=2, column=0, columnspan=2, pady=5)
        ttk.Button(button_frame, text="Next Room", command=self.next_room).grid(row=3, column=0, columnspan=2, pady=5)
        
        # Status/Output Area
        status_frame = ttk.LabelFrame(main_frame, text="Status/Output", padding="5")
        status_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.status_text = tk.Text(status_frame, height=10, width=40)
        self.status_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbar for status text
        scrollbar = ttk.Scrollbar(status_frame, orient=tk.VERTICAL, command=self.status_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.status_text['yscrollcommand'] = scrollbar.set
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(0, weight=1)
        
        # No threading - direct calls
        self.is_training = False
    
    def update_status(self, message):
        """Update status text directly (no threading)"""
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)
        self.update()  # Force UI update
    
    def update_epsilon(self, value):
        """Update epsilon display"""
        self.current_epsilon.config(text=f"{value:.3f}")
        self.update()
    
    def train(self):
        """Train directly without threading"""
        if not self.is_training:
            self.is_training = True
            try:
                # Get parameters from UI
                learning_rate = float(self.learning_rate.get())
                discount_factor = float(self.discount_factor.get())
                max_steps = int(self.max_steps.get())
                num_episodes = int(self.num_episodes.get())
                initial_epsilon = float(self.initial_epsilon.get())
                min_epsilon = float(self.min_epsilon.get())
                epsilon_decay = float(self.epsilon_decay.get())
                
                params = {
                    'learning_rate': learning_rate,
                    'discount_factor': discount_factor,
                    'max_steps': max_steps,
                    'num_episodes': num_episodes,
                    'initial_epsilon': initial_epsilon,
                    'min_epsilon': min_epsilon,
                    'epsilon_decay': epsilon_decay
                }
                
                self.update_status("Starting training...")
                self.update_status(f"Parameters: LR={learning_rate}, γ={discount_factor}, Episodes={num_episodes}")
                
                # Call the training method directly
                if hasattr(self.env, 'train'):
                    self.env.train(**params)
                    self.update_status("Training completed successfully!")
                else:
                    self.update_status("Error: Environment does not have a train method!")
                    
            except ValueError as e:
                self.update_status(f"Parameter error: {str(e)}")
            except Exception as e:
                self.update_status(f"Training error: {str(e)}")
                import traceback
                self.update_status(f"Traceback: {traceback.format_exc()}")
            finally:
                self.is_training = False
    
    def play(self):
        """Start AI play directly"""
        self.update_status("Starting AI play...")
        self.env.play()
    
    def manual(self):
        """Switch to manual control directly"""
        self.update_status("Switching to manual control...")
        self.env.manual()
    
    def stop(self):
        """Stop current operation directly"""
        self.update_status("Stopping current operation...")
        self.env.stop()
        self.is_training = False
    
    def next_room(self):
        """Move to next room"""
        print("🚪 Next Room button clicked!")
        self.update_status("Moving to next room...")
        
        try:
            # Call the callback to notify the room
            self.next_room_callback()
            print("🚪 next_room_callback completed")
            
            # Close the panel
            self.quit()  # This will exit mainloop
            self.destroy()
            print("🚪 Panel destroyed")
            
        except Exception as e:
            print(f"❌ Error in next_room: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
