# Reinforcement Learning Escape Room Project

This project implements a reinforcement learning environment where an agent learns to navigate through different rooms using various RL algorithms.

## Project Structure

### Core Files
- `main.py`: The entry point of the application. Manages the flow between different rooms and initializes the UI components.
- `grid.py`: Contains the core grid world implementation and basic environment setup. Includes the `GridWorld` class that defines the basic structure of the environment.
- `rl_algorithms.py`: Implements various reinforcement learning algorithms:
  - Value Iteration
  - SARSA
  - Q-Learning
  - DQN (Deep Q-Network)

### Room Implementations
- `Room1.py`: Implementation of the first room environment (Value Iteration)
- `Room2.py`: Implementation of the second room environment (SARSA)
- `Room3.py`: Implementation of the third room environment (Q-Learning)
- `Room4.py`: Implementation of the fourth room environment (DQN)

### UI Components
- `ui.py`: Contains the user interface components:
  - `ControlPanel`: Tkinter-based control panel for managing the RL agent
  - `GameWindow`: Pygame-based visualization of the environment

## Features
- Multiple room environments for the agent to learn
- Various RL algorithms implementation
- Interactive UI with both control panel and visualization
- Support for different learning parameters
- Real-time visualization of agent's learning process

## Requirements
- Python 3.9
- Pygame
- Tkinter
- NumPy (compatible version)

## Installation

1. Clone or download the project
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Troubleshooting NumPy DLL Issues (Windows)

If you encounter NumPy DLL import errors on Windows, try:

```bash
pip uninstall numpy -y
pip install numpy==1.21.6
```

This installs a compatible NumPy version for Python 3.9 on Windows.

## Usage
1. Run `main.py` to start the application:
   ```bash
   python main.py
   ```
2. Use the control panel to:
   - Select the RL algorithm
   - Adjust learning parameters
   - Start/stop training
   - Switch between rooms
3. Watch the agent learn in real-time through the game window

## Project Goals
- Demonstrate different RL algorithms in action
- Provide an interactive learning environment
- Allow comparison between different RL approaches
- Create an engaging way to learn about reinforcement learning

## Current Status
- ✅ Project structure implemented
- ✅ All room classes created
- ✅ Dependencies resolved
- ⚠️ RL algorithm implementations need completion
- ⚠️ UI components need full implementation
- ⚠️ Game visualization needs implementation

## Future Improvements
- Complete RL algorithm implementations
- Implement proper game mechanics and visualization
- Add more complex room environments
- Improve UI with better controls and feedback
- Add performance metrics and logging
- Implement save/load functionality for trained agents 