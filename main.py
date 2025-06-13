from Room1 import Room1
from Room2 import Room2
from Room3 import Room3
from Room4 import Room4

def main():
    # List of room classes
    rooms = [Room1, Room2, Room3, Room4]
    current_room_idx = 0

    def start_room(idx):
        if idx < len(rooms):
            env = rooms[idx]()  # Instantiate room environment
            result = env.run()  # Run the room
            if result == "next":
                start_room(idx + 1)
            # If result is "quit" or anything else, exit

    start_room(current_room_idx)

if __name__ == "__main__":
    main()
