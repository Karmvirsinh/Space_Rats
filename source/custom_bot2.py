import numpy as np
import random
import matplotlib.pyplot as plt
from utils import sense_blocked, move, ping_detector, manhattan_distance, bfs_path
from ship_generator import generate_ship

def custom_bot2(ship, alpha=0.15, visualize=False):
    D = ship.shape[0]
    moves, senses, pings = 0, 0, 0

    # Initialize bot and rat positions
    open_cells = [(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1]
    true_bot_pos = random.choice(open_cells)
    bot_pos = true_bot_pos
    bot_knowledge = set(open_cells)
    steps = [(bot_pos, moves, senses, pings, len(bot_knowledge))]

    # Phase 1: Localization (Adapted from Friend's BaselineSpaceRatsBot)
    phase1_sense_next = True
    step = 0
    max_steps = 150
    print(f"True Bot Spawn: {true_bot_pos}")
    print(f"Step {step}: Bot at {bot_pos}, Knowledge size: {len(bot_knowledge)}")

    if visualize:
        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 6))

    while len(bot_knowledge) > 1 and step < max_steps:
        if phase1_sense_next:
            sensor_val = sense_blocked(ship, bot_pos)
            senses += 1
            bot_knowledge = {cell for cell in bot_knowledge if sense_blocked(ship, cell) == sensor_val}
            phase1_sense_next = False
            print(f"Step {step+1}: Sensed {sensor_val} blocked neighbors, Knowledge size: {len(bot_knowledge)}")
        else:
            directions = {'up': 0, 'down': 0, 'left': 0, 'right': 0}
            for cell in bot_knowledge:
                for d in directions:
                    new_pos = move(ship, cell, d)
                    if new_pos != cell:
                        directions[d] += 1
            direction = max(directions, key=directions.get)
            new_pos = move(ship, bot_pos, direction)
            move_success = new_pos != bot_pos
            moves += 1
            if move_success:
                bot_knowledge = {move(ship, pos, direction) for pos in bot_knowledge}
                print(f"Step {step+1}: Moved {direction} to {new_pos}, Knowledge size: {len(bot_knowledge)}")
            else:
                bot_knowledge = {pos for pos in bot_knowledge if move(ship, pos, direction) == pos}
                print(f"Step {step+1}: Tried {direction} (blocked), Bot at {bot_pos}, Knowledge size: {len(bot_knowledge)}")
            bot_pos = new_pos
            phase1_sense_next = True

        steps.append((bot_pos, moves, senses, pings, len(bot_knowledge)))
        step += 1

        # Visualization for Phase 1
        if visualize:
            ax.clear()
            grid = np.array(ship)
            ax.imshow(grid, cmap='gray_r')
            bot_y, bot_x = bot_pos
            ax.plot(bot_x, bot_y, 'o', markersize=12, color='blue', label="Bot")
            if bot_knowledge:
                xs = [j for (i, j) in bot_knowledge]
                ys = [i for (i, j) in bot_knowledge]
                ax.plot(xs, ys, 'o', markersize=4, color='green', label="KB Candidates")
            ax.set_title(f"Phase 1: Localization\nStep {step}: Moves: {moves}, Senses: {senses}, KB Size: {len(bot_knowledge)}")
            ax.legend(loc='upper right')
            plt.pause(0.3)

    if visualize:
        plt.ioff()
        plt.show()

    if len(bot_knowledge) == 1:
        final_pos = bot_knowledge.pop()
        print(f"Step {step+1}: Localized, Bot at {final_pos}")
        steps.append((final_pos, moves, senses, pings, 1))
    else:
        final_pos = bot_pos
        print(f"Max steps reached, stuck at {len(bot_knowledge)} positions")
    print(f"Phase 1 done, Bot at {final_pos}, True Spawn was {true_bot_pos}")

    # Phase 2: Placeholder
    rat_knowledge = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1 and (r, c) != final_pos}
    if not rat_knowledge:
        print("No valid positions for rat spawn, ending simulation.")
        return moves, senses, pings, steps, None
    true_rat_pos = random.choice(list(rat_knowledge))
    bot_pos = final_pos
    print(f"True Rat Spawn: {true_rat_pos}")
    return moves, senses, pings, steps, true_rat_pos

if __name__ == "__main__":
    ship = generate_ship(30)
    moves, senses, pings, steps, true_rat_pos = custom_bot2(ship, visualize=True)
    print(f"Final: Moves: {moves}, Senses: {senses}, Pings: {pings}")