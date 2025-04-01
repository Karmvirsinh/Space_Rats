import numpy as np
import random
from utils import sense_blocked, move, ping_detector, manhattan_distance, bfs_path

def precompute_blocked(ship):
    D = ship.shape[0]
    blocked_map = np.zeros((D, D), dtype=int)
    for r in range(D):
        for c in range(D):
            if ship[r, c] == 1:
                blocked_map[r, c] = sense_blocked(ship, (r, c))
    return blocked_map

def custom_bot(ship, alpha=0.15):
    D = ship.shape[0]
    moves, senses, pings = 0, 0, 0
    blocked_map = precompute_blocked(ship)

    # Phase 1: Localize bot (with history tracking for KB)
    bot_knowledge = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1}
    true_bot_pos = random.choice(list(bot_knowledge))
    bot_pos = true_bot_pos
    steps = [(bot_pos, moves, senses, pings, len(bot_knowledge))]
    print(f"True Bot Spawn: {true_bot_pos}")
    print(f"Step 0: Bot at {bot_pos}, Knowledge size: {len(bot_knowledge)}")

    prev_size = len(bot_knowledge)
    step = 1
    visited = {bot_pos: 1}
    max_steps = 150
    directions_list = ['up', 'down', 'left', 'right']
    stagnant_steps = 0
    # Track history of senses and moves
    sense_history = []  # List of sensed blocked counts
    move_history = []   # List of (direction, success) tuples

    while len(bot_knowledge) > 1 and step < max_steps:
        # Sense blocked neighbors and update KB
        blocked_count = blocked_map[bot_pos[0], bot_pos[1]]
        senses += 1
        sense_history.append(blocked_count)
        bot_knowledge = {pos for pos in bot_knowledge if blocked_map[pos[0], pos[1]] == blocked_count}
        print(f"Step {step}: Sensed {blocked_count} blocked neighbors, Knowledge size: {len(bot_knowledge)}")
        steps.append((bot_pos, moves, senses, pings, len(bot_knowledge)))
        step += 1

        if len(bot_knowledge) == 1:
            break

        # Check for oscillation
        if len(bot_knowledge) == prev_size:
            stagnant_steps += 1
        else:
            stagnant_steps = 0

        # Trigger smarter random move if stuck
        if stagnant_steps >= 3 and visited.get(bot_pos, 0) > 1:
            print(f"Stuck at {len(bot_knowledge)} positions: {bot_knowledge}")
            # Choose direction that maximizes variance in blocked counts
            direction_scores = {'up': 0, 'down': 0, 'left': 0, 'right': 0}
            for direction in direction_scores:
                next_positions = {move(ship, pos, direction) for pos in bot_knowledge}
                if len(next_positions) > 0:
                    blocked_counts = {}
                    for pos in next_positions:
                        count = blocked_map[pos[0], pos[1]]
                        blocked_counts[count] = blocked_counts.get(count, 0) + 1
                    if blocked_counts:
                        variance = np.var(list(blocked_counts.values()))
                        direction_scores[direction] = variance
            direction = max(direction_scores, key=direction_scores.get)
            new_pos = move(ship, bot_pos, direction)
            moves += 1
            move_success = new_pos != bot_pos
            move_history.append((direction, move_success))
            if move_success:
                bot_knowledge = {move(ship, pos, direction) for pos in bot_knowledge}
                print(f"Step {step}: Moved {direction} (smart random) to {new_pos}, Knowledge size: {len(bot_knowledge)}")
            else:
                bot_knowledge = {pos for pos in bot_knowledge if move(ship, pos, direction) == pos}
                print(f"Step {step}: Tried {direction} (smart random, blocked), Bot at {bot_pos}, Knowledge size: {len(bot_knowledge)}")
            bot_pos = new_pos
            visited[bot_pos] = visited.get(bot_pos, 0) + 1
            steps.append((bot_pos, moves, senses, pings, len(bot_knowledge)))
            step += 1
            prev_size = len(bot_knowledge)
            stagnant_steps = 0
            continue

        # Direction selection: most commonly open
        direction_counts = {'up': 0, 'down': 0, 'left': 0, 'right': 0}
        for pos in bot_knowledge:
            for direction in direction_counts:
                if move(ship, pos, direction) != pos:
                    direction_counts[direction] += 1
        direction = max(direction_counts, key=direction_counts.get)

        # Attempt to move and update KB
        new_pos = move(ship, bot_pos, direction)
        moves += 1
        move_success = new_pos != bot_pos
        move_history.append((direction, move_success))
        if move_success:
            bot_knowledge = {move(ship, pos, direction) for pos in bot_knowledge}
            print(f"Step {step}: Moved {direction} to {new_pos}, Knowledge size: {len(bot_knowledge)}")
        else:
            bot_knowledge = {pos for pos in bot_knowledge if move(ship, pos, direction) == pos}
            print(f"Step {step}: Tried {direction} (blocked), Bot at {bot_pos}, Knowledge size: {len(bot_knowledge)}")
        bot_pos = new_pos
        visited[bot_pos] = visited.get(bot_pos, 0) + 1
        steps.append((bot_pos, moves, senses, pings, len(bot_knowledge)))
        prev_size = len(bot_knowledge)
        step += 1

    else:
        print(f"Max steps reached, stuck at {len(bot_knowledge)} positions: {bot_knowledge}")
        final_pos = bot_pos

    if len(bot_knowledge) == 1:
        final_pos = bot_knowledge.pop()
        print(f"Step {step}: Localized, Bot at {final_pos}")
        steps.append((final_pos, moves, senses, pings, 1))
    else:
        final_pos = bot_pos
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
    from ship_generator import generate_ship
    ship = generate_ship(30)
    moves, senses, pings, steps, true_rat_pos = custom_bot(ship)
    print(f"Final: Moves: {moves}, Senses: {senses}, Pings: {pings}")