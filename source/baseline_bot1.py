import numpy as np
import random
from utils import sense_blocked, move, ping_detector, manhattan_distance, bfs_path
from ship_generator import generate_ship

def precompute_blocked(ship):
    D = ship.shape[0]
    blocked_map = np.zeros((D, D), dtype=int)
    for r in range(D):
        for c in range(D):
            if ship[r, c] == 1:
                blocked_map[r, c] = sense_blocked(ship, (r, c))
    return blocked_map

def get_surroundings(ship, pos):
    D = ship.shape[0]
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    pattern = []
    for dx, dy in directions:
        r, c = pos[0] + dx, pos[1] + dy
        if 0 <= r < D and 0 <= c < D:
            pattern.append(ship[r, c])
        else:
            pattern.append(0)
    return tuple(pattern)

def add_move(pos, direction):
    deltas = {'up': (-1,0), 'down': (1,0), 'left': (0,-1), 'right': (0,1)}
    dx, dy = deltas[direction]
    return (pos[0] + dx, pos[1] + dy)

def reverse_move(direction):
    rev = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}
    return rev[direction]

def baseline_bot(ship, alpha=0.15):
    D = ship.shape[0]
    moves, senses, pings = 0, 0, 0
    blocked_map = precompute_blocked(ship)

    # --- Phase 1: Localization (new baseline_bot, old logging style) ---
    bot_knowledge = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1}
    true_bot_pos = random.choice(list(bot_knowledge))
    bot_pos = true_bot_pos
    steps = [(bot_pos, moves, senses, pings, len(bot_knowledge))]
    print(f"True Bot Spawn: {true_bot_pos}")
    print(f"Step 0: Bot at {bot_pos}, Knowledge size: {len(bot_knowledge)}")

    move_history = []
    current_sensor = get_surroundings(ship, bot_pos)
    senses += 1
    blocked_count = sum(1 for x in current_sensor if x == 0)
    bot_knowledge = {pos for pos in bot_knowledge if get_surroundings(ship, pos) == current_sensor}
    print(f"Step 1: Sensed {blocked_count} blocked neighbors, Knowledge size: {len(bot_knowledge)}")
    steps.append((bot_pos, moves, senses, pings, len(bot_knowledge)))
    step = 2
    max_steps = 100

    while len(bot_knowledge) > 1 and step < max_steps:
        valid_moves = [m for m in ['up', 'down', 'left', 'right'] if move(ship, bot_pos, m) != bot_pos]
        if not valid_moves:
            print(f"Step {step}: No valid moves available, stopping")
            break
        chosen_move = random.choice(valid_moves)
        move_history.append(chosen_move)

        new_pos = move(ship, bot_pos, chosen_move)
        moves += 1
        bot_pos = new_pos
        current_sensor = get_surroundings(ship, bot_pos)
        senses += 1
        blocked_count = sum(1 for x in current_sensor if x == 0)
        print(f"Step {step}: Moved {chosen_move} to {bot_pos}, Sensed {blocked_count} blocked neighbors")

        new_knowledge = set()
        for pos in bot_knowledge:
            new_pos_candidate = move(ship, pos, chosen_move)
            if get_surroundings(ship, new_pos_candidate) == current_sensor:
                new_knowledge.add(new_pos_candidate)
        bot_knowledge = new_knowledge
        print(f"Step {step}: Knowledge size: {len(bot_knowledge)}")
        steps.append((bot_pos, moves, senses, pings, len(bot_knowledge)))
        step += 1

    if len(bot_knowledge) == 1:
        unique_candidate = bot_knowledge.pop()
        estimated_spawn = unique_candidate
        for m in reversed(move_history):
            estimated_spawn = add_move(estimated_spawn, reverse_move(m))
        print(f"Step {step}: Localized, Bot at {estimated_spawn}")
        final_pos = estimated_spawn
        steps.append((final_pos, moves, senses, pings, 1))
    else:
        final_pos = bot_pos
        print(f"Max steps reached, stuck at {len(bot_knowledge)} positions: {bot_knowledge}")
    print(f"Phase 1 done, Bot at {final_pos}, True Spawn was {true_bot_pos}")

    # --- Phase 2: Rat Tracking (old baseline_bot) ---
    rat_knowledge = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1 and (r, c) != final_pos}
    true_rat_pos = random.choice(list(rat_knowledge))
    bot_pos = final_pos
    print(f"True Rat Spawn: {true_rat_pos}")

    rat_probs = {pos: 1.0 / len(rat_knowledge) for pos in rat_knowledge}
    steps.append((bot_pos, moves, senses, pings, len(rat_knowledge)))
    target_path = []
    max_steps = 200

    while bot_pos != true_rat_pos and step < max_steps:
        if not target_path:
            if not rat_knowledge:
                print("Rat knowledge base is empty, cannot find rat.")
                return moves, senses, pings, steps, true_rat_pos
            target_pos = max(rat_probs, key=rat_probs.get)
            target_path = bfs_path(ship, bot_pos, target_pos)
            if not target_path:
                print(f"No path to {target_pos}, removing from KB")
                del rat_probs[target_pos]
                rat_knowledge.remove(target_pos)
                continue

        direction = target_path.pop(0)
        new_pos = move(ship, bot_pos, direction)
        moves += 1
        if new_pos != bot_pos:
            print(f"Step {step}: Moved {direction} to {new_pos}, Rat KB size: {len(rat_knowledge)}")
        bot_pos = new_pos
        steps.append((bot_pos, moves, senses, pings, len(rat_knowledge)))

        pings += 1
        ping = ping_detector(bot_pos, true_rat_pos, alpha)
        dist_to_rat = manhattan_distance(bot_pos, true_rat_pos)
        ping_prob_true = 1.0 if dist_to_rat == 0 else np.exp(-alpha * (dist_to_rat - 1))
        print(f"Step {step}: Pinged at {bot_pos}, Heard ping: {ping}, Ping prob: {ping_prob_true:.3f}")

        if ping and dist_to_rat == 0:
            print(f"Rat found at {bot_pos}")
            return moves, senses, pings, steps, true_rat_pos

        if bot_pos in rat_probs:
            rat_probs[bot_pos] = 0
        total_prob = 0.0
        for pos in rat_knowledge:
            if pos == bot_pos:
                continue
            dist = manhattan_distance(bot_pos, pos)
            ping_prob = 1.0 if dist == 0 else np.exp(-alpha * (dist - 1))
            if ping:
                rat_probs[pos] *= ping_prob
            else:
                rat_probs[pos] *= (1.0 - ping_prob)
            total_prob += rat_probs[pos]

        if total_prob > 0:
            for pos in rat_knowledge:
                if pos != bot_pos:
                    rat_probs[pos] /= total_prob

        to_remove = [pos for pos in rat_knowledge if rat_probs[pos] < 0.0001]
        for pos in to_remove:
            del rat_probs[pos]
            rat_knowledge.remove(pos)

        print(f"Step {step}: Rat KB size after pruning: {len(rat_knowledge)}")
        steps.append((bot_pos, moves, senses, pings, len(rat_knowledge)))
        step += 1

    print(f"Phase 2 done, Bot at {bot_pos}, Rat at {true_rat_pos}, Found: {bot_pos == true_rat_pos}")
    return moves, senses, pings, steps, true_rat_pos

if __name__ == "__main__":
    ship = generate_ship(30)
    moves, senses, pings, steps, true_rat_pos = baseline_bot(ship)
    print(f"Final: Moves: {moves}, Senses: {senses}, Pings: {pings}")