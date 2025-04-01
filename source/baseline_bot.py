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

def baseline_bot(ship, alpha=0.15):
    D = ship.shape[0]
    moves, senses, pings = 0, 0, 0
    blocked_map = precompute_blocked(ship)

    # Phase 1: Localize bot (improved with proper oscillation handling)
    bot_knowledge = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1}
    true_bot_pos = random.choice(list(bot_knowledge))
    bot_pos = true_bot_pos
    steps = [(bot_pos, moves, senses, pings, len(bot_knowledge))]
    print(f"True Bot Spawn: {true_bot_pos}")
    print(f"Step 0: Bot at {bot_pos}, Knowledge size: {len(bot_knowledge)}")

    prev_size = len(bot_knowledge)
    step = 1
    visited = {bot_pos: 1}
    max_steps = 100

    while len(bot_knowledge) > 1 and step < max_steps:
        blocked_count = blocked_map[bot_pos[0], bot_pos[1]]
        senses += 1
        bot_knowledge = {pos for pos in bot_knowledge if blocked_map[pos[0], pos[1]] == blocked_count}
        print(f"Step {step}: Sensed {blocked_count} blocked neighbors, Knowledge size: {len(bot_knowledge)}")
        steps.append((bot_pos, moves, senses, pings, len(bot_knowledge)))
        step += 1

        if len(bot_knowledge) == 1:
            break
        if len(bot_knowledge) == prev_size and visited.get(bot_pos, 0) > 2:
            print(f"Stuck at {len(bot_knowledge)} positions: {bot_knowledge}")
            # Move randomly to break oscillation (professor's advice)
            direction = random.choice(['up', 'down', 'left', 'right'])
            new_pos = move(ship, bot_pos, direction)
            moves += 1
            if new_pos != bot_pos:
                bot_knowledge = {move(ship, pos, direction) for pos in bot_knowledge}
                print(f"Step {step}: Moved {direction} (random) to {new_pos}, Knowledge size: {len(bot_knowledge)}")
            else:
                bot_knowledge = {pos for pos in bot_knowledge if move(ship, pos, direction) == pos}
                print(f"Step {step}: Tried {direction} (random, blocked), Bot at {bot_pos}, Knowledge size: {len(bot_knowledge)}")
            bot_pos = new_pos
            visited[bot_pos] = visited.get(bot_pos, 0) + 1
            steps.append((bot_pos, moves, senses, pings, len(bot_knowledge)))
            step += 1
            prev_size = len(bot_knowledge)
            continue

        direction_counts = {'up': 0, 'down': 0, 'left': 0, 'right': 0}
        for pos in bot_knowledge:
            for direction in direction_counts:
                if move(ship, pos, direction) != pos:
                    direction_counts[direction] += 1
        direction = max(direction_counts, key=direction_counts.get)

        new_pos = move(ship, bot_pos, direction)
        moves += 1
        if new_pos != bot_pos:
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

    # Phase 2: Find the stationary space rat (unchanged)
    rat_knowledge = {(r, c) for r in range(D) for c in range(D) if ship[r, c] == 1 and (r, c) != final_pos}
    true_rat_pos = random.choice(list(rat_knowledge))
    bot_pos = final_pos
    print(f"True Rat Spawn: {true_rat_pos}")

    # Initialize probabilities
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
    from ship_generator import generate_ship
    ship = generate_ship(30)
    moves, senses, pings, steps, true_rat_pos = baseline_bot(ship)
    print(f"Final: Moves: {moves}, Senses: {senses}, Pings: {pings}")