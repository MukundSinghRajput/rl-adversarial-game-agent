import os
import pickle
import random
import threading
import webbrowser
from typing import Optional

from flask import Flask, jsonify, render_template, request

LEARNING_RATE = 0.3
DISCOUNT      = 0.9
EPISODES      = 200_000
Q_TABLE_PATH  = "q_table.pkl"
PORT          = 5000

WINNING_COMBOS = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]


def check_winner(board: list) -> int:
    """Return 1 (X wins), 2 (O wins), -1 (draw), or 0 (ongoing)."""
    for a, b, c in WINNING_COMBOS:
        if board[a] == board[b] == board[c] != 0:
            return board[a]
    return -1 if 0 not in board else 0


def free_cells(board: list) -> list:
    return [i for i, v in enumerate(board) if v == 0]


def _minimax(board, depth, is_maximising, ai, human, alpha, beta):
    result = check_winner(board)
    if result == ai:    return 10 - depth
    if result == human: return depth - 10
    if result == -1:    return 0

    moves = free_cells(board)
    if is_maximising:
        best = -float("inf")
        for m in moves:
            board[m] = ai
            best = max(best, _minimax(board, depth+1, False, ai, human, alpha, beta))
            board[m] = 0
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best
    else:
        best = float("inf")
        for m in moves:
            board[m] = human
            best = min(best, _minimax(board, depth+1, True, ai, human, alpha, beta))
            board[m] = 0
            beta = min(beta, best)
            if beta <= alpha:
                break
        return best


def minimax_move(board: list, ai_player: int, human_player: int) -> int:
    moves = free_cells(board)
    best_score, best_move = -float("inf"), moves[0]
    for m in moves:
        board[m] = ai_player
        score = _minimax(board, 0, False, ai_player, human_player, -float("inf"), float("inf"))
        board[m] = 0
        if score > best_score:
            best_score, best_move = score, m
    return best_move


class QLearningAgent:
    def __init__(self):
        self.q: dict = {}

    def q_val(self, state: tuple, action: int) -> float:
        return self.q.get((state, action), 0.0)

    def best_action(self, state: tuple, moves: list) -> int:
        return max(moves, key=lambda a: self.q_val(state, a))

    def choose(self, state: tuple, moves: list, epsilon: float = 0.0) -> int:
        if random.random() < epsilon:
            return random.choice(moves)
        return self.best_action(state, moves)

    def update(self, state, action, reward, next_state, next_moves):
        future  = max((self.q_val(next_state, a) for a in next_moves), default=0.0)
        current = self.q_val(state, action)
        self.q[(state, action)] = current + LEARNING_RATE * (
            reward + DISCOUNT * future - current
        )

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.q, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            self.q = pickle.load(f)


def train() -> QLearningAgent:
    print(f"\n🎓  Training AI for {EPISODES:,} episodes …  (one-time setup)\n")
    agent = QLearningAgent()

    wins = draws = losses = 0

    for ep in range(EPISODES):
        epsilon = max(0.05, 1.0 - ep / (EPISODES * 0.80))

        board           = [0] * 9
        prev_s, prev_a  = None, None

        while True:
            x_moves = free_cells(board)
            if not x_moves:
                if prev_s is not None:
                    agent.update(prev_s, prev_a, 0.4, tuple(board), [])
                draws += 1
                break

            board[random.choice(x_moves)] = 1
            result = check_winner(board)

            if result != 0:
                if prev_s is not None:
                    reward = -1.0 if result == 1 else 0.4
                    agent.update(prev_s, prev_a, reward, tuple(board), [])
                if result == 1:   losses += 1
                else:             draws  += 1
                break

            state   = tuple(board)
            o_moves = free_cells(board)
            action  = agent.choose(state, o_moves, epsilon)

            if prev_s is not None:
                agent.update(prev_s, prev_a, 0.0, state, o_moves)

            prev_s, prev_a = state, action
            board[action]  = 2

            result     = check_winner(board)
            next_state = tuple(board)
            next_moves = free_cells(board)

            if result == 2:
                agent.update(state, action, 1.0, next_state, [])
                wins  += 1
                break
            elif result == -1:
                agent.update(state, action, 0.5, next_state, [])
                draws += 1
                break

        if (ep + 1) % 50_000 == 0:
            total = wins + draws + losses or 1
            print(f"   Ep {ep+1:>7,}/{EPISODES:,} | "
                  f"Win {wins/total*100:4.1f}%  "
                  f"Draw {draws/total*100:4.1f}%  "
                  f"Loss {losses/total*100:4.1f}%  "
                  f"(ε={epsilon:.3f})")
            wins = draws = losses = 0

    agent.save(Q_TABLE_PATH)
    print(f"\n✅  Training complete — model saved to '{Q_TABLE_PATH}'\n")
    return agent


app: Flask = Flask(__name__)
agent: Optional[QLearningAgent] = None


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ai_move", methods=["POST"])
def ai_move():
    data       = request.json
    board      = data["board"]
    difficulty = data.get("difficulty", "hard")

    moves = free_cells(board)
    if not moves:
        return jsonify({"move": -1})

    if difficulty == "hard":
        move = minimax_move(list(board), ai_player=2, human_player=1)
    elif difficulty == "medium":
        move = agent.choose(tuple(board), moves, epsilon=0.30)
    else:
        move = agent.choose(tuple(board), moves, epsilon=0.85)

    return jsonify({"move": move})


if __name__ == "__main__":
    if os.path.exists(Q_TABLE_PATH):
        print("📂  Loading saved model …")
        agent = QLearningAgent()
        agent.load(Q_TABLE_PATH)
        q_size = len(agent.q)
        print(f"✅  Model loaded  ({q_size:,} Q-values)\n")
    else:
        agent = train()

    threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}")).start()

    print(f"🌐  Game running → http://127.0.0.1:{PORT}")
    print("     Press Ctrl+C to stop.\n")

    app.run(debug=False, port=PORT)