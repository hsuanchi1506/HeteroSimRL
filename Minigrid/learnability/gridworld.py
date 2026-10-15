import torch

EMPTY, WALL, DOOR, KEY, BALL, BOX, GOAL = range(7)
RED, GREEN, BLUE, PURPLE, YELLOW, GREY = range(6)
OPEN, CLOSED, LOCKED = range(3)
LEFT, RIGHT, FORWARD, PICKUP, DROP, TOGGLE = range(6)
N_ACTIONS = 6
PAD = 6
VIEW = 7
SMAX = 12
G = SMAX + 2 * PAD


def code(t, c=0, s=0):
    return t | (c << 3) | (s << 6)


WALL_C = code(WALL, GREY)
KEY_C = code(KEY, YELLOW)
BALL_C = code(BALL, RED)
BOX_C = code(BOX, BLUE)
GOAL_C = code(GOAL, GREEN)
BALL2_C = code(BALL, PURPLE)
DOOR_C = [code(DOOR, YELLOW, s) for s in range(3)]

PREDICATES = [
    "pick_key", "pick_ball", "pick_box", "face_door", "open_door", "enter_right",
    "close_door", "goal", "goal_with_key", "pick_ball2", "key_right", "ball_near_box",
    "key_near_ball", "ball_right", "goal_with_ball", "ball2_left", "ball_near_door", "key_near_goal",
    "open_with_ball", "close_from_right", "box_right", "ball2_near_goal", "line3", "box_corner",
    "pair_corner", "box_near_key", "key_near_box", "ball_corner", "key_corner", "box_near_door",
    "drop_key", "face_box", "key_near_door", "box_near_ball",
]
P = len(PREDICATES)
INTRINSIC = {"null": -1, "explore": -2, "impact": -3}

FEAT = 7 + 6 + 3
OBS_DIM = (VIEW * VIEW + 1) * FEAT


def feature_lut(device):
    lut = torch.zeros(256, FEAT, device=device)
    for c in range(256):
        t, col, s = c & 7, (c >> 3) & 7, c >> 6
        if t < 7:
            lut[c, t] = 1
        if col < 6 and t != EMPTY:
            lut[c, 7 + col] = 1
        if t == DOOR and s < 3:
            lut[c, 13 + s] = 1
    return lut


def featurize(codes, lut):
    return lut[codes.long()].flatten(-2)


def view_offsets(device):
    dirs = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=device)
    j, i = torch.meshgrid(torch.arange(VIEW, device=device), torch.arange(VIEW, device=device), indexing="ij")
    off = []
    for d in range(4):
        f = dirs[d]
        r = torch.stack([-f[1], f[0]])
        off.append((6 - j).reshape(-1, 1) * f + (i - 3).reshape(-1, 1) * r)
    return dirs, torch.stack(off)


def gen_layouts(n, S, gen, device, variant=0):
    yy, xx = torch.meshgrid(torch.arange(G, device=device), torch.arange(G, device=device), indexing="ij")
    grid = torch.full((n, G, G), WALL_C, dtype=torch.uint8, device=device)
    inside = (yy >= PAD + 1) & (yy <= PAD + S - 2) & (xx >= PAD + 1) & (xx <= PAD + S - 2)
    wx = torch.randint(3, S - 3, (n,), generator=gen, device=device) + PAD
    dy = torch.randint(1, S - 1, (n,), generator=gen, device=device) + PAD
    X, Y = xx[None], yy[None]
    grid[inside[None] & (X != wx[:, None, None])] = EMPTY
    ar = torch.arange(n, device=device)
    grid[ar, dy, wx] = DOOR_C[LOCKED]
    left = inside[None] & (X < wx[:, None, None])
    right = inside[None] & (X > wx[:, None, None])
    blocked = ((Y == dy[:, None, None]) & ((X == wx[:, None, None] - 1) | (X == wx[:, None, None] + 1)))
    free_l, free_r = left & ~blocked, right & ~blocked

    def place(free):
        score = torch.rand(n, G, G, generator=gen, device=device)
        score[~free] = -1
        idx = score.flatten(1).argmax(1)
        y, x = idx // G, idx % G
        free[ar, y, x] = False
        return y, x

    ay, ax = place(free_l)
    for c, free in [(KEY_C, free_l), (BALL_C, free_l), (BOX_C, free_l), (GOAL_C, free_r), (BALL2_C, free_r)]:
        y, x = place(free)
        grid[ar, y, x] = c
    agent_dir = torch.randint(0, 4, (n,), generator=gen, device=device)
    carry = torch.zeros(n, dtype=torch.long, device=device)
    if variant == 1:
        grid[grid == KEY_C] = EMPTY
        carry[:] = KEY_C
    if variant == 2:
        grid[ar, dy, wx] = DOOR_C[OPEN]
    return dict(grid=grid, pos=torch.stack([ax, ay], 1), dir=agent_dir, wx=wx, door=torch.stack([wx, dy], 1), carry=carry)


class World:
    def __init__(self, task, size, max_steps, device="cuda", bank=2 ** 17, seed=0, desync=True, variant=None, once=None):
        self.device = device
        self.once = torch.zeros(len(task), dtype=torch.bool, device=device) if once is None else once.to(device)
        self.B = len(task)
        self.task = task.to(device)
        self.max_steps = max_steps.to(device)
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)
        variant = torch.zeros_like(size) if variant is None else variant
        keys = sorted(set(zip(size.tolist(), variant.tolist())))
        banks = [gen_layouts(bank, S, self.gen, device, v) for S, v in keys]
        self.bank = {k: torch.cat([b[k] for b in banks]) for k in banks[0]}
        self.bank_start = torch.tensor([keys.index(k) * bank for k in zip(size.tolist(), variant.tolist())], device=device)
        self.bank_n = bank
        self.dirs, self.offsets = view_offsets(device)
        self.ar = torch.arange(self.B, device=device)
        self.hash_w = torch.randint(-2 ** 62, 2 ** 62, (256, G * G), generator=self.gen, device=device)
        self.carry_w = torch.randint(-2 ** 62, 2 ** 62, (256,), generator=self.gen, device=device)
        self.n_hash = 64
        self.cells = torch.arange(G * G, device=device)
        self.grid = torch.zeros(self.B, G, G, dtype=torch.uint8, device=device)
        self.pos = torch.zeros(self.B, 2, dtype=torch.long, device=device)
        self.dir = torch.zeros(self.B, dtype=torch.long, device=device)
        self.wx = torch.zeros(self.B, dtype=torch.long, device=device)
        self.door = torch.zeros(self.B, 2, dtype=torch.long, device=device)
        self.carry = torch.zeros(self.B, dtype=torch.long, device=device)
        self.t = torch.zeros(self.B, dtype=torch.long, device=device)
        self.hit = torch.zeros(self.B, P, dtype=torch.bool, device=device)
        self.visited = torch.zeros(self.B, G, G, dtype=torch.bool, device=device)
        self.hashes = torch.zeros(self.B, self.n_hash, dtype=torch.long, device=device)
        self.n_seen = torch.zeros(self.B, dtype=torch.long, device=device)
        self.reset_mask(torch.ones(self.B, dtype=torch.bool, device=device))
        if desync:
            self.t = (torch.rand(self.B, generator=self.gen, device=device) * self.max_steps).long()

    def reset_mask(self, m):
        idx = self.bank_start + torch.randint(0, self.bank_n, (self.B,), generator=self.gen, device=self.device)
        mm = m[:, None]
        self.grid = torch.where(m[:, None, None], self.bank["grid"][idx], self.grid)
        self.pos = torch.where(mm, self.bank["pos"][idx], self.pos)
        self.dir = torch.where(m, self.bank["dir"][idx], self.dir)
        self.wx = torch.where(m, self.bank["wx"][idx], self.wx)
        self.door = torch.where(mm, self.bank["door"][idx], self.door)
        self.carry = torch.where(m, self.bank["carry"][idx], self.carry)
        self.t = torch.where(m, 0, self.t)
        self.hit &= ~mm
        self.visited &= ~m[:, None, None]
        self.visited[self.ar, self.pos[:, 1], self.pos[:, 0]] |= m
        h = self.config_hash()
        self.hashes = torch.where(mm, h[:, None], self.hashes)
        self.n_seen = torch.where(m, 1, self.n_seen)

    def config_hash(self):
        g = self.grid.flatten(1).long()
        return self.hash_w[g, self.cells].sum(1) + self.carry_w[self.carry]

    def obs(self):
        xy = self.pos[:, None, :] + self.offsets[self.dir]
        v = self.grid[self.ar[:, None], xy[..., 1], xy[..., 0]]
        return torch.cat([v, self.carry[:, None].to(torch.uint8)], 1)

    def predicates(self, dropped, front_after):
        dk, db, dx, d2 = dropped == KEY_C, dropped == BALL_C, dropped == BOX_C, dropped == BALL2_C
        g = self.grid
        door = g[self.ar, self.door[:, 1], self.door[:, 0]]
        under = g[self.ar, self.pos[:, 1], self.pos[:, 0]]
        in_right = self.pos[:, 0] > self.wx
        xs = torch.arange(G, device=self.device)
        right_cols = xs[None, None, :] > self.wx[:, None, None]
        ball, box, key, ball2 = g == BALL_C, g == BOX_C, g == KEY_C, g == BALL2_C
        goal, door_m, wall_m = g == GOAL_C, (g & 7) == DOOR, (g & 7) == WALL

        def adj(a, b):
            n = torch.zeros_like(b)
            n[:, 1:] |= b[:, :-1]
            n[:, :-1] |= b[:, 1:]
            n[:, :, 1:] |= b[:, :, :-1]
            n[:, :, :-1] |= b[:, :, 1:]
            return (a & n).flatten(1).any(1)

        on_goal = under == GOAL_C
        cornerm = (wall_m.roll(1, 1) | wall_m.roll(-1, 1)) & (wall_m.roll(1, 2) | wall_m.roll(-1, 2))
        corner = box & cornerm

        return torch.stack([
            self.carry == KEY_C,
            self.carry == BALL_C,
            self.carry == BOX_C,
            (front_after & 7) == DOOR,
            door == DOOR_C[OPEN],
            in_right,
            door == DOOR_C[CLOSED],
            on_goal,
            on_goal & (self.carry == KEY_C),
            self.carry == BALL2_C,
            (key & right_cols).flatten(1).any(1),
            (db | dx) & adj(ball, box),
            (dk | db) & adj(key, ball),
            (ball & right_cols).flatten(1).any(1),
            on_goal & (self.carry == BALL_C),
            (ball2 & ~right_cols).flatten(1).any(1),
            db & adj(ball, door_m),
            dk & adj(key, goal),
            (door == DOOR_C[OPEN]) & (self.carry == BALL_C),
            in_right & (door == DOOR_C[CLOSED]),
            (box & right_cols).flatten(1).any(1),
            d2 & adj(ball2, goal),
            (dk | db | dx) & adj(key, ball) & adj(ball, box),
            dx & corner.flatten(1).any(1),
            (db | dx) & adj(ball, corner),
            dx & adj(box, key),
            dk & adj(key, box),
            db & (ball & cornerm).flatten(1).any(1),
            dk & (key & cornerm).flatten(1).any(1),
            dx & adj(box, door_m),
            dk,
            (front_after & 7) == BOX,
            dk & adj(key, door_m),
            dx & adj(box, ball),
        ], 1)

    def step(self, a, reset=True):
        ar = self.ar
        fwd = self.pos + self.dirs[self.dir]
        front = self.grid[ar, fwd[:, 1], fwd[:, 0]].long()
        ft, fc, fs = front & 7, (front >> 3) & 7, front >> 6
        self.dir = torch.where(a == LEFT, (self.dir - 1) % 4, torch.where(a == RIGHT, (self.dir + 1) % 4, self.dir))
        can_move = (ft == EMPTY) | (ft == GOAL) | ((ft == DOOR) & (fs == OPEN))
        mv = (a == FORWARD) & can_move
        self.pos = torch.where(mv[:, None], fwd, self.pos)
        pk = (a == PICKUP) & ((ft == KEY) | (ft == BALL) | (ft == BOX)) & (self.carry == 0)
        dp = (a == DROP) & (ft == EMPTY) & (self.carry != 0)
        tg = (a == TOGGLE) & (ft == DOOR)
        unlock = tg & (fs == LOCKED) & ((self.carry & 7) == KEY) & (((self.carry >> 3) & 7) == fc)
        new_door = torch.where(unlock | (tg & (fs == CLOSED)), OPEN, torch.where(tg & (fs == OPEN), CLOSED, fs))
        new_front = torch.where(pk, 0, torch.where(dp, self.carry, torch.where(ft == DOOR, (front & 63) | (new_door << 6), front)))
        self.grid[ar, fwd[:, 1], fwd[:, 0]] = new_front.to(torch.uint8)
        carry_before = self.carry
        self.carry = torch.where(pk, front, torch.where(dp, 0, self.carry))
        self.t += 1

        fwd2 = self.pos + self.dirs[self.dir]
        front_after = self.grid[ar, fwd2[:, 1], fwd2[:, 0]].long()
        pred = self.predicates(torch.where(dp, carry_before, -1), front_after)
        first = pred & ~self.hit
        self.hit |= pred
        tid = self.task.clamp(min=0)
        success = (self.task >= 0) & pred[ar, tid]
        reward = torch.where(success & (first[ar, tid] | ~self.once), 1 - 0.9 * self.t / self.max_steps, 0.0)

        new_cell = ~self.visited[ar, self.pos[:, 1], self.pos[:, 0]]
        self.visited[ar, self.pos[:, 1], self.pos[:, 0]] = True
        reward = reward + torch.where(self.task == INTRINSIC["explore"], 0.05 * new_cell, 0.0)
        changed = pk | dp | tg
        h = self.config_hash()
        novel = changed & ~(self.hashes == h[:, None]).any(1)
        slot = self.n_seen % self.n_hash
        self.hashes[ar, slot] = torch.where(novel, h, self.hashes[ar, slot])
        self.n_seen += novel
        reward = reward + torch.where(self.task == INTRINSIC["impact"], 0.2 * novel, 0.0)

        done = (success & ~self.once) | (self.t >= self.max_steps)
        info = dict(success=torch.where(self.once, (self.task >= 0) & self.hit[ar, tid], success), hit=self.hit.clone(), first=first, t=self.t.clone(), cells=self.visited.flatten(1).sum(1))
        if reset:
            self.reset_mask(done)
        return self.obs(), reward, done, info
