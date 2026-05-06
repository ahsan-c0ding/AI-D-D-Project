"""
Tkinter GUI for the AI Dungeon Master.

Two tabs are stacked in a Notebook:

  1. Generation Inspector — the marquee tab. Pick an algorithm
     (CSP / BFS / DFS / Greedy) and watch the dungeon get built one
     step at a time. CSP shows backtracking; the others show their
     characteristic expansion patterns. Counters track nodes explored
     and backtracks live. This is the tab that demonstrates that AI
     algorithms are actually doing the work.

  2. Play — the playable D&D layer (move, fight, talk, skill checks)
     on top of a CSP-generated dungeon. Useful for showing the
     decision tree + Bayesian combat in action.

The GUI is a thin shell over GameState (game.py) and the algorithm
solve_steps() generators (csp_generator.py, comparison_generators.py).
All AI logic lives in those modules.
"""

from __future__ import annotations

import argparse
import tkinter as tk
import random
from tkinter import font as tkfont
from tkinter import messagebox, ttk
from typing import Optional

from bayesian_combat import (
    BayesianCombatSystem,
    CombatOutcome,
    DiceRoller,
)
from comparison_generators import (
    BFSDungeonGenerator,
    DFSDungeonGenerator,
    GreedyDungeonGenerator,
)
from csp_generator import DungeonCSP
from game import GameState
from models import Item, ItemType, NPC, NPCType, Player, RoomType
from npc_decision_tree import (
    DecisionNode,
    NPCAction,
    NPCBehaviorManager,
    NPCDecisionTree,
)

# Optional matplotlib embedding for calibration plot. Tab will degrade
# gracefully if matplotlib isn't installed (the rest of the GUI still works).
try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    _HAVE_MATPLOTLIB = True
except Exception:
    _HAVE_MATPLOTLIB = False


# Visual constants

ROOM_COLORS = {
    RoomType.START:    "#5b8def",
    RoomType.NORMAL:   "#8b8b8b",
    RoomType.TREASURE: "#f1c40f",
    RoomType.BOSS:     "#c0392b",
    RoomType.TRAP:     "#9b59b6",
    RoomType.MERCHANT: "#27ae60",
}
UNVISITED_COLOR = "#2b2b2b"
FOG_COLOR = "#1a1a1a"
PLAYER_COLOR = "#ffffff"
CONNECTION_COLOR = "#666666"
GRID_BG = "#0e0e10"
HIGHLIGHT_SELECT = "#f1c40f"
HIGHLIGHT_BACKTRACK = "#e74c3c"
HIGHLIGHT_CONSIDER = "#3498db"

CELL_PX = 60
PADDING = 12

ALGORITHMS = {
    "CSP (backtracking)": DungeonCSP,
    "BFS":                BFSDungeonGenerator,
    "DFS":                DFSDungeonGenerator,
    "Greedy":             GreedyDungeonGenerator,
}


# Generation Inspector tab

class GenerationInspectorTab:
    """Animates a dungeon-generation algorithm one step at a time."""

    def __init__(self, parent: tk.Widget):
        self.parent = parent
        self.mono = tkfont.Font(family="Consolas", size=10)
        self.bold = tkfont.Font(family="Consolas", size=11, weight="bold")
        self.title_f = tkfont.Font(family="Segoe UI", size=14, weight="bold")

        self.gen = None                      # current generator instance
        self.step_iter = None                # solve_steps generator
        self.last_event: Optional[dict] = None
        self.flash_coords: Optional[tuple] = None
        self.flash_kind: Optional[str] = None
        self._after_id: Optional[str] = None
        self.running = False
        self.event_count = 0

        self._build()
        # Initial blank canvas with default config
        self.reset()

    # layout

    def _build(self):
        outer = tk.Frame(self.parent, bg=GRID_BG)
        outer.pack(fill=tk.BOTH, expand=True, padx=PADDING, pady=PADDING)

        # Top: control bar
        controls = tk.Frame(outer, bg=GRID_BG)
        controls.pack(fill=tk.X, pady=(0, 8))

        tk.Label(controls, text="Algorithm:", bg=GRID_BG, fg="#e6e6e6",
                 font=self.mono).pack(side=tk.LEFT, padx=(0, 4))
        self.algo_var = tk.StringVar(value="CSP (backtracking)")
        self.algo_menu = ttk.Combobox(controls, textvariable=self.algo_var,
                                      values=list(ALGORITHMS.keys()),
                                      width=22, state="readonly")
        self.algo_menu.pack(side=tk.LEFT, padx=(0, 12))

        for label, var, default, lo, hi in [
            ("Grid", "grid_var", 6, 4, 14),
            ("Rooms", "rooms_var", 10, 3, 80),
            ("Seed", "seed_var", 42, 0, 9999),
        ]:
            tk.Label(controls, text=f"{label}:", bg=GRID_BG, fg="#e6e6e6",
                     font=self.mono).pack(side=tk.LEFT, padx=(0, 4))
            v = tk.IntVar(value=default)
            setattr(self, var, v)
            tk.Spinbox(controls, from_=lo, to=hi, width=5, textvariable=v,
                       font=self.mono).pack(side=tk.LEFT, padx=(0, 12))

        tk.Label(controls, text="Speed (ms/step):", bg=GRID_BG, fg="#e6e6e6",
                 font=self.mono).pack(side=tk.LEFT, padx=(0, 4))
        self.speed_var = tk.IntVar(value=80)
        tk.Spinbox(controls, from_=1, to=2000, width=6,
                   textvariable=self.speed_var,
                   font=self.mono).pack(side=tk.LEFT, padx=(0, 12))

        # Action buttons
        actions = tk.Frame(outer, bg=GRID_BG)
        actions.pack(fill=tk.X, pady=(0, 8))
        tk.Button(actions, text="Reset", command=self.reset,
                  width=10).pack(side=tk.LEFT, padx=2)
        self.btn_step = tk.Button(actions, text="Step ▷", command=self.step_once,
                                  width=10)
        self.btn_step.pack(side=tk.LEFT, padx=2)
        self.btn_run = tk.Button(actions, text="Animate ▶", command=self.toggle_run,
                                 width=12)
        self.btn_run.pack(side=tk.LEFT, padx=2)
        self.btn_finish = tk.Button(actions, text="Run to end ⏭",
                                    command=self.run_to_end, width=14)
        self.btn_finish.pack(side=tk.LEFT, padx=2)

        # Body: canvas + side info
        body = tk.Frame(outer, bg=GRID_BG)
        body.pack(fill=tk.BOTH, expand=True)

        # canvas (size will be re-set on reset)
        self.canvas = tk.Canvas(body, bg=GRID_BG, highlightthickness=0,
                                width=14 * CELL_PX, height=14 * CELL_PX)
        self.canvas.pack(side=tk.LEFT, anchor=tk.N)

        right = tk.Frame(body, bg=GRID_BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(PADDING, 0))

        self._build_counter_panel(right)
        self._build_legend_panel(right)
        self._build_event_log(right)

    def _section(self, parent: tk.Widget, title: str) -> tk.Frame:
        wrapper = tk.Frame(parent, bg=GRID_BG, pady=4)
        wrapper.pack(fill=tk.X, pady=(0, 8))
        tk.Label(wrapper, text=title, font=self.title_f,
                 fg="#e6e6e6", bg=GRID_BG, anchor="w").pack(fill=tk.X)
        body = tk.Frame(wrapper, bg="#1a1a1f", padx=10, pady=8)
        body.pack(fill=tk.X)
        return body

    def _build_counter_panel(self, parent):
        body = self._section(parent, "Algorithm state")
        self.counter_text = tk.Label(body, font=self.mono, bg="#1a1a1f",
                                     fg="#e6e6e6", justify="left")
        self.counter_text.pack(anchor="w")

    def _build_legend_panel(self, parent):
        body = self._section(parent, "Legend")
        legend = tk.Frame(body, bg="#1a1a1f")
        legend.pack(fill=tk.X)
        items = [
            ("S", ROOM_COLORS[RoomType.START],    "Start"),
            ("·", ROOM_COLORS[RoomType.NORMAL],   "Normal"),
            ("T", ROOM_COLORS[RoomType.TREASURE], "Treasure"),
            ("B", ROOM_COLORS[RoomType.BOSS],     "Boss"),
            ("X", ROOM_COLORS[RoomType.TRAP],     "Trap"),
            ("M", ROOM_COLORS[RoomType.MERCHANT], "Merchant"),
        ]
        for ch, color, name in items:
            row = tk.Frame(legend, bg="#1a1a1f")
            row.pack(anchor="w")
            sq = tk.Label(row, text=f" {ch} ", bg=color, fg="#0a0a0a",
                          font=self.bold, width=3)
            sq.pack(side=tk.LEFT, padx=(0, 6))
            tk.Label(row, text=name, bg="#1a1a1f", fg="#cfcfcf",
                     font=self.mono).pack(side=tk.LEFT)
        # Highlight legend
        for color, label in [
            (HIGHLIGHT_SELECT, "= currently selected (variable being tried)"),
            (HIGHLIGHT_CONSIDER, "= considering value"),
            (HIGHLIGHT_BACKTRACK, "= just backtracked (assignment undone)"),
        ]:
            row = tk.Frame(legend, bg="#1a1a1f")
            row.pack(anchor="w")
            sq = tk.Label(row, text="    ", bg=color, font=self.bold, width=3)
            sq.pack(side=tk.LEFT, padx=(0, 6))
            tk.Label(row, text=label, bg="#1a1a1f", fg="#cfcfcf",
                     font=self.mono).pack(side=tk.LEFT)

    def _build_event_log(self, parent):
        body = self._section(parent, "Event trace")
        self.log_box = tk.Text(body, height=14, width=42, font=self.mono,
                               bg="#0e0e10", fg="#cfcfcf",
                               borderwidth=0, wrap=tk.NONE)
        self.log_box.pack(fill=tk.BOTH, expand=True)
        self.log_box.config(state=tk.DISABLED)

    # control flow

    def _make_generator(self):
        cls = ALGORITHMS[self.algo_var.get()]
        try:
            return cls(self.grid_var.get(), self.grid_var.get(),
                       self.rooms_var.get(), seed=self.seed_var.get())
        except Exception as e:
            messagebox.showerror("Init error", str(e))
            return None

    def reset(self):
        self._stop()
        self.gen = self._make_generator()
        if self.gen is None:
            return
        self.step_iter = self.gen.solve_steps()
        self.last_event = None
        self.flash_coords = None
        self.flash_kind = None
        self.event_count = 0
        # Resize canvas to grid
        cw = self.gen.width * CELL_PX + 2 * PADDING
        ch = self.gen.height * CELL_PX + 2 * PADDING
        self.canvas.config(width=cw, height=ch)
        self._clear_log()
        self._log(f"--- {self.algo_var.get()} on "
                  f"{self.gen.width}x{self.gen.height} grid, "
                  f"{self.gen.num_rooms} rooms, seed={self.seed_var.get()} ---")
        self._refresh()

    def step_once(self):
        if self.step_iter is None:
            return False
        try:
            event = next(self.step_iter)
        except StopIteration:
            self._on_complete()
            return False
        self._on_event(event)
        return True

    def toggle_run(self):
        if self.running:
            self._stop()
        else:
            self.running = True
            self.btn_run.config(text="Pause ⏸")
            self._tick()

    def _tick(self):
        if not self.running:
            return
        keep_going = self.step_once()
        if not keep_going:
            return
        self._after_id = self.parent.after(self.speed_var.get(), self._tick)

    def _stop(self):
        self.running = False
        self.btn_run.config(text="Animate ▶")
        if self._after_id is not None:
            try:
                self.parent.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def run_to_end(self):
        """Skip animation, run the search to completion as fast as possible."""
        if self.step_iter is None:
            return
        # Limit so a runaway never freezes the UI
        for _ in range(500_000):
            try:
                event = next(self.step_iter)
            except StopIteration:
                break
            self.last_event = event
            self.event_count += 1
            if event["kind"] in ("succeed", "fail"):
                break
        self._on_complete()

    # event handling

    def _on_event(self, event: dict):
        self.last_event = event
        self.event_count += 1
        kind = event["kind"]
        if kind in ("select", "consider"):
            self.flash_coords = event["coords"]
            self.flash_kind = kind
        elif kind == "assign":
            self.flash_coords = event["coords"]
            self.flash_kind = "assign"
        elif kind == "backtrack":
            self.flash_coords = event["coords"]
            self.flash_kind = "backtrack"
        elif kind == "reject":
            self.flash_coords = event["coords"]
            self.flash_kind = "reject"
        elif kind in ("succeed", "fail"):
            self.flash_coords = None
            self.flash_kind = None

        self._log_event(event)
        self._refresh()

        if kind in ("succeed", "fail"):
            self._on_complete()

    def _on_complete(self):
        self._stop()
        if self.last_event is None:
            return
        if self.last_event.get("kind") == "succeed":
            try:
                self.gen.finalize()
            except Exception as e:
                self._log(f"finalize error: {e}")
            self._log(f"COMPLETE in {self.event_count} events. "
                      f"nodes_explored={self.last_event['nodes']} "
                      f"backtracks={self.last_event['backtracks']}")
        elif self.last_event.get("kind") == "fail":
            self._log(f"FAILED. nodes_explored={self.last_event['nodes']} "
                      f"backtracks={self.last_event['backtracks']}")
        self._refresh()

    # rendering 

    def _refresh(self):
        self._draw_canvas()
        self._draw_counters()

    def _draw_canvas(self):
        self.canvas.delete("all")
        if self.gen is None:
            return
        d = self.gen.dungeon

        # Connections
        for coords, room in d.rooms.items():
            cx1, cy1 = self._cell_center(coords)
            for nb in room.connections:
                if nb in d.rooms:
                    cx2, cy2 = self._cell_center(nb)
                    self.canvas.create_line(cx1, cy1, cx2, cy2,
                                            fill=CONNECTION_COLOR, width=3)

        # Cells
        for x in range(self.gen.width):
            for y in range(self.gen.height):
                room = d.rooms.get((x, y))
                x0, y0, x1, y1 = self._cell_box((x, y))
                if room is None:
                    self.canvas.create_rectangle(
                        x0, y0, x1, y1, fill=UNVISITED_COLOR,
                        outline="#222", width=1)
                else:
                    color = ROOM_COLORS.get(room.room_type, "#444")
                    self.canvas.create_rectangle(
                        x0, y0, x1, y1, fill=color, outline="#333", width=2)
                    glyph = {
                        RoomType.START: "S", RoomType.BOSS: "B",
                        RoomType.TREASURE: "T", RoomType.MERCHANT: "M",
                        RoomType.TRAP: "X", RoomType.NORMAL: "·",
                    }.get(room.room_type, "?")
                    self.canvas.create_text(
                        (x0 + x1) / 2, (y0 + y1) / 2,
                        text=glyph, fill="#0a0a0a", font=self.bold)

        # Highlight current activity (selection / assignment / backtrack)
        if self.flash_coords is not None:
            x0, y0, x1, y1 = self._cell_box(self.flash_coords)
            color = {
                "select": HIGHLIGHT_SELECT,
                "consider": HIGHLIGHT_CONSIDER,
                "assign": HIGHLIGHT_SELECT,
                "backtrack": HIGHLIGHT_BACKTRACK,
                "reject": HIGHLIGHT_BACKTRACK,
            }.get(self.flash_kind, HIGHLIGHT_SELECT)
            self.canvas.create_rectangle(
                x0 - 3, y0 - 3, x1 + 3, y1 + 3,
                outline=color, width=4)

    def _cell_box(self, coords):
        x, y = coords
        x0 = PADDING + x * CELL_PX + 6
        y0 = PADDING + y * CELL_PX + 6
        x1 = x0 + CELL_PX - 12
        y1 = y0 + CELL_PX - 12
        return x0, y0, x1, y1

    def _cell_center(self, coords):
        x0, y0, x1, y1 = self._cell_box(coords)
        return (x0 + x1) / 2, (y0 + y1) / 2

    def _draw_counters(self):
        if self.gen is None:
            self.counter_text.config(text="(no run)")
            return
        e = self.last_event or {}
        rooms_assigned = len(getattr(self.gen, "assignment", {})) \
            if hasattr(self.gen, "assignment") else len(self.gen.dungeon.rooms)
        text = (
            f"Algorithm        : {self.algo_var.get()}\n"
            f"Grid             : {self.gen.width} x {self.gen.height}\n"
            f"Target rooms     : {self.gen.num_rooms}\n"
            f"Rooms assigned   : {rooms_assigned}\n"
            f"Events processed : {self.event_count}\n"
            f"Nodes explored   : {self.gen.nodes_explored}\n"
            f"Backtracks       : {self.gen.backtrack_count}\n"
            f"\nLast event: {e.get('kind', '—')}\n"
        )
        if e:
            for k in ("coords", "room_type", "depth", "note"):
                if k in e:
                    text += f"  {k}: {e[k]}\n"
        self.counter_text.config(text=text)

    # log 

    def _log_event(self, event: dict):
        kind = event["kind"]
        if kind == "select":
            self._log(f"[{self.event_count:>4}] select   {event['coords']} "
                      f"(depth {event.get('depth', '?')})")
        elif kind == "consider":
            self._log(f"[{self.event_count:>4}] consider {event['coords']} "
                      f"= {event['room_type']}")
        elif kind == "reject":
            self._log(f"[{self.event_count:>4}] REJECT   {event['coords']} "
                      f"= {event['room_type']}  ({event.get('reason', '')})")
        elif kind == "assign":
            self._log(f"[{self.event_count:>4}] ASSIGN   {event['coords']} "
                      f"= {event['room_type']}  conn={event['connections']}")
        elif kind == "backtrack":
            self._log(f"[{self.event_count:>4}] ↶ BACKTRK {event['coords']} "
                      f"(undo {event['room_type']})")
        elif kind == "succeed":
            self._log(f"[{self.event_count:>4}] ✓ SUCCESS  "
                      f"nodes={event['nodes']} backtracks={event['backtracks']}")
        elif kind == "fail":
            self._log(f"[{self.event_count:>4}] ✗ FAIL     "
                      f"nodes={event['nodes']} backtracks={event['backtracks']}")

    def _log(self, msg: str):
        self.log_box.config(state=tk.NORMAL)
        self.log_box.insert(tk.END, msg + "\n")
        self.log_box.see(tk.END)
        self.log_box.config(state=tk.DISABLED)

    def _clear_log(self):
        self.log_box.config(state=tk.NORMAL)
        self.log_box.delete("1.0", tk.END)
        self.log_box.config(state=tk.DISABLED)


# NPC Brain tab — interactive decision tree visualization

# Action -> color used for leaf nodes
ACTION_COLORS = {
    NPCAction.ATTACK:    "#c0392b",
    NPCAction.FLEE:      "#e67e22",
    NPCAction.DEFEND:    "#2980b9",
    NPCAction.TALK:      "#27ae60",
    NPCAction.TRADE:     "#16a085",
    NPCAction.HELP:      "#8e44ad",
    NPCAction.IDLE:      "#7f8c8d",
    NPCAction.STEAL:     "#d35400",
    NPCAction.SURRENDER: "#bdc3c7",
}


class NPCBrainTab:
    """
    Interactive decision-tree explorer.

    User picks an NPC type, twiddles sliders for the player/world state,
    and the tree on the left re-evaluates live, highlighting the path
    taken to its leaf action. Demonstrates the AI doing reasoning.
    """

    NODE_W = 170
    NODE_H = 44
    LEVEL_H = 110
    CANVAS_W = 820
    CANVAS_H = 540

    NPC_TYPES = [t for t in NPCType]

    def __init__(self, parent: tk.Widget):
        self.parent = parent
        self.mono = tkfont.Font(family="Consolas", size=9)
        self.bold = tkfont.Font(family="Consolas", size=10, weight="bold")
        self.title_f = tkfont.Font(family="Segoe UI", size=14, weight="bold")

        self.brain = NPCBehaviorManager()
        self.current_tree: Optional[DecisionNode] = None
        self.node_positions: dict = {}  # id(node) -> (x, y)
        self.current_path: list = []
        self.current_branches: list = []
        self.current_action: Optional[NPCAction] = None

        self._build()
        self._on_state_change()  # initial render

    # layout 

    def _build(self):
        outer = tk.Frame(self.parent, bg=GRID_BG)
        outer.pack(fill=tk.BOTH, expand=True, padx=PADDING, pady=PADDING)

        body = tk.Frame(outer, bg=GRID_BG)
        body.pack(fill=tk.BOTH, expand=True)

        # Tree canvas (left)
        self.canvas = tk.Canvas(body, width=self.CANVAS_W, height=self.CANVAS_H,
                                bg=GRID_BG, highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, anchor=tk.N)

        # Controls (right)
        right = tk.Frame(body, bg=GRID_BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(PADDING, 0))

        self._build_controls(right)
        self._build_result_panel(right)
        self._build_explanation_panel(right)

    def _section(self, parent, title):
        wrapper = tk.Frame(parent, bg=GRID_BG, pady=4)
        wrapper.pack(fill=tk.X, pady=(0, 8))
        tk.Label(wrapper, text=title, font=self.title_f,
                 fg="#e6e6e6", bg=GRID_BG, anchor="w").pack(fill=tk.X)
        body = tk.Frame(wrapper, bg="#1a1a1f", padx=10, pady=8)
        body.pack(fill=tk.X)
        return body

    def _build_controls(self, parent):
        body = self._section(parent, "Inputs")

        # NPC type
        row = tk.Frame(body, bg="#1a1a1f")
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text="NPC type:", bg="#1a1a1f", fg="#cfcfcf",
                 font=self.mono, width=14, anchor="w").pack(side=tk.LEFT)
        self.npc_type_var = tk.StringVar(value=NPCType.ENEMY.value)
        cb = ttk.Combobox(row, textvariable=self.npc_type_var,
                          values=[t.value for t in self.NPC_TYPES],
                          width=14, state="readonly")
        cb.pack(side=tk.LEFT)
        cb.bind("<<ComboboxSelected>>", lambda e: self._on_state_change())

        # NPC name (only matters for the neutral 'thief' branch)
        row = tk.Frame(body, bg="#1a1a1f")
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text="NPC name:", bg="#1a1a1f", fg="#cfcfcf",
                 font=self.mono, width=14, anchor="w").pack(side=tk.LEFT)
        self.npc_name_var = tk.StringVar(value="Goblin")
        e = tk.Entry(row, textvariable=self.npc_name_var, width=18,
                     font=self.mono, bg="#0e0e10", fg="#fafafa",
                     insertbackground="#fafafa")
        e.pack(side=tk.LEFT)
        e.bind("<KeyRelease>", lambda evt: self._on_state_change())
        tk.Label(body, text='  (try "Mysterious Thief" to enable steal branch)',
                 bg="#1a1a1f", fg="#888", font=self.mono).pack(anchor="w")

        # Numeric inputs as scales
        self.scale_vars = {}
        for label, key, lo, hi, default in [
            ("Player HP",    "player_hp", 0, 100, 80),
            ("Player max HP", "player_max_hp", 1, 200, 100),
            ("NPC HP",       "npc_hp",    0, 200, 30),
            ("NPC defense",  "npc_def",   0, 20,  3),
            ("Player gold",  "gold",      0, 500, 100),
            ("Inv. items",   "inventory", 0, 10,  0),
        ]:
            row = tk.Frame(body, bg="#1a1a1f")
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=f"{label}:", bg="#1a1a1f", fg="#cfcfcf",
                     font=self.mono, width=14, anchor="w").pack(side=tk.LEFT)
            v = tk.IntVar(value=default)
            self.scale_vars[key] = v
            s = tk.Scale(row, from_=lo, to=hi, orient=tk.HORIZONTAL,
                         variable=v, length=180, bg="#1a1a1f",
                         fg="#cfcfcf", troughcolor="#0e0e10",
                         highlightthickness=0, font=self.mono,
                         command=lambda val: self._on_state_change())
            s.pack(side=tk.LEFT)

        # Boolean flags
        self.flag_vars = {}
        for label, key, default in [
            ("Player attacked first", "player_attacked", False),
            ("NPC has been met before", "npc_met", False),
        ]:
            v = tk.BooleanVar(value=default)
            self.flag_vars[key] = v
            cb = tk.Checkbutton(body, text=label, variable=v,
                                bg="#1a1a1f", fg="#cfcfcf",
                                selectcolor="#0e0e10",
                                activebackground="#1a1a1f",
                                activeforeground="#fafafa",
                                font=self.mono,
                                command=self._on_state_change)
            cb.pack(anchor="w")

    def _build_result_panel(self, parent):
        body = self._section(parent, "Result")
        self.result_label = tk.Label(body, font=self.bold,
                                     bg="#1a1a1f", fg="#fafafa",
                                     justify="left", anchor="w")
        self.result_label.pack(fill=tk.X)
        self.path_label = tk.Label(body, font=self.mono,
                                   bg="#1a1a1f", fg="#cfcfcf",
                                   justify="left", anchor="w")
        self.path_label.pack(fill=tk.X)

    def _build_explanation_panel(self, parent):
        body = self._section(parent, "How to read this")
        text = (
            "Yellow path  = decisions taken for this state\n"
            "Green edge   = condition was True\n"
            "Red edge     = condition was False\n"
            "Coloured leaf= chosen NPC action\n"
            "\n"
            "Try: drag Player HP down to 25 with NPC type=enemy.\n"
            "Watch the path flip from DEFEND/ATTACK to ATTACK\n"
            "(aggressive when player is wounded)."
        )
        tk.Label(body, text=text, font=self.mono, bg="#1a1a1f",
                 fg="#cfcfcf", justify="left", anchor="w",
                 wraplength=320).pack(fill=tk.X)

    # -- state assembly 

    def _build_player(self) -> Player:
        return Player(
            name="Hero",
            hp=self.scale_vars["player_hp"].get(),
            max_hp=max(1, self.scale_vars["player_max_hp"].get()),
            attack=15, defense=8, position=(0, 0),
            gold=self.scale_vars["gold"].get(),
            inventory=[None] * self.scale_vars["inventory"].get(),
        )

    def _build_npc(self) -> NPC:
        npc_type = NPCType(self.npc_type_var.get())
        return NPC(
            name=self.npc_name_var.get() or "NPC",
            npc_type=npc_type,
            hp=self.scale_vars["npc_hp"].get(),
            attack=10,
            defense=self.scale_vars["npc_def"].get(),
            dialogue=["..."],
        )

    def _build_state(self) -> dict:
        return {
            "player_attacked": self.flag_vars["player_attacked"].get(),
            "npc_met": self.flag_vars["npc_met"].get(),
            "turn_count": 1,
        }

    # recompute 

    def _on_state_change(self, *_):
        npc_type = NPCType(self.npc_type_var.get())
        self.current_tree = NPCDecisionTree.get_tree_for_npc(npc_type)
        npc = self._build_npc()
        player = self._build_player()
        gs = self._build_state()
        action, path, branches = (
            self.current_tree.evaluate_with_path(npc, player, gs))
        self.current_path = path
        self.current_branches = branches
        self.current_action = action
        self._layout_tree()
        self._draw()
        self._draw_result()

    # layout 

    def _count_leaves(self, node: DecisionNode) -> int:
        if node is None:
            return 0
        if node.is_leaf():
            return 1
        return (self._count_leaves(node.true_branch)
                + self._count_leaves(node.false_branch)) or 1

    def _layout_tree(self):
        self.node_positions.clear()
        if self.current_tree is None:
            return
        x_min, x_max = 30, self.CANVAS_W - 30
        y_top = 40
        self._layout(self.current_tree, x_min, x_max, y_top)

    def _layout(self, node, x_min, x_max, y):
        if node is None:
            return
        cx = (x_min + x_max) / 2
        self.node_positions[id(node)] = (cx, y)
        if node.is_leaf():
            return
        # Split horizontally proportional to leaf counts
        lt = self._count_leaves(node.true_branch)
        lf = self._count_leaves(node.false_branch)
        total = lt + lf or 1
        mid = x_min + (x_max - x_min) * (lt / total)
        if node.true_branch is not None:
            self._layout(node.true_branch, x_min, mid, y + self.LEVEL_H)
        if node.false_branch is not None:
            self._layout(node.false_branch, mid, x_max, y + self.LEVEL_H)

    # draw 

    def _draw(self):
        self.canvas.delete("all")
        if self.current_tree is None:
            return

        path_ids = {id(n) for n in self.current_path}

        # First: draw all edges
        self._draw_edges(self.current_tree, path_ids)
        # Then: draw all nodes
        self._draw_nodes(self.current_tree, path_ids)

    def _draw_edges(self, node, path_ids):
        if node is None or node.is_leaf():
            return
        cx, cy = self.node_positions[id(node)]
        for branch_node, branch_value in (
            (node.true_branch, True),
            (node.false_branch, False),
        ):
            if branch_node is None:
                continue
            tx, ty = self.node_positions[id(branch_node)]
            # Path highlighting: was this edge taken?
            on_path = False
            if id(node) in path_ids and id(branch_node) in path_ids:
                # Check that it's the actually-taken branch
                idx = self.current_path.index(node)
                if idx < len(self.current_branches):
                    on_path = (self.current_branches[idx] == branch_value)
            color = "#27ae60" if branch_value else "#c0392b"
            width = 4 if on_path else 1
            self.canvas.create_line(
                cx, cy + self.NODE_H / 2 - 2,
                tx, ty - self.NODE_H / 2 + 2,
                fill=color, width=width, smooth=True)
            # Label T/F near top of edge
            label_x = (cx + tx) / 2
            label_y = (cy + ty) / 2
            self.canvas.create_text(
                label_x, label_y, text=("T" if branch_value else "F"),
                fill="#fafafa" if on_path else "#888",
                font=self.bold)
            self._draw_edges(branch_node, path_ids)

    def _draw_nodes(self, node, path_ids):
        if node is None:
            return
        cx, cy = self.node_positions[id(node)]
        on_path = id(node) in path_ids
        is_chosen_leaf = (node.is_leaf()
                          and self.current_action is not None
                          and node.action == self.current_action
                          and on_path)

        x0, y0 = cx - self.NODE_W / 2, cy - self.NODE_H / 2
        x1, y1 = cx + self.NODE_W / 2, cy + self.NODE_H / 2

        if node.is_leaf():
            fill = ACTION_COLORS.get(node.action, "#444")
            outline = "#f1c40f" if is_chosen_leaf else "#333"
            border = 4 if is_chosen_leaf else 2
            self.canvas.create_rectangle(x0, y0, x1, y1,
                                         fill=fill, outline=outline,
                                         width=border)
            self.canvas.create_text(cx, cy, text=node.label,
                                    fill="#fafafa", font=self.bold)
        else:
            fill = "#1a1a1f"
            outline = "#f1c40f" if on_path else "#333"
            border = 3 if on_path else 1
            # rounded-ish (rectangle with thick border)
            self.canvas.create_rectangle(x0, y0, x1, y1,
                                         fill=fill, outline=outline,
                                         width=border)
            self.canvas.create_text(cx, cy, text=node.label,
                                    fill="#fafafa", font=self.mono,
                                    width=self.NODE_W - 8)

        if not node.is_leaf():
            self._draw_nodes(node.true_branch, path_ids)
            self._draw_nodes(node.false_branch, path_ids)

    def _draw_result(self):
        if self.current_action is None:
            self.result_label.config(text="(no result)")
            self.path_label.config(text="")
            return
        self.result_label.config(
            text=f"→ NPC chooses: {self.current_action.value.upper()}",
            fg=ACTION_COLORS.get(self.current_action, "#fafafa"))
        # Path lines
        lines = []
        for i, node in enumerate(self.current_path):
            if i < len(self.current_branches):
                tag = "T" if self.current_branches[i] else "F"
                lines.append(f"  {node.label}  → [{tag}]")
            else:
                lines.append(f"  ↳ {node.label}")
        self.path_label.config(text="\n".join(lines))


# Combat Math tab — Bayesian breakdown + live calibration plot

class CombatMathTab:
    """
    Show what the Bayesian combat system actually computes:
      - the prior P(hit), the stat-difference posterior, the d20 roll,
        the threshold check, and the outcome are all displayed numerically.
      - a running scatter plot updates with empirical hit rate per
        (attack, defense) bucket, with the perfect-calibration diagonal.
    """

    BUCKET_GRID = (range(5, 31, 5), range(0, 26, 5))  # attack, defense bins

    def __init__(self, parent: tk.Widget):
        self.parent = parent
        self.mono = tkfont.Font(family="Consolas", size=10)
        self.bold = tkfont.Font(family="Consolas", size=11, weight="bold")
        self.title_f = tkfont.Font(family="Segoe UI", size=14, weight="bold")

        self.combat = BayesianCombatSystem()

        # Per-bucket running stats: (attack, defense) -> [hits, total]
        self.bucket_hits: dict = {}
        self.last_breakdown: dict = {}
        self.total_rolls = 0
        self.total_hits = 0
        self.total_crit_hits = 0
        self.total_crit_misses = 0
        self.total_damage = 0

        self._build()
        self._refresh()

    # -- layout 

    def _build(self):
        outer = tk.Frame(self.parent, bg=GRID_BG)
        outer.pack(fill=tk.BOTH, expand=True, padx=PADDING, pady=PADDING)

        body = tk.Frame(outer, bg=GRID_BG)
        body.pack(fill=tk.BOTH, expand=True)

        # Left column: controls + roll breakdown
        left = tk.Frame(body, bg=GRID_BG)
        left.pack(side=tk.LEFT, fill=tk.Y)

        self._build_controls(left)
        self._build_breakdown(left)
        self._build_summary(left)

        # Right column: calibration plot
        right = tk.Frame(body, bg=GRID_BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(PADDING, 0))
        self._build_plot(right)

    def _section(self, parent, title):
        wrapper = tk.Frame(parent, bg=GRID_BG, pady=4)
        wrapper.pack(fill=tk.X, pady=(0, 8))
        tk.Label(wrapper, text=title, font=self.title_f,
                 fg="#e6e6e6", bg=GRID_BG, anchor="w").pack(fill=tk.X)
        body = tk.Frame(wrapper, bg="#1a1a1f", padx=10, pady=8)
        body.pack(fill=tk.X)
        return body

    def _build_controls(self, parent):
        body = self._section(parent, "Combatants")

        # Attacker
        tk.Label(body, text="Attacker", font=self.bold, bg="#1a1a1f",
                 fg="#fafafa").pack(anchor="w")
        self.atk_attack = self._slider(body, "  attack", 5, 30, 15)
        self.atk_defense = self._slider(body, "  defense", 0, 25, 10)

        # Defender
        tk.Label(body, text="Defender", font=self.bold, bg="#1a1a1f",
                 fg="#fafafa").pack(anchor="w", pady=(6, 0))
        self.def_attack = self._slider(body, "  attack", 5, 30, 10)
        self.def_defense = self._slider(body, "  defense", 0, 25, 5)

        # Buttons
        btn_row = tk.Frame(body, bg="#1a1a1f")
        btn_row.pack(fill=tk.X, pady=(8, 0))
        tk.Button(btn_row, text="Roll once",
                  command=lambda: self._do_rolls(1),
                  width=10).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_row, text="Roll 100×",
                  command=lambda: self._do_rolls(100),
                  width=10).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_row, text="Roll 1000×",
                  command=lambda: self._do_rolls(1000),
                  width=12).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_row, text="Reset",
                  command=self._reset, width=8).pack(side=tk.LEFT, padx=2)

    def _slider(self, parent, label, lo, hi, default):
        row = tk.Frame(parent, bg="#1a1a1f")
        row.pack(fill=tk.X)
        tk.Label(row, text=label, bg="#1a1a1f", fg="#cfcfcf",
                 font=self.mono, width=12, anchor="w").pack(side=tk.LEFT)
        v = tk.IntVar(value=default)
        tk.Scale(row, from_=lo, to=hi, orient=tk.HORIZONTAL,
                 variable=v, length=160, bg="#1a1a1f",
                 fg="#cfcfcf", troughcolor="#0e0e10",
                 highlightthickness=0, font=self.mono).pack(side=tk.LEFT)
        return v

    def _build_breakdown(self, parent):
        body = self._section(parent, "Last roll — Bayesian breakdown")
        self.breakdown_text = tk.Label(body, font=self.mono, bg="#1a1a1f",
                                       fg="#cfcfcf", justify="left", anchor="w",
                                       width=46)
        self.breakdown_text.pack(fill=tk.X)

    def _build_summary(self, parent):
        body = self._section(parent, "Running totals")
        self.summary_text = tk.Label(body, font=self.mono, bg="#1a1a1f",
                                     fg="#e6e6e6", justify="left", anchor="w",
                                     width=46)
        self.summary_text.pack(fill=tk.X)

    def _build_plot(self, parent):
        body = self._section(parent, "Calibration: predicted vs empirical")

        if not _HAVE_MATPLOTLIB:
            tk.Label(body, text="matplotlib not available — install matplotlib "
                                "to see the live calibration plot.",
                     bg="#1a1a1f", fg="#cfcfcf", font=self.mono,
                     wraplength=500).pack(fill=tk.X)
            self.fig = None
            self.canvas = None
            return

        self.fig = Figure(figsize=(6.6, 6.0), facecolor="#0e0e10")
        self.ax = self.fig.add_subplot(111, facecolor="#0e0e10")
        self._init_plot()
        self.canvas = FigureCanvasTkAgg(self.fig, master=body)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.draw()

    def _init_plot(self):
        self.ax.clear()
        self.ax.set_facecolor("#0e0e10")
        self.ax.plot([0, 1], [0, 1], "--", color="#666",
                     linewidth=1, label="perfect calibration")
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel("Predicted P(hit)", color="#cfcfcf")
        self.ax.set_ylabel("Empirical hit rate (per stat bucket)",
                           color="#cfcfcf")
        self.ax.set_title("Combat hit-probability calibration",
                          color="#fafafa")
        for s in self.ax.spines.values():
            s.set_color("#444")
        self.ax.tick_params(colors="#cfcfcf")
        self.ax.legend(loc="lower right", facecolor="#1a1a1f",
                       edgecolor="#333", labelcolor="#cfcfcf")

    # behaviour 

    def _do_rolls(self, n: int):
        for _ in range(n):
            self._roll_once()
        self._refresh()
        self._refresh_plot()

    def _roll_once(self):
        atk_atk = self.atk_attack.get()
        atk_def = self.atk_defense.get()
        def_atk = self.def_attack.get()
        def_def = self.def_defense.get()

        attacker = Player(name="A", hp=999, max_hp=999,
                          attack=atk_atk, defense=atk_def, position=(0, 0))
        defender = NPC(name="D", npc_type=NPCType.ENEMY,
                       hp=10**9, attack=def_atk, defense=def_def,
                       dialogue=[])

        # Compute the breakdown ourselves (mirrors the BayesianCombatSystem)
        roll = self.combat.dice.d20()
        base = self.combat.BASE_HIT_CHANCE
        stat_diff = atk_atk - def_def
        stat_mod = stat_diff * 0.03
        raw_prob = base + stat_mod
        clamped_prob = max(0.05, min(0.95, raw_prob))
        roll_q = (roll - 1) / 19.0 if roll not in (1, 20) else None

        # Resolve
        outcome, dmg = self.combat.resolve_attack(attacker, defender)
        is_hit = outcome in (CombatOutcome.HIT, CombatOutcome.CRITICAL_HIT)

        self.total_rolls += 1
        if is_hit:
            self.total_hits += 1
            self.total_damage += dmg
        if outcome == CombatOutcome.CRITICAL_HIT:
            self.total_crit_hits += 1
        if outcome == CombatOutcome.CRITICAL_MISS:
            self.total_crit_misses += 1

        bucket = (atk_atk, def_def)
        b = self.bucket_hits.setdefault(bucket,
                                        {"hits": 0, "total": 0,
                                         "predicted": clamped_prob})
        b["hits"] += int(is_hit)
        b["total"] += 1
        b["predicted"] = clamped_prob  # refresh in case stats changed

        self.last_breakdown = {
            "atk": atk_atk, "atk_def": atk_def, "def_atk": def_atk,
            "def_def": def_def,
            "roll": roll, "roll_q": roll_q,
            "base": base, "stat_diff": stat_diff, "stat_mod": stat_mod,
            "raw_prob": raw_prob, "clamped_prob": clamped_prob,
            "outcome": outcome, "dmg": dmg,
        }

    def _reset(self):
        self.bucket_hits.clear()
        self.total_rolls = 0
        self.total_hits = 0
        self.total_crit_hits = 0
        self.total_crit_misses = 0
        self.total_damage = 0
        self.last_breakdown = {}
        self._refresh()
        self._refresh_plot()

    # render 

    def _refresh(self):
        b = self.last_breakdown
        if b:
            outcome = b["outcome"]
            note = ""
            if b["roll"] == 20:
                note = "  [natural 20 — auto crit]"
            elif b["roll"] == 1:
                note = "  [natural 1 — auto fumble]"
            else:
                note = (f"  roll quantile = ({b['roll']}-1)/19 = "
                        f"{b['roll_q']:.3f}")

            text = (
                f"P(hit | atk={b['atk']}, def={b['def_def']})\n"
                f"  prior P(hit)        = {b['base']:.2f}\n"
                f"  stat_diff           = {b['atk']} - {b['def_def']} "
                f"= {b['stat_diff']}\n"
                f"  posterior modifier  = stat_diff × 0.03 "
                f"= {b['stat_mod']:+.3f}\n"
                f"  raw P(hit)          = {b['raw_prob']:.3f}\n"
                f"  clamped to [0.05, 0.95] = {b['clamped_prob']:.3f}\n"
                f"\n"
                f"d20 roll              = {b['roll']}{note}\n"
                f"\n"
                f"=> outcome: {outcome.value}   damage: {b['dmg']}"
            )
        else:
            text = "(no rolls yet — click Roll once / Roll 100× / Roll 1000×)"
        self.breakdown_text.config(text=text)

        if self.total_rolls:
            emp = self.total_hits / self.total_rolls
        else:
            emp = 0.0
        avg_dmg = (self.total_damage / max(1, self.total_hits))
        n_buckets = sum(1 for b in self.bucket_hits.values() if b["total"] >= 5)
        # Compute mean |empirical - predicted| over buckets with enough samples
        if n_buckets > 0:
            errs = [abs(b["hits"]/b["total"] - b["predicted"])
                    for b in self.bucket_hits.values() if b["total"] >= 5]
            mae = sum(errs) / len(errs)
        else:
            mae = float("nan")

        summary = (
            f"Total rolls           : {self.total_rolls}\n"
            f"Hits                  : {self.total_hits}\n"
            f"Crits                 : {self.total_crit_hits}\n"
            f"Critical fumbles      : {self.total_crit_misses}\n"
            f"Empirical hit rate    : {emp:.3f}\n"
            f"Avg damage per hit    : {avg_dmg:.1f}\n"
            f"Stat buckets with ≥5  : {n_buckets}\n"
            f"Mean |emp - pred|     : {mae:.3f}"
        )
        self.summary_text.config(text=summary)

    def _refresh_plot(self):
        if not _HAVE_MATPLOTLIB or self.canvas is None:
            return
        self._init_plot()
        # Plot one point per bucket; size = sqrt(samples) for visibility
        xs, ys, sizes = [], [], []
        for bucket, info in self.bucket_hits.items():
            if info["total"] < 1:
                continue
            xs.append(info["predicted"])
            ys.append(info["hits"] / info["total"])
            sizes.append(min(220, 20 + info["total"] * 1.5))
        if xs:
            self.ax.scatter(xs, ys, s=sizes, alpha=0.7,
                            color="#5b8def", edgecolor="white",
                            linewidth=0.8, zorder=3,
                            label=f"{len(xs)} stat buckets")
            self.ax.legend(loc="lower right", facecolor="#1a1a1f",
                           edgecolor="#333", labelcolor="#cfcfcf")
        self.canvas.draw()


# Algorithm Comparison tab — runs the dungeon-generation sweep on demand
class AlgorithmComparisonTab:
    """
    A 'research mode' tab. The user picks a configuration grid and a
    seed count; the tab runs the same dungeon-eval sweep used offline,
    then renders the comparison results inline (table + bar chart).

    Single-threaded so the UI freezes briefly during long runs — the
    button label flips to "running..." for feedback.
    """

    def __init__(self, parent: tk.Widget):
        self.parent = parent
        self.mono = tkfont.Font(family="Consolas", size=10)
        self.bold = tkfont.Font(family="Consolas", size=11, weight="bold")
        self.title_f = tkfont.Font(family="Segoe UI", size=14, weight="bold")
        self.last_summary: list = []
        self._build()

    def _build(self):
        outer = tk.Frame(self.parent, bg=GRID_BG)
        outer.pack(fill=tk.BOTH, expand=True, padx=PADDING, pady=PADDING)

        # Controls row
        controls = tk.Frame(outer, bg=GRID_BG)
        controls.pack(fill=tk.X, pady=(0, 8))
        for label, var_name, default, lo, hi in [
            ("Grid", "grid_var", 8, 4, 14),
            ("Rooms", "rooms_var", 14, 3, 80),
            ("Seeds", "seeds_var", 10, 1, 100),
        ]:
            tk.Label(controls, text=f"{label}:", bg=GRID_BG, fg="#e6e6e6",
                     font=self.mono).pack(side=tk.LEFT, padx=(0, 4))
            v = tk.IntVar(value=default)
            setattr(self, var_name, v)
            tk.Spinbox(controls, from_=lo, to=hi, width=5, textvariable=v,
                       font=self.mono).pack(side=tk.LEFT, padx=(0, 12))

        self.run_btn = tk.Button(controls, text="Run sweep",
                                 command=self._run_sweep, width=14)
        self.run_btn.pack(side=tk.LEFT, padx=4)
        self.status_label = tk.Label(controls, text="(ready)",
                                     bg=GRID_BG, fg="#cfcfcf",
                                     font=self.mono)
        self.status_label.pack(side=tk.LEFT, padx=(8, 0))

        body = tk.Frame(outer, bg=GRID_BG)
        body.pack(fill=tk.BOTH, expand=True)

        # Results table on left
        left = tk.Frame(body, bg=GRID_BG)
        left.pack(side=tk.LEFT, fill=tk.Y)
        self._build_results_table(left)

        # Bar chart on right
        right = tk.Frame(body, bg=GRID_BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(PADDING, 0))
        self._build_plot(right)

    def _section(self, parent, title):
        wrapper = tk.Frame(parent, bg=GRID_BG, pady=4)
        wrapper.pack(fill=tk.X, pady=(0, 8))
        tk.Label(wrapper, text=title, font=self.title_f,
                 fg="#e6e6e6", bg=GRID_BG, anchor="w").pack(fill=tk.X)
        body = tk.Frame(wrapper, bg="#1a1a1f", padx=10, pady=8)
        body.pack(fill=tk.X)
        return body

    def _build_results_table(self, parent):
        body = self._section(parent, "Results table")
        cols = ("algo", "succ", "solv", "time_ms", "nodes", "bt",
                "branch", "deadends", "boss_dist")
        self.table = ttk.Treeview(body, columns=cols, show="headings",
                                  height=14)
        for c, w, lbl in [
            ("algo", 70, "Algo"),
            ("succ", 60, "Succ"),
            ("solv", 60, "Solv"),
            ("time_ms", 80, "Time(ms)"),
            ("nodes", 70, "Nodes"),
            ("bt", 60, "Backtr"),
            ("branch", 70, "Branch"),
            ("deadends", 90, "DeadEnd"),
            ("boss_dist", 90, "BossDist"),
        ]:
            self.table.heading(c, text=lbl)
            self.table.column(c, width=w, anchor="center")
        self.table.pack(fill=tk.BOTH, expand=True)

    def _build_plot(self, parent):
        body = self._section(parent, "Solvability + structural metrics")
        if not _HAVE_MATPLOTLIB:
            tk.Label(body, text="matplotlib not available — install matplotlib "
                                "to see the inline charts.",
                     bg="#1a1a1f", fg="#cfcfcf", font=self.mono,
                     wraplength=500).pack(fill=tk.X)
            self.fig = None
            self.canvas = None
            return
        self.fig = Figure(figsize=(6.6, 5.0), facecolor="#0e0e10")
        self.canvas = FigureCanvasTkAgg(self.fig, master=body)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.draw()

    # run the sweep 

    def _run_sweep(self):
        grid = self.grid_var.get()
        rooms = self.rooms_var.get()
        n_seeds = self.seeds_var.get()
        self.run_btn.config(state=tk.DISABLED, text="Running…")
        self.status_label.config(text=f"running {4 * n_seeds} configs…")
        self.parent.update_idletasks()
        try:
            summary = self._run_compare(grid, rooms, n_seeds)
            self.last_summary = summary
            self._populate_table(summary)
            self._populate_chart(summary)
            self.status_label.config(
                text=f"done — {4 * n_seeds} runs total",
                fg="#27ae60")
        except Exception as e:
            self.status_label.config(text=f"error: {e}", fg="#c0392b")
        finally:
            self.run_btn.config(state=tk.NORMAL, text="Run sweep")

    def _run_compare(self, grid: int, rooms: int, n_seeds: int) -> list:
        """Mirrors evaluation/dungeon_eval.py for one config across seeds."""
        import io
        import time
        from contextlib import redirect_stdout
        from evaluation.quality_metrics import metric_bundle

        algos = {
            "CSP":    DungeonCSP,
            "BFS":    BFSDungeonGenerator,
            "DFS":    DFSDungeonGenerator,
            "Greedy": GreedyDungeonGenerator,
        }
        per_algo = {k: [] for k in algos}
        for algo, cls in algos.items():
            for seed in range(n_seeds):
                gen = cls(grid, grid, rooms, seed=seed)
                buf = io.StringIO()
                t0 = time.perf_counter()
                try:
                    with redirect_stdout(buf):
                        d = gen.generate()
                except Exception:
                    d = None
                elapsed = time.perf_counter() - t0
                row = {
                    "time_s": elapsed,
                    "nodes": gen.nodes_explored,
                    "backtracks": gen.backtrack_count,
                    "succeeded": int(d is not None),
                    **metric_bundle(d),
                }
                per_algo[algo].append(row)

        summary = []
        for algo, runs in per_algo.items():
            ok = [r for r in runs if r["succeeded"]]
            mean = lambda key, src=runs: (sum(r[key] for r in src) / len(src)) if src else 0
            summary.append({
                "algo": algo,
                "success_rate": mean("succeeded"),
                "solvable_rate": mean("solvable"),
                "mean_time_ms": mean("time_s") * 1000,
                "mean_nodes": mean("nodes"),
                "mean_backtracks": mean("backtracks"),
                "mean_branching": (sum(r["branching"] for r in ok) / len(ok)) if ok else 0,
                "mean_dead_end": (sum(r["dead_end_ratio"] for r in ok) / len(ok)) if ok else 0,
                "mean_boss_dist": (sum(r["start_boss_dist"] for r in ok) / len(ok)) if ok else 0,
            })
        return summary

    def _populate_table(self, summary):
        for row in self.table.get_children():
            self.table.delete(row)
        for r in summary:
            self.table.insert("", tk.END, values=(
                r["algo"],
                f"{r['success_rate']:.0%}",
                f"{r['solvable_rate']:.0%}",
                f"{r['mean_time_ms']:.2f}",
                f"{r['mean_nodes']:.1f}",
                f"{r['mean_backtracks']:.1f}",
                f"{r['mean_branching']:.2f}",
                f"{r['mean_dead_end']:.2f}",
                f"{r['mean_boss_dist']:.1f}",
            ))

    def _populate_chart(self, summary):
        if not _HAVE_MATPLOTLIB or self.canvas is None:
            return
        self.fig.clear()
        ax1 = self.fig.add_subplot(2, 1, 1, facecolor="#0e0e10")
        ax2 = self.fig.add_subplot(2, 1, 2, facecolor="#0e0e10")
        algos = [r["algo"] for r in summary]
        solv = [r["solvable_rate"] for r in summary]
        branch = [r["mean_branching"] for r in summary]
        dead = [r["mean_dead_end"] for r in summary]

        bars = ax1.bar(algos, solv,
                       color=["#5b8def", "#e67e22", "#27ae60", "#9b59b6"])
        ax1.set_ylim(0, 1.05)
        ax1.set_title("Solvability rate by algorithm", color="#fafafa")
        ax1.tick_params(colors="#cfcfcf")
        for bar, v in zip(bars, solv):
            ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                     f"{v:.0%}", ha="center", color="#cfcfcf")
        for s in ax1.spines.values():
            s.set_color("#444")

        x = list(range(len(algos)))
        ax2.bar([xi - 0.2 for xi in x], branch, width=0.4,
                color="#27ae60", label="branching factor")
        ax2.bar([xi + 0.2 for xi in x], dead, width=0.4,
                color="#c0392b", label="dead-end ratio")
        ax2.set_xticks(x); ax2.set_xticklabels(algos)
        ax2.set_title("Structural metrics", color="#fafafa")
        ax2.legend(facecolor="#1a1a1f", edgecolor="#333", labelcolor="#cfcfcf")
        ax2.tick_params(colors="#cfcfcf")
        for s in ax2.spines.values():
            s.set_color("#444")

        self.fig.tight_layout()
        self.canvas.draw()

# Play tab
class PlayTab:
    """The playable D&D layer (decision tree NPCs + Bayesian combat + skill checks)."""

    def __init__(self, parent: tk.Widget, state: GameState, root: tk.Tk):
        self.parent = parent
        self.state = state
        self.root = root
        self.mono = tkfont.Font(family="Consolas", size=10)
        self.bold = tkfont.Font(family="Consolas", size=11, weight="bold")
        self.title_f = tkfont.Font(family="Segoe UI", size=14, weight="bold")
        self._build()
        self._bind_keys()
        self.refresh()

    def _build(self):
        main = tk.Frame(self.parent, bg=GRID_BG)
        main.pack(fill=tk.BOTH, expand=True, padx=PADDING, pady=PADDING)

        top = tk.Frame(main, bg=GRID_BG)
        top.pack(fill=tk.BOTH, expand=True)

        d = self.state.dungeon
        canvas_w = d.width * CELL_PX + 2 * PADDING
        canvas_h = d.height * CELL_PX + 2 * PADDING
        self.canvas = tk.Canvas(top, width=canvas_w, height=canvas_h,
                                bg=GRID_BG, highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, anchor=tk.N)

        right = tk.Frame(top, bg=GRID_BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(PADDING, 0))

        self._build_stats_panel(right)
        self._build_room_panel(right)
        self._build_inventory_panel(right)

        self._build_action_bar(main)
        self._build_event_log(main)

    def _section(self, parent: tk.Widget, title: str) -> tk.Frame:
        wrapper = tk.Frame(parent, bg=GRID_BG, pady=4)
        wrapper.pack(fill=tk.X, pady=(0, 8))
        tk.Label(wrapper, text=title, font=self.title_f,
                 fg="#e6e6e6", bg=GRID_BG, anchor="w").pack(fill=tk.X)
        body = tk.Frame(wrapper, bg="#1a1a1f", padx=10, pady=8)
        body.pack(fill=tk.X)
        return body

    def _build_stats_panel(self, parent):
        body = self._section(parent, "Player")
        self.stat_name = tk.Label(body, font=self.bold, bg="#1a1a1f", fg="#fafafa")
        self.stat_name.pack(anchor="w")
        self.hp_bar = ttk.Progressbar(body, length=240, maximum=100)
        self.hp_bar.pack(fill=tk.X, pady=(4, 2))
        self.hp_text = tk.Label(body, font=self.mono, bg="#1a1a1f", fg="#cfcfcf")
        self.hp_text.pack(anchor="w")
        self.stat_text = tk.Label(body, font=self.mono, bg="#1a1a1f",
                                  fg="#cfcfcf", justify="left")
        self.stat_text.pack(anchor="w")

    def _build_room_panel(self, parent):
        body = self._section(parent, "Current room")
        self.room_text = tk.Label(body, font=self.mono, bg="#1a1a1f",
                                  fg="#cfcfcf", justify="left", wraplength=300,
                                  anchor="w")
        self.room_text.pack(fill=tk.X, anchor="w")
        self.room_items_frame = tk.Frame(body, bg="#1a1a1f")
        self.room_items_frame.pack(fill=tk.X, pady=(6, 0))

    def _build_inventory_panel(self, parent):
        body = self._section(parent, "Inventory")
        self.inv_listbox = tk.Listbox(
            body, height=6, font=self.mono,
            bg="#0e0e10", fg="#e6e6e6",
            selectbackground="#5b8def", borderwidth=0,
            activestyle="none")
        self.inv_listbox.pack(fill=tk.X)
        btns = tk.Frame(body, bg="#1a1a1f")
        btns.pack(fill=tk.X, pady=(6, 0))
        tk.Button(btns, text="Use selected", command=self._on_use_item,
                  width=14).pack(side=tk.LEFT, padx=(0, 6))

    def _build_action_bar(self, parent):
        bar = tk.Frame(parent, bg=GRID_BG, pady=8)
        bar.pack(fill=tk.X)

        self.move_buttons = {}
        for label in ("North", "South", "East", "West"):
            b = tk.Button(bar, text=label, width=8,
                          command=lambda l=label: self._on_move(l))
            b.pack(side=tk.LEFT, padx=2)
            self.move_buttons[label] = b
        tk.Frame(bar, width=20, bg=GRID_BG).pack(side=tk.LEFT)

        self.btn_attack = tk.Button(bar, text="Attack", width=8,
                                    command=self._on_attack)
        self.btn_attack.pack(side=tk.LEFT, padx=2)
        self.btn_flee = tk.Button(bar, text="Flee", width=8, command=self._on_flee)
        self.btn_flee.pack(side=tk.LEFT, padx=2)
        self.btn_talk = tk.Button(bar, text="Talk / Trade", width=12,
                                  command=self._on_talk)
        self.btn_talk.pack(side=tk.LEFT, padx=2)

        tk.Frame(bar, width=20, bg=GRID_BG).pack(side=tk.LEFT)
        self.btn_insight = tk.Button(bar, text="Insight (WIS)", width=12,
                                     command=self._on_insight)
        self.btn_insight.pack(side=tk.LEFT, padx=2)
        self.btn_persuade = tk.Button(bar, text="Persuade (CHA)", width=14,
                                      command=self._on_persuade)
        self.btn_persuade.pack(side=tk.LEFT, padx=2)
        self.btn_force = tk.Button(bar, text="Force door (STR)", width=16,
                                   command=self._on_force_door)
        self.btn_force.pack(side=tk.LEFT, padx=2)

    def _build_event_log(self, parent):
        body = self._section(parent, "Event log")
        self.log_box = tk.Text(body, height=10, font=self.mono,
                               bg="#0e0e10", fg="#cfcfcf",
                               borderwidth=0, wrap=tk.WORD)
        self.log_box.pack(fill=tk.BOTH, expand=True)
        self.log_box.config(state=tk.DISABLED)

    # handlers 

    def _on_move(self, direction):
        x, y = self.state.player.position
        target = {"North": (x, y - 1), "South": (x, y + 1),
                  "East": (x + 1, y), "West": (x - 1, y)}[direction]
        self.state.move(target)
        self.refresh()
        self._maybe_endgame()

    def _on_attack(self):
        self.state.attack(); self.refresh(); self._maybe_endgame()

    def _on_select_target(self, idx: int):
        """Switch the current combat target."""
        enemies = self.state.status.combat_enemies
        if 0 <= idx < len(enemies):
            self.state.status.target_idx = idx
            self.state.log(f"Targeting {enemies[idx].name}…")
        else:
            self.state.status.target_idx = 0
        self.refresh()

    def _on_flee(self):
        self.state.flee(); self.refresh(); self._maybe_endgame()

    def _on_talk(self):
        self.state.talk(); self.refresh()

    def _on_insight(self):
        self.state.insight_check(); self.refresh()

    def _on_persuade(self):
        self.state.persuade(); self.refresh(); self._maybe_endgame()

    def _on_force_door(self):
        self.state.force_boss_door(); self.refresh(); self._maybe_endgame()

    def _on_pickup(self, idx):
        self.state.pick_up(idx); self.refresh()

    def _on_use_item(self):
        sel = self.inv_listbox.curselection()
        if sel:
            self.state.use_item(sel[0])
            self.refresh()
            self._maybe_endgame()

    def _bind_keys(self):
        """Bind WASD/arrow keys and action hotkeys to the root window."""
        move_keys = ['w', 'a', 's', 'd', 'W', 'A', 'S', 'D',
                     'Up', 'Down', 'Left', 'Right']
        action_keys = ['e', 'E', 'f', 'F', 'q', 'Q']
        for key in move_keys + action_keys:
            ev = f"<{key}>" if len(key) > 1 else key
            self.root.bind(ev, self._handle_keypress)

    def _handle_keypress(self, event: tk.Event):
        key = event.keysym.lower() if len(event.keysym) == 1 else event.keysym
        
        # 1. Handle Target Cycling if in combat
        in_combat = self.state.status.in_combat
        enemies = self.state.status.combat_enemies
        
        if in_combat and len(enemies) > 1:
            current_idx = self.state.status.target_idx
            # Cycle Up (W or Up Arrow)
            if key in ('w', 'Up'):
                new_idx = (current_idx - 1) % len(enemies)
                self._on_select_target(new_idx)
                return # Intercept the key so it doesn't trigger movement
            # Cycle Down (S or Down Arrow)
            elif key in ('s', 'Down'):
                new_idx = (current_idx + 1) % len(enemies)
                self._on_select_target(new_idx)
                return # Intercept the key so it doesn't trigger movement

        # 2. Existing Movement Logic
        move_map = {
            'w': 'North', 'Up': 'North',
            's': 'South', 'Down': 'South',
            'a': 'West',  'Left': 'West',
            'd': 'East',  'Right': 'East',
        }
        
        if key in move_map:
            direction = move_map[key]
            # Movement is already gated by button state in your logic
            if self.move_buttons[direction]['state'] == tk.NORMAL:
                self._on_move(direction)
        
        # 3. Action Hotkeys
        elif key == 'e':
            if self.btn_attack['state'] == tk.NORMAL:
                self._on_attack()
        elif key == 'f':
            if self.btn_flee['state'] == tk.NORMAL:
                self._on_flee()
        elif key == 'q':
            if self.btn_talk['state'] == tk.NORMAL:
                self._on_talk()

    def _maybe_endgame(self):
        if self.state.status.game_over:
            self.refresh()
            if self.state.status.victory:
                messagebox.showinfo(
                    "Victory!",
                    f"You defeated the boss in {self.state.turn} turns.\n"
                    f"Final HP: {self.state.player.hp}/{self.state.player.max_hp}\n"
                    f"Gold: {self.state.player.gold}")
            else:
                messagebox.showwarning(
                    "Defeated",
                    f"You fell after {self.state.turn} turns.\n"
                    f"Gold collected: {self.state.player.gold}")

    # repaint 

    def refresh(self):
        self._draw_map()
        self._draw_stats()
        self._draw_room()
        self._draw_inventory()
        self._draw_log()
        self._update_button_state()

    def _draw_map(self):
        self.canvas.delete("all")
        d = self.state.dungeon
        for coords, room in d.rooms.items():
            if not room.visited: continue
            cx1, cy1 = self._cell_center(coords)
            for nb in room.connections:
                if nb not in d.rooms: continue
                if not d.rooms[nb].visited: continue
                cx2, cy2 = self._cell_center(nb)
                self.canvas.create_line(cx1, cy1, cx2, cy2,
                                        fill=CONNECTION_COLOR, width=3)
        for coords, room in d.rooms.items():
            x, y = coords
            x0 = PADDING + x * CELL_PX + 6
            y0 = PADDING + y * CELL_PX + 6
            x1 = x0 + CELL_PX - 12
            y1 = y0 + CELL_PX - 12
            if not room.visited:
                fill = FOG_COLOR; outline = "#222"
            else:
                fill = ROOM_COLORS.get(room.room_type, UNVISITED_COLOR)
                outline = "#333"
            if (room.room_type == RoomType.BOSS
                    and not self.state.status.boss_unlocked
                    and room.visited):
                outline = "#f39c12"
            self.canvas.create_rectangle(x0, y0, x1, y1,
                                         fill=fill, outline=outline, width=2)
            if room.visited:
                glyph = {
                    RoomType.START: "S", RoomType.BOSS: "B",
                    RoomType.TREASURE: "T", RoomType.MERCHANT: "M",
                    RoomType.TRAP: "X", RoomType.NORMAL: "·",
                }.get(room.room_type, "?")
                self.canvas.create_text((x0 + x1) / 2, (y0 + y1) / 2,
                                        text=glyph, fill="#0a0a0a",
                                        font=self.bold)
        cx, cy = self._cell_center(self.state.player.position)
        r = 10
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                fill=PLAYER_COLOR, outline="#000", width=2)

    def _cell_center(self, coords):
        x, y = coords
        return (PADDING + x * CELL_PX + CELL_PX / 2,
                PADDING + y * CELL_PX + CELL_PX / 2)

    def _draw_stats(self):
        p = self.state.player
        self.stat_name.config(text=f"{p.name}")
        pct = (p.hp / p.max_hp) * 100 if p.max_hp else 0
        self.hp_bar["value"] = pct
        self.hp_text.config(text=f"HP {p.hp}/{p.max_hp}")
        keystr = "✓ Boss Key" if self.state.has_key() else "no key"
        self.stat_text.config(
            text=(f"ATK {p.attack}    DEF {p.defense}\n"
                  f"Gold {p.gold}    Pos {p.position}\n"
                  f"Turn {self.state.turn}    {keystr}\n"
                  f"\nAbility scores (used for skill checks):\n"
                  f"  STR {p.strength:2}   DEX {p.dexterity:2}   "
                  f"CON {p.constitution:2}\n"
                  f"  INT {p.intelligence:2}   WIS {p.wisdom:2}   "
                  f"CHA {p.charisma:2}"))

    def _draw_room(self):
        room = self.state.current_room()
        lines = [f"Type: {room.room_type.value}",
                 f"Connections: {sorted(room.connections)}"]
        if room.npcs:
            lines.append("NPCs:")
            for n in room.npcs:
                tag = "(hostile)" if n.npc_type in (NPCType.ENEMY, NPCType.BOSS) else ""
                lines.append(f"  - {n.name} HP {n.hp} {tag}")
        else:
            lines.append("No NPCs.")
        if self.state.status.in_combat and self.state.status.combat_enemies:
            names = ", ".join(e.name for e in self.state.status.combat_enemies)
            lines.append(f"** IN COMBAT vs: {names} **")
            idx = self.state.status.target_idx
            if 0 <= idx < len(self.state.status.combat_enemies):
                t = self.state.status.combat_enemies[idx]
                lines.append(f"   Current target: {t.name} (HP {t.hp})")
        self.room_text.config(text="\n".join(lines))

        for w in self.room_items_frame.winfo_children():
            w.destroy()

        # Target selector buttons (shown only during multi-enemy combat)
        if (self.state.status.in_combat
                and len(self.state.status.combat_enemies) > 1):
            tk.Label(self.room_items_frame, text="Select target:",
                     font=self.mono, bg="#1a1a1f", fg="#cfcfcf").pack(anchor="w")
            for i, enemy in enumerate(self.state.status.combat_enemies):
                is_target = (i == self.state.status.target_idx)
                tk.Button(
                    self.room_items_frame,
                    text=f"{'▶ ' if is_target else '   '}{enemy.name} (HP {enemy.hp})",
                    bg="#c0392b" if is_target else "#444",
                    fg="white",
                    command=lambda idx=i: self._on_select_target(idx),
                ).pack(anchor="w", pady=1)

        if room.items:
            tk.Label(self.room_items_frame, text="On the floor:",
                     font=self.mono, bg="#1a1a1f",
                     fg="#cfcfcf").pack(anchor="w")
            for i, item in enumerate(list(room.items)):
                tk.Button(self.room_items_frame,
                          text=f"Pick up: {item.name}",
                          command=lambda idx=i: self._on_pickup(idx)
                          ).pack(anchor="w", pady=1)

    def _draw_inventory(self):
        self.inv_listbox.delete(0, tk.END)
        for item in self.state.player.inventory:
            label = f"{item.name}  [{item.item_type.value}]"
            if item.item_type == ItemType.WEAPON:
                label += f"  +{item.properties.get('damage', 0)} ATK"
            elif item.item_type == ItemType.POTION:
                label += f"  heals {item.properties.get('heal', 0)}"
            self.inv_listbox.insert(tk.END, label)

    def _draw_log(self):
        self.log_box.config(state=tk.NORMAL)
        self.log_box.delete("1.0", tk.END)
        self.log_box.insert(tk.END, "\n".join(self.state.event_log))
        self.log_box.see(tk.END)
        self.log_box.config(state=tk.DISABLED)

    def _update_button_state(self):
        x, y = self.state.player.position
        targets = {"North": (x, y - 1), "South": (x, y + 1),
                   "East": (x + 1, y), "West": (x - 1, y)}
        room = self.state.current_room()
        in_combat = self.state.status.in_combat
        game_over = self.state.status.game_over
        for label, target in targets.items():
            enabled = (target in room.connections
                       and not in_combat and not game_over)
            self.move_buttons[label].config(
                state=tk.NORMAL if enabled else tk.DISABLED)

        self.btn_attack.config(state=tk.NORMAL if in_combat and not game_over else tk.DISABLED)
        self.btn_flee.config(state=tk.NORMAL if in_combat and not game_over else tk.DISABLED)
        has_friendly = any(n.npc_type not in (NPCType.ENEMY, NPCType.BOSS)
                           for n in room.npcs)
        self.btn_talk.config(
            state=tk.NORMAL if has_friendly and not in_combat and not game_over else tk.DISABLED)
        any_npc = any(n.hp > 0 for n in room.npcs)
        self.btn_insight.config(
            state=tk.NORMAL if any_npc and not game_over else tk.DISABLED)
        self.btn_persuade.config(
            state=tk.NORMAL if in_combat and not game_over else tk.DISABLED)
        self.btn_force.config(
            state=tk.NORMAL if self.state.can_force_boss_door() else tk.DISABLED)

class CharacterCreationDialog(tk.Toplevel):
    """Modal window for creating the player character."""

    def __init__(self, master, initial_name="Hero", player_class="Warrior",
                 grid_size=8, num_rooms=12, seed=None):
        super().__init__(master)
        self.title("Character Creation")
        self.resizable(False, False)
        self.transient(master)
        self.grab_set()
        self.configure(bg=GRID_BG) 

        self.player_class = player_class
        self.grid_size = grid_size
        self.num_rooms = num_rooms
        self.current_seed = seed if seed is not None else random.randint(0, 999999)

        self.result = None   # will hold (name, class, seed) on confirm
        self.initial_name = initial_name   # store for later us
        self.mono = tkfont.Font(family="Consolas", size=10)
        self.bold = tkfont.Font(family="Consolas", size=11, weight="bold")

        self._build_ui()
        self._refresh_preview()
        self.update_idletasks()
        self.geometry("800x600")           # set a reasonable size
        self.geometry("800x600")
        # Dark ttk style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TCombobox",
                        fieldbackground="#0e0e10",
                        background="#1a1a1f",
                        foreground="#fafafa",
                        arrowcolor="#fafafa",
                        selectbackground="#5b8def",
                        selectforeground="#0a0a0a")
        style.map("TCombobox",
                fieldbackground=[("readonly", "#0e0e10")],
                foreground=[("readonly", "#fafafa")])

        self.protocol("WM_DELETE_WINDOW", self._on_confirm)  # treat close as confirm

    def _build_ui(self):
        info = (
            "Escape the dungeon by defeating the boss.\n"
            "You'll need to find the boss key first, then force open the boss door.\n"
            "Enter your name, choose a class, view your stats, and click Start Adventure."
        )
        tk.Label(
            self, text=info, font=self.mono, justify="left",
            bg="#1a1a1f", fg="#cfcfcf", wraplength=420
        ).pack(padx=20, pady=(20, 10))

        # Name
        name_frame = tk.Frame(self, bg="#1a1a1f")
        name_frame.pack(padx=20, pady=5, fill=tk.X)
        tk.Label(name_frame, text="Name:", font=self.mono,
                bg="#1a1a1f", fg="#cfcfcf").pack(side=tk.LEFT)
        self.name_var = tk.StringVar(value=self.initial_name)
        # Name entry
        name_entry = tk.Entry(
            name_frame, textvariable=self.name_var,
            font=self.mono, width=20,
            bg="#0e0e10", fg="#fafafa",
            insertbackground="#fafafa",
            relief="flat", borderwidth=2,
            highlightbackground="#333", highlightcolor="#5b8def"
        )
        name_entry.pack(side=tk.LEFT, padx=(6, 0))
        name_entry.focus_set()

        # Class picker
        class_frame = tk.Frame(self, bg="#1a1a1f")
        class_frame.pack(padx=20, pady=5, fill=tk.X)
        tk.Label(class_frame, text="Class:", font=self.mono,
                bg="#1a1a1f", fg="#cfcfcf").pack(side=tk.LEFT)
        self.class_var = tk.StringVar(value=self.player_class)
       # Class combobox
        class_combo = ttk.Combobox(
            class_frame, textvariable=self.class_var,
            values=["Warrior", "Rogue", "Cleric", "Mage"],
            state="readonly", width=12, font=self.mono
        )
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TCombobox",
                        fieldbackground="#0e0e10",
                        background="#1a1a1f",
                        foreground="#fafafa",
                        arrowcolor="#fafafa",
                        selectbackground="#5b8def",
                        selectforeground="#0a0a0a")
        style.map("TCombobox",
                fieldbackground=[("readonly", "#0e0e10")],
                foreground=[("readonly", "#fafafa")])
        class_combo.pack(side=tk.LEFT, padx=(6, 0))
        class_combo.bind("<<ComboboxSelected>>", lambda e: self._on_class_changed())

        # Stats display
        stats_frame = tk.Frame(self, bg="#1a1a1f", borderwidth=1,
                            relief="solid", bd=1, padx=12, pady=10)
        stats_frame.pack(padx=20, pady=10, fill=tk.X)
        self.stats_label = tk.Label(stats_frame, font=self.mono,
                                    bg="#1a1a1f", fg="#e6e6e6",
                                    justify="left")
        self.stats_label.pack()

        # Buttons
        btn_frame = tk.Frame(self, bg="#1a1a1f")
        btn_frame.pack(pady=(0, 20))
        tk.Button(btn_frame, text="Re‑roll stats", command=self._on_reroll,
                width=14).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Start Adventure", command=self._on_confirm,
                bg="#27ae60", fg="white", font=self.bold,
                width=16).pack(side=tk.LEFT, padx=5)

    def _on_class_changed(self):
        self.player_class = self.class_var.get()
        self._refresh_preview()

    def _create_preview_game(self, seed):
        """Create a minimal GameState just to extract the rolled player stats."""
        # Use smallest possible dungeon so it's fast
        return GameState(
            grid_size=3, num_rooms=2,
            player_name=self.name_var.get() or "Hero",
            player_class=self.player_class,
            seed=seed
        )

    def _refresh_preview(self):
        try:
            temp = self._create_preview_game(self.current_seed)
            p = temp.player
            text = (
                f"Class: {self.player_class}\n"
                f"HP:    {p.hp} / {p.max_hp}\n"
                f"ATK:   {p.attack}   DEF:   {p.defense}\n"
                f"STR:   {p.strength:2}   DEX:   {p.dexterity:2}   CON:   {p.constitution:2}\n"
                f"INT:   {p.intelligence:2}   WIS:   {p.wisdom:2}   CHA:   {p.charisma:2}\n"
                f"Gold:  {p.gold}"
            )
            temp = None
        except Exception as e:
            text = f"(Could not preview stats: {e})"
        self.stats_label.config(text=text)

    def _on_reroll(self):
        self.current_seed = random.randint(0, 999999)
        self._refresh_preview()

    def _on_confirm(self):
        self.result = (
            self.name_var.get().strip() or "Hero",
            self.class_var.get(),          # <-- use the combobox value
            self.current_seed
        )
        self.destroy()


# Application shell
class App:
    def __init__(self, root: tk.Tk, state: Optional[GameState] = None,
                 grid_size: int = 8, num_rooms: int = 12, seed: int = None,
                 player_name: str = "Hero", player_class: str = "Warrior"):
        self.root = root
        root.title("AI Dungeon Master — algorithm inspector & playable demo")
        root.configure(bg=GRID_BG)

        self.player_name = player_name
        self.player_class = player_class
        self.grid_size = grid_size
        self.num_rooms = num_rooms
        self.seed = seed

        self.nb = nb = ttk.Notebook(root)          # store as self.nb
        nb.pack(fill=tk.BOTH, expand=True)

        inspector_frame = tk.Frame(nb, bg=GRID_BG)
        nb.add(inspector_frame, text="Generation Inspector")
        self.inspector = GenerationInspectorTab(inspector_frame)

        brain_frame = tk.Frame(nb, bg=GRID_BG)
        nb.add(brain_frame, text="NPC Brain")
        self.brain = NPCBrainTab(brain_frame)

        combat_frame = tk.Frame(nb, bg=GRID_BG)
        nb.add(combat_frame, text="Combat Math")
        self.combat_tab = CombatMathTab(combat_frame)

        compare_frame = tk.Frame(nb, bg=GRID_BG)
        nb.add(compare_frame, text="Algorithm Comparison")
        self.compare_tab = AlgorithmComparisonTab(compare_frame)

        # Play tab frame – filled after character creation if needed
        self.play_frame = tk.Frame(nb, bg=GRID_BG)
        nb.add(self.play_frame, text="Play")
        self.play = None

        self._bind_tab_keys()

        self._char_creation_done = False
        if state is not None:
            self.play = PlayTab(self.play_frame, state, root)
            self._char_creation_done = True
        else:
            self.nb.bind("<<NotebookTabChanged>>", self._on_tab_changed)
            # initial tab is index 0 (Generation Inspector), so no popup yet

    def _show_character_creation(self):
        dlg = CharacterCreationDialog(
            self.root,
            initial_name=self.player_name,
            player_class=self.player_class,
            grid_size=self.grid_size,
            num_rooms=self.num_rooms,
            seed=self.seed
        )
        self.root.wait_window(dlg)

        name, pclass, seed = dlg.result or (self.player_name, self.player_class, self.seed)
        self.state = GameState(
            grid_size=self.grid_size,
            num_rooms=self.num_rooms,
            player_name=name,
            player_class=pclass,
            seed=seed
        )
        self.play = PlayTab(self.play_frame, self.state, self.root)
        self.nb.select(4)   # switch to Play tab

    def _bind_tab_keys(self):
        """Binds Ctrl+Tab and Ctrl+Shift+Tab for navigation."""
        # Note: Control-Tab is a standard Tkinter event
        self.root.bind("<Control-Tab>", lambda e: self._handle_tab_switch(1))
        self.root.bind("<Control-ISO_Left_Tab>", lambda e: self._handle_tab_switch(-1))
        # On some Windows/Linux systems, Ctrl+Shift+Tab maps to Control-Shift-Tab
        self.root.bind("<Control-Shift-Tab>", lambda e: self._handle_tab_switch(-1))

    def _handle_tab_switch(self, direction: int):
        """Calculates the next tab index and selects it."""
        current = self.nb.index(self.nb.select())
        total = self.nb.index("end")
        
        # Calculate new index with wraparound logic
        new_index = (current + direction) % total
        self.nb.select(new_index)
        
        # Return 'break' to prevent the default system behavior if necessary
        return "break"
    
    def _on_tab_changed(self, event=None):
    #"""Open character creation the first time the Play tab is selected."""
        if self._char_creation_done:
            return
        current = self.nb.index(self.nb.select())
        if current == 4:                     # Play tab
            self._char_creation_done = True
            self._show_character_creation()

# Launcher
def launch(grid: int = 8, rooms: int = 12, seed: Optional[int] = None,
           player_name: str = "Hero", player_class: str = "Warrior") -> None:
    root = tk.Tk()
    App(root, state=None, grid_size=grid, num_rooms=rooms, seed=seed,
        player_name=player_name, player_class=player_class)
    root.mainloop()


def main() -> None:
    ap = argparse.ArgumentParser(description="AI Dungeon Master GUI")
    ap.add_argument("--grid", type=int, default=8)
    ap.add_argument("--rooms", type=int, default=12)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--name", type=str, default="Hero")
    ap.add_argument("--class", dest="player_class", type=str, default="Warrior",
                    choices=["Warrior", "Rogue", "Cleric", "Mage"])
    args = ap.parse_args()
    launch(grid=args.grid, rooms=args.rooms, seed=args.seed,
           player_name=args.name, player_class=args.player_class)


if __name__ == "__main__":
    main()
