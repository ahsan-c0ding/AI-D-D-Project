"""
CSP-based Dungeon Generator

Uses Constraint Satisfaction Problem solving with backtracking to generate
valid dungeons. Two ways to drive the search:

  - ``backtrack()`` / ``generate()`` — one-shot, returns when done.
  - ``solve_steps()`` — generator that yields events at each select / consider
    / assign / reject / backtrack / succeed / fail step. Used by the GUI's
    Generation Inspector to animate the algorithm.

Incorporates the circular growth heuristic from the gameplay branch for
more interesting, non-linear dungeon layouts, plus a richer populate_rooms()
with diverse enemy groups and key-insurance logic.
"""

import math
import random
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from models import Dungeon, Item, ItemType, NPC, NPCType, Room, RoomType


class DungeonCSP:
    """
    Constraint Satisfaction Problem for dungeon generation.

    Variables: Room positions in the grid
    Domains: Valid room types and configurations
    Constraints:
        1. All rooms must be connected
        2. Boss room must be farthest from start
        3. Key must be placed before boss room is accessible
        4. No overlapping rooms
        5. Minimum number of rooms

    Heuristics: Circular growth via Euclidean distance to center
    (gameplay branch) prevents long snake-like corridors.
    """

    def __init__(self, width: int, height: int, num_rooms: int,
                 seed: Optional[int] = None):
        self.width = width
        self.height = height
        self.num_rooms = num_rooms
        self.start_pos = (width // 2, height // 2)
        self.dungeon = Dungeon(width, height)

        if seed is not None:
            random.seed(seed)

        # CSP components
        self.variables: List[Tuple[int, int]] = []
        self.domains: Dict[Tuple[int, int], List[RoomType]] = {}
        self.assignment: Dict[Tuple[int, int], Room] = {}

        # Tracking
        self.backtrack_count = 0
        self.nodes_explored = 0

    # Constraint helpers
    def is_valid_position(self, coords: Tuple[int, int]) -> bool:
        """Check if position is within bounds."""
        x, y = coords
        return 0 <= x < self.width and 0 <= y < self.height

    def get_neighbors(self, coords: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get all valid neighboring positions (4-directional)."""
        x, y = coords
        neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [n for n in neighbors if self.is_valid_position(n)]

    def is_connected(self) -> bool:
        """All assigned rooms reachable via connections (BFS)."""
        if not self.assignment:
            return True
        start = self.start_pos
        if start not in self.assignment:
            start = next(iter(self.assignment.keys()))
        visited: Set[Tuple[int, int]] = set()
        queue = deque([start])
        visited.add(start)
        while queue:
            current = queue.popleft()
            for neighbor in self.get_neighbors(current):
                if neighbor in self.assignment and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return len(visited) == len(self.assignment)

    def get_distance_from_start(self, coords: Tuple[int, int]) -> int:
        """Shortest distance from start room using BFS over connections."""
        if not self.assignment:
            return 0
        start = next((pos for pos, room in self.assignment.items()
                      if room.room_type == RoomType.START), None)
        if start is None or coords == start:
            return 0
        visited = {start: 0}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            if current == coords:
                return visited[current]
            for neighbor in self.get_neighbors(current):
                if neighbor in self.assignment and neighbor not in visited:
                    visited[neighbor] = visited[current] + 1
                    queue.append(neighbor)
        return -1

    def is_consistent(self, coords: Tuple[int, int],
                      room_type: RoomType) -> bool:
        """Check whether assigning ``room_type`` to ``coords`` is consistent."""
        if not self.is_valid_position(coords) or coords in self.assignment:
            return False

        if self.assignment:
            has_connection = any(neighbor in self.assignment
                                 for neighbor in self.get_neighbors(coords))
            if not has_connection:
                return False

        if room_type == RoomType.START:
            if any(room.room_type == RoomType.START
                   for room in self.assignment.values()):
                return False

        if room_type == RoomType.BOSS:
            assigned_types = [r.room_type for r in self.assignment.values()]
            if RoomType.START not in assigned_types:
                return False
            if len(self.assignment) < self.num_rooms // 2:
                return False

        if room_type == RoomType.TREASURE:
            count = sum(1 for r in self.assignment.values()
                        if r.room_type == RoomType.TREASURE)
            if count >= 2:
                return False

        if room_type == RoomType.TRAP:
            count = sum(1 for r in self.assignment.values()
                        if r.room_type == RoomType.TRAP)
            if count >= 2:
                return False

        if room_type == RoomType.MERCHANT:
            count = sum(1 for r in self.assignment.values()
                        if r.room_type == RoomType.MERCHANT)
            if count >= 1:
                return False

        return True
                        
    # Variable & value ordering — circular growth heuristic (gameplay)
    def select_unassigned_variable(self) -> Optional[Tuple[int, int]]:
        """
        Circular Growth heuristic: prioritizes candidate positions closest
        to the centre of the grid (+ small random noise for organic feel).
        Prevents long snake-like corridors.
        """
        if not self.assignment:
            return self.start_pos

        candidates: Set[Tuple[int, int]] = set()
        for assigned_pos in self.assignment:
            for neighbor in self.get_neighbors(assigned_pos):
                if neighbor not in self.assignment:
                    candidates.add(neighbor)

        if not candidates:
            return None

        sorted_candidates = sorted(
            candidates,
            key=lambda c: math.dist(c, self.start_pos) + random.random() * 1.8,
        )
        return random.choice(sorted_candidates[:3])

    def order_domain_values(self,
                             coords: Tuple[int, int]) -> List[RoomType]:
        """LCV-flavoured ordering tuned to the dungeon distribution."""
        assigned_types = [room.room_type for room in self.assignment.values()]
        num_assigned = len(self.assignment)

        if num_assigned == 0:
            return [RoomType.START]

        if num_assigned == self.num_rooms - 1:
            if RoomType.BOSS not in assigned_types:
                return [RoomType.BOSS]

        priority: List[RoomType] = []
        if RoomType.START not in assigned_types:
            priority.append(RoomType.START)
        priority.extend([RoomType.NORMAL] * 6)
        if assigned_types.count(RoomType.TREASURE) < 2:
            priority.append(RoomType.TREASURE)
        if assigned_types.count(RoomType.TRAP) < 2:
            priority.append(RoomType.TRAP)
        if assigned_types.count(RoomType.MERCHANT) < 1:
            priority.append(RoomType.MERCHANT)
        if (num_assigned > self.num_rooms // 2
                and RoomType.BOSS not in assigned_types):
            priority.append(RoomType.BOSS)

        random.shuffle(priority)
        return priority

    # Backtracking — one-shot wrapper + step-by-step generator
    def backtrack(self) -> bool:
        """
        CSP Backtracking algorithm (one-shot wrapper).
        Returns True if a valid assignment is found.
        """
        for event in self.solve_steps():
            if event["kind"] == "succeed":
                return True
            if event["kind"] == "fail":
                return False
        return False

    def solve_steps(self):
        """
        Step-by-step CSP backtracking as a generator.  Yields a dict event
        at each significant point so the Generation Inspector can animate
        the algorithm.

        Event kinds (each event also carries running counters
        ``nodes`` and ``backtracks``):

        - ``select``    : about to pick values for ``coords``
        - ``consider``  : trying ``room_type`` at ``coords``
        - ``reject``    : ``(coords, room_type)`` violated a constraint
        - ``assign``    : committed ``coords -> room_type`` with connections
        - ``backtrack`` : un-assigned ``coords``
        - ``succeed``   : final solution found
        - ``fail``      : root call exhausted with no solution
        """
        success = yield from self._solve_steps_recursive()
        if success:
            yield {"kind": "succeed",
                   "nodes": self.nodes_explored,
                   "backtracks": self.backtrack_count}
        else:
            yield {"kind": "fail",
                   "nodes": self.nodes_explored,
                   "backtracks": self.backtrack_count}

    def _solve_steps_recursive(self):
        """Recursive generator. Returns True/False via StopIteration.value."""
        self.nodes_explored += 1

        if len(self.assignment) == self.num_rooms:
            types = [room.room_type for room in self.assignment.values()]
            if RoomType.START in types and RoomType.BOSS in types:
                return self.is_connected()
            return False

        coords = self.select_unassigned_variable()
        if coords is None:
            return False

        yield {"kind": "select",
               "coords": coords,
               "nodes": self.nodes_explored,
               "backtracks": self.backtrack_count,
               "depth": len(self.assignment)}

        for room_type in self.order_domain_values(coords):
            yield {"kind": "consider",
                   "coords": coords,
                   "room_type": room_type.value,
                   "nodes": self.nodes_explored,
                   "backtracks": self.backtrack_count}

            if not self.is_consistent(coords, room_type):
                yield {"kind": "reject",
                       "coords": coords,
                       "room_type": room_type.value,
                       "reason": "constraint_violation",
                       "nodes": self.nodes_explored,
                       "backtracks": self.backtrack_count}
                continue

            room = Room(
                coordinates=coords,
                room_type=room_type,
                description=self.generate_room_description(room_type),
            )
            self.assignment[coords] = room
            for neighbor in self.get_neighbors(coords):
                if neighbor in self.assignment:
                    room.connect_to(neighbor)
                    self.assignment[neighbor].connect_to(coords)

            yield {"kind": "assign",
                   "coords": coords,
                   "room_type": room_type.value,
                   "connections": sorted(room.connections),
                   "nodes": self.nodes_explored,
                   "backtracks": self.backtrack_count,
                   "depth": len(self.assignment)}

            success = yield from self._solve_steps_recursive()
            if success:
                return True

            self.backtrack_count += 1
            for neighbor in self.get_neighbors(coords):
                if neighbor in self.assignment:
                    self.assignment[neighbor].connections.discard(coords)
            del self.assignment[coords]

            yield {"kind": "backtrack",
                   "coords": coords,
                   "room_type": room_type.value,
                   "nodes": self.nodes_explored,
                   "backtracks": self.backtrack_count,
                   "depth": len(self.assignment)}

        return False

    def finalize(self) -> "Dungeon":
        """
        Call after solve_steps() yields a ``succeed`` event to populate
        the dungeon with items and NPCs and return it.
        """
        for room in self.assignment.values():
            self.dungeon.add_room(room)
        self.populate_rooms()
        return self.dungeon

  
    # Room descriptions
    def generate_room_description(self, room_type: RoomType) -> str:
        descriptions = {
            RoomType.START:    ["A dusty entrance with a faint breeze from above."],
            RoomType.NORMAL:   ["A damp stone chamber.", "A corridor filled with echoes.",
                                "Crumbling walls line this passage."],
            RoomType.TREASURE: ["Glinting piles of gold and forgotten relics await.",
                                "A heavy iron chest sits in the corner."],
            RoomType.BOSS:     ["A massive, ominous throne room. Something stirs within."],
            RoomType.TRAP:     ["The air feels heavy with hidden danger.",
                                "Pressure plates dot the floor."],
            RoomType.MERCHANT: ["A cozy nook lit by a small campfire.",
                                "The smell of pipe smoke lingers here."],
        }
        return random.choice(descriptions.get(room_type, ["An empty room."]))

    # Population — diverse enemies, merchants, key insurance (gameplay)
    def populate_rooms(self) -> None:
        """Add items and NPCs to rooms after the layout is fixed."""
        for coords, room in self.assignment.items():

            # Treasure rooms
            if room.room_type == RoomType.TREASURE:
                room.items.append(Item("Gold Coins", ItemType.GOLD,
                                       {"amount": random.randint(50, 150)}))
                if random.random() > 0.5:
                    room.items.append(Item("Health Potion", ItemType.POTION,
                                          {"heal": 30}))

            # Merchant rooms
            elif room.room_type == RoomType.MERCHANT:
                merchant = NPC(
                    name="Traveling Merchant",
                    npc_type=NPCType.MERCHANT,
                    hp=50, attack=5, defense=10,
                    dialogue=["Greetings! Supplies for the brave.",
                              "A dangerous journey needs the right gear."],
                )
                room.npcs.append(merchant)
                room.items.append(Item("Shop Potion", ItemType.POTION,
                                      {"heal": 25}))

            # Boss room
            elif room.room_type == RoomType.BOSS:
                room.npcs.append(
                    NPC("Dragon King", NPCType.BOSS, 120, 22, 12,
                        ["BURN!", "You dare challenge me?!"]))

            # Normal rooms — diverse enemy groups (40 % chance, 1-3 enemies)
            elif room.room_type == RoomType.NORMAL:
                if random.random() > 0.6:
                    num_enemies = random.randint(1, 3)
                    for _ in range(num_enemies):
                        room.npcs.append(self._create_random_enemy())

            # Trap rooms get a small gold reward for surviving
            elif room.room_type == RoomType.TRAP:
                if random.random() > 0.5:
                    room.items.append(Item("Trap Loot", ItemType.GOLD,
                                          {"amount": random.randint(10, 40)}))

        # Key insurance — guarantee the Boss Key exists somewhere
        all_rooms = list(self.assignment.values())
        has_key = any(
            any(i.item_type == ItemType.KEY for i in r.items)
            for r in all_rooms
        )
        if not has_key:
            treasure_rooms = [r for r in all_rooms
                              if r.room_type == RoomType.TREASURE]
            if treasure_rooms:
                target_room = random.choice(treasure_rooms)
            else:
                candidates = [r for r in all_rooms
                              if r.room_type != RoomType.START
                              and r.room_type != RoomType.BOSS]
                target_room = random.choice(candidates) if candidates else all_rooms[0]
            target_room.items.append(
                Item("Boss Key", ItemType.KEY, {"opens": "boss_room"}))

    def _create_random_enemy(self) -> NPC:
        """Factory for generating diverse enemies."""
        types = [
            ("Goblin",       25,  8,  2, ["Hehehe!"]),
            ("Kobold",       15, 10,  1, ["Yip yip!"]),
            ("Wolf",         20, 12,  3, ["Grrr..."]),
            ("Skeleton",     30,  9,  4, ["..."]),
            ("Goblin Leader", 45, 14,  5, ["To arms!"]),
            ("Dark Mage",    20, 16,  1, ["Begone!"]),
        ]
        name, hp, atk, def_, dialogue = random.choice(types)
        return NPC(name=name, npc_type=NPCType.ENEMY,
                   hp=hp, attack=atk, defense=def_, dialogue=dialogue)
      

    # One-shot generation

    def generate(self) -> Optional[Dungeon]:
        """Run the CSP to completion and return a populated Dungeon, or None."""
        success = self.backtrack()
        if not success:
            return None
        return self.finalize()
