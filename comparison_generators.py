"""
Alternative dungeon-generation algorithms for comparison with CSP.

Each generator exposes:
  - ``generate()``     — one-shot, returns a populated Dungeon
  - ``solve_steps()``  — generator yielding ``select`` / ``assign`` / ``succeed``
                          events so the GUI Generation Inspector can animate
                          and compare the algorithms side by side with CSP.
  - ``finalize()``     — call after a ``succeed`` event from solve_steps to
                          populate items + NPCs and return the Dungeon.
"""

from typing import List, Tuple, Optional, Set
from models import Dungeon, Room, RoomType, Item, ItemType, NPC, NPCType
from collections import deque
import random

# Base
class BaseDungeonGenerator:
    """Base class for dungeon generators (shared helpers + populate logic)."""

    def __init__(self, width: int, height: int, num_rooms: int,
                 seed: Optional[int] = None):
        self.width = width
        self.height = height
        self.num_rooms = num_rooms
        self.dungeon = Dungeon(width, height)

        if seed is not None:
            random.seed(seed)

        self.nodes_explored = 0
        self.generation_time = 0
        # Backtracks are conceptually 0 for these algorithms but we expose the
        # field so the eval harness can read it uniformly.
        self.backtrack_count = 0

    def is_valid_position(self, coords: Tuple[int, int]) -> bool:
        x, y = coords
        return 0 <= x < self.width and 0 <= y < self.height

    def get_neighbors(self, coords: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = coords
        neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [n for n in neighbors if self.is_valid_position(n)]

    # -- shared post-generation populate (items + NPCs) 

    def populate_rooms(self):
        """Add items and NPCs to rooms after the layout is fixed."""
        for coords, room in self.dungeon.rooms.items():
            if room.room_type == RoomType.TREASURE:
                room.items.append(Item("Gold Coins", ItemType.GOLD,
                                       {"amount": random.randint(50, 150)}))
                if random.random() > 0.5:
                    room.items.append(Item("Health Potion", ItemType.POTION,
                                           {"heal": 30}))
            elif room.room_type == RoomType.NORMAL:
                if random.random() > 0.7:
                    weapons = ["Iron Sword", "Rusty Dagger", "Wooden Staff"]
                    room.items.append(Item(
                        random.choice(weapons),
                        ItemType.WEAPON,
                        {"damage": random.randint(5, 15)},
                    ))

        treasure_rooms = [r for r in self.dungeon.rooms.values()
                          if r.room_type == RoomType.TREASURE]
        if treasure_rooms:
            key_room = random.choice(treasure_rooms)
            key_room.items.append(Item("Boss Key", ItemType.KEY,
                                       {"opens": "boss_room"}))

        for room in self.dungeon.rooms.values():
            if room.room_type == RoomType.NORMAL and random.random() > 0.6:
                enemies = ["Goblin", "Skeleton", "Giant Rat"]
                room.npcs.append(NPC(
                    name=random.choice(enemies),
                    npc_type=NPCType.ENEMY,
                    hp=random.randint(20, 40),
                    attack=random.randint(5, 12),
                    defense=random.randint(1, 5),
                    dialogue=["Grr!", "You shall not pass!"],
                ))
            elif room.room_type == RoomType.MERCHANT:
                room.npcs.append(NPC(
                    name="Merchant",
                    npc_type=NPCType.MERCHANT,
                    hp=50, attack=0, defense=10,
                    dialogue=["Welcome! Care to see my wares?"],
                    inventory=[
                        Item("Health Potion", ItemType.POTION, {"heal": 50}),
                        Item("Steel Sword", ItemType.WEAPON, {"damage": 20}),
                    ],
                ))
            elif room.room_type == RoomType.BOSS:
                room.npcs.append(NPC(
                    name="Dragon",
                    npc_type=NPCType.BOSS,
                    hp=100, attack=25, defense=10,
                    dialogue=["You dare challenge me?!"],
                ))

    # -- step API 

    def finalize(self) -> Dungeon:
        """Call after a ``succeed`` event to populate content and return."""
        self.populate_rooms()
        return self.dungeon

    # Subclasses must implement solve_steps()
    def solve_steps(self):  # pragma: no cover
        raise NotImplementedError


# BFS
class BFSDungeonGenerator(BaseDungeonGenerator):
    """Breadth-first dungeon expansion from a centre point."""

    def solve_steps(self):
        start_coords = (self.width // 2, self.height // 2)
        start_room = Room(coordinates=start_coords,
                          room_type=RoomType.START,
                          description="The entrance to the dungeon.")
        self.dungeon.add_room(start_room)
        self.nodes_explored += 1
        yield {"kind": "assign",
               "coords": start_coords,
               "room_type": RoomType.START.value,
               "connections": [],
               "nodes": self.nodes_explored,
               "backtracks": 0,
               "depth": 1}

        queue = deque([(start_coords, 0)])
        visited = {start_coords}
        rooms_created = 1
        max_distance = 0
        farthest_room = start_coords

        while queue and rooms_created < self.num_rooms:
            current_coords, distance = queue.popleft()
            self.nodes_explored += 1
            yield {"kind": "select",
                   "coords": current_coords,
                   "nodes": self.nodes_explored,
                   "backtracks": 0,
                   "depth": rooms_created}

            neighbors = self.get_neighbors(current_coords)
            random.shuffle(neighbors)

            for neighbor in neighbors:
                if neighbor in visited or rooms_created >= self.num_rooms:
                    continue
                room_type = self._select_room_type(rooms_created)
                new_room = Room(coordinates=neighbor,
                                room_type=room_type,
                                description=f"A {room_type.value} room.")
                current_room = self.dungeon.get_room(current_coords)
                new_room.connect_to(current_coords)
                current_room.connect_to(neighbor)
                self.dungeon.add_room(new_room)
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
                rooms_created += 1
                if distance + 1 > max_distance:
                    max_distance = distance + 1
                    farthest_room = neighbor

                yield {"kind": "assign",
                       "coords": neighbor,
                       "room_type": room_type.value,
                       "connections": sorted(new_room.connections),
                       "nodes": self.nodes_explored,
                       "backtracks": 0,
                       "depth": rooms_created}

        # Place boss at farthest room
        if farthest_room != start_coords:
            boss_room = self.dungeon.get_room(farthest_room)
            boss_room.room_type = RoomType.BOSS
            boss_room.description = "The boss chamber."
            yield {"kind": "assign",
                   "coords": farthest_room,
                   "room_type": RoomType.BOSS.value,
                   "connections": sorted(boss_room.connections),
                   "nodes": self.nodes_explored,
                   "backtracks": 0,
                   "depth": rooms_created,
                   "note": "boss promoted to farthest leaf"}

        yield {"kind": "succeed",
               "nodes": self.nodes_explored,
               "backtracks": 0}

    def generate(self) -> Dungeon:
        for _ in self.solve_steps():
            pass
        self.populate_rooms()
        return self.dungeon

    def _select_room_type(self, room_count: int) -> RoomType:
        if room_count == self.num_rooms - 1:
            return RoomType.BOSS
        types = [RoomType.NORMAL] * 5 + [RoomType.TREASURE] * 2 + [RoomType.TRAP, RoomType.MERCHANT]
        return random.choice(types)


# DFS
class DFSDungeonGenerator(BaseDungeonGenerator):
    """Depth-first dungeon expansion (long winding corridors)."""

    def solve_steps(self):
        start_coords = (self.width // 2, self.height // 2)
        start_room = Room(coordinates=start_coords,
                          room_type=RoomType.START,
                          description="The entrance to the dungeon.")
        self.dungeon.add_room(start_room)
        self.nodes_explored += 1
        yield {"kind": "assign",
               "coords": start_coords,
               "room_type": RoomType.START.value,
               "connections": [],
               "nodes": self.nodes_explored,
               "backtracks": 0,
               "depth": 1}

        visited = {start_coords}
        yield from self._dfs_steps(start_coords, visited, 1)

        # Boss at a random leaf
        leaf_rooms = [coords for coords, room in self.dungeon.rooms.items()
                      if len(room.connections) == 1
                      and room.room_type != RoomType.START]
        if leaf_rooms:
            boss_coords = random.choice(leaf_rooms)
            boss_room = self.dungeon.get_room(boss_coords)
            boss_room.room_type = RoomType.BOSS
            yield {"kind": "assign",
                   "coords": boss_coords,
                   "room_type": RoomType.BOSS.value,
                   "connections": sorted(boss_room.connections),
                   "nodes": self.nodes_explored,
                   "backtracks": 0,
                   "depth": len(self.dungeon.rooms),
                   "note": "boss promoted to random leaf"}

        yield {"kind": "succeed",
               "nodes": self.nodes_explored,
               "backtracks": 0}

    def _dfs_steps(self, coords: Tuple[int, int],
                   visited: Set[Tuple[int, int]], depth: int):
        if len(self.dungeon.rooms) >= self.num_rooms:
            return
        self.nodes_explored += 1
        yield {"kind": "select",
               "coords": coords,
               "nodes": self.nodes_explored,
               "backtracks": 0,
               "depth": depth}

        neighbors = self.get_neighbors(coords)
        random.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor in visited or len(self.dungeon.rooms) >= self.num_rooms:
                continue
            room_type = self._select_room_type(len(self.dungeon.rooms))
            new_room = Room(coordinates=neighbor,
                            room_type=room_type,
                            description=f"A {room_type.value} room.")
            current_room = self.dungeon.get_room(coords)
            new_room.connect_to(coords)
            current_room.connect_to(neighbor)
            self.dungeon.add_room(new_room)
            visited.add(neighbor)
            yield {"kind": "assign",
                   "coords": neighbor,
                   "room_type": room_type.value,
                   "connections": sorted(new_room.connections),
                   "nodes": self.nodes_explored,
                   "backtracks": 0,
                   "depth": len(self.dungeon.rooms)}
            yield from self._dfs_steps(neighbor, visited, depth + 1)

    def generate(self) -> Dungeon:
        for _ in self.solve_steps():
            pass
        self.populate_rooms()
        return self.dungeon

    def _select_room_type(self, room_count: int) -> RoomType:
        types = [RoomType.NORMAL] * 5 + [RoomType.TREASURE] * 2 + [RoomType.TRAP, RoomType.MERCHANT]
        return random.choice(types)

# Greedy
class GreedyDungeonGenerator(BaseDungeonGenerator):
    """Greedy dungeon expansion guided by a distance/clustering heuristic."""

    def solve_steps(self):
        start_coords = (self.width // 2, self.height // 2)
        start_room = Room(coordinates=start_coords,
                          room_type=RoomType.START,
                          description="The entrance to the dungeon.")
        self.dungeon.add_room(start_room)
        self.nodes_explored += 1
        yield {"kind": "assign",
               "coords": start_coords,
               "room_type": RoomType.START.value,
               "connections": [],
               "nodes": self.nodes_explored,
               "backtracks": 0,
               "depth": 1}

        occupied = {start_coords}
        for i in range(1, self.num_rooms):
            candidates = set()
            for coords in occupied:
                for neighbor in self.get_neighbors(coords):
                    if neighbor not in occupied:
                        candidates.add(neighbor)
            if not candidates:
                break

            self.nodes_explored += len(candidates)
            best_coords = max(candidates,
                              key=lambda c: self._heuristic(c, occupied))
            yield {"kind": "select",
                   "coords": best_coords,
                   "nodes": self.nodes_explored,
                   "backtracks": 0,
                   "depth": i,
                   "note": f"argmax heuristic over {len(candidates)} candidates"}

            room_type = self._select_room_type(i)
            new_room = Room(coordinates=best_coords,
                            room_type=room_type,
                            description=f"A {room_type.value} room.")
            for neighbor in self.get_neighbors(best_coords):
                if neighbor in occupied:
                    new_room.connect_to(neighbor)
                    self.dungeon.get_room(neighbor).connect_to(best_coords)
            self.dungeon.add_room(new_room)
            occupied.add(best_coords)
            yield {"kind": "assign",
                   "coords": best_coords,
                   "room_type": room_type.value,
                   "connections": sorted(new_room.connections),
                   "nodes": self.nodes_explored,
                   "backtracks": 0,
                   "depth": i + 1}

        yield {"kind": "succeed",
               "nodes": self.nodes_explored,
               "backtracks": 0}

    def _heuristic(self, coords: Tuple[int, int],
                   occupied: Set[Tuple[int, int]]) -> float:
        x, y = coords
        center_x, center_y = self.width // 2, self.height // 2
        distance_from_center = abs(x - center_x) + abs(y - center_y)
        occupied_neighbors = sum(1 for n in self.get_neighbors(coords)
                                 if n in occupied)
        score = distance_from_center - (occupied_neighbors * 2)
        score += random.uniform(-1, 1)
        return score

    def generate(self) -> Dungeon:
        for _ in self.solve_steps():
            pass
        self.populate_rooms()
        return self.dungeon

    def _select_room_type(self, room_count: int) -> RoomType:
        if room_count == self.num_rooms - 1:
            return RoomType.BOSS
        types = [RoomType.NORMAL] * 5 + [RoomType.TREASURE] * 2 + [RoomType.TRAP, RoomType.MERCHANT]
        return random.choice(types)

# Smoke test
if __name__ == "__main__":
    import time

    print("=" * 60)
    print("DUNGEON GENERATION ALGORITHM COMPARISON")
    print("=" * 60)

    params = {"width": 6, "height": 6, "num_rooms": 10, "seed": 42}
    generators = [
        ("BFS", BFSDungeonGenerator),
        ("DFS", DFSDungeonGenerator),
        ("Greedy", GreedyDungeonGenerator),
    ]
    results = []
    for name, GenCls in generators:
        gen = GenCls(**params)
        t0 = time.time()
        dungeon = gen.generate()
        results.append({
            "name": name,
            "nodes_explored": gen.nodes_explored,
            "time": time.time() - t0,
            "rooms": len(dungeon.rooms) if dungeon else 0,
        })

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Algorithm':<15} {'Nodes Explored':<20} {'Time (s)':<15} {'Rooms'}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<15} {r['nodes_explored']:<20} "
              f"{r['time']:<15.6f} {r['rooms']}")
