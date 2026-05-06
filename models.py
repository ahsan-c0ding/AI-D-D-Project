"""
Core data models for the AI Dungeon Master project.
Shared by csp_generator, comparison_generators, npc_decision_tree, bayesian_combat.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


# Enums

class RoomType(Enum):
    START = "start"
    NORMAL = "normal"
    TREASURE = "treasure"
    BOSS = "boss"
    TRAP = "trap"
    MERCHANT = "merchant"


class ItemType(Enum):
    WEAPON = "weapon"
    POTION = "potion"
    GOLD = "gold"
    KEY = "key"
    ARMOR = "armor"


class NPCType(Enum):
    ENEMY = "enemy"
    FRIENDLY = "friendly"
    NEUTRAL = "neutral"
    MERCHANT = "merchant"
    BOSS = "boss"


# Items / NPCs / Player

@dataclass
class Item:
    name: str
    item_type: ItemType
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NPC:
    name: str
    npc_type: NPCType
    hp: int = 0
    attack: int = 0
    defense: int = 0
    dialogue: List[str] = field(default_factory=list)
    inventory: List[Item] = field(default_factory=list)


@dataclass
class Player:
    name: str
    hp: int = 100
    max_hp: int = 100
    attack: int = 10
    defense: int = 5
    position: Tuple[int, int] = (0, 0)
    gold: int = 0
    inventory: List[Item] = field(default_factory=list)

    # D&D-style ability scores (used by skill checks).
    # Default 10 = average human; higher = better at the corresponding check.
    strength: int = 10       # break doors, smash obstacles
    dexterity: int = 10      # dodge traps, sneak, flee
    constitution: int = 10   # poison resistance (future)
    intelligence: int = 10   # arcane / lore checks (future)
    wisdom: int = 10         # insight — read NPC intent
    charisma: int = 10       # persuasion, intimidation, haggle


# Room / Dungeon

@dataclass
class Room:
    coordinates: Tuple[int, int]
    room_type: RoomType
    description: str = ""
    connections: Set[Tuple[int, int]] = field(default_factory=set)
    items: List[Item] = field(default_factory=list)
    npcs: List[NPC] = field(default_factory=list)
    visited: bool = False

    def connect_to(self, coords: Tuple[int, int]) -> None:
        """Add a bidirectional connection (caller is responsible for the other side)."""
        self.connections.add(coords)


class Dungeon:
    """Container for all rooms in a generated dungeon."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.rooms: Dict[Tuple[int, int], Room] = {}

    def add_room(self, room: Room) -> None:
        self.rooms[room.coordinates] = room

    def get_room(self, coords: Tuple[int, int]) -> Optional[Room]:
        return self.rooms.get(coords)

    @property
    def start_position(self) -> Optional[Tuple[int, int]]:
        for coords, room in self.rooms.items():
            if room.room_type == RoomType.START:
                return coords
        return None

    @property
    def boss_position(self) -> Optional[Tuple[int, int]]:
        for coords, room in self.rooms.items():
            if room.room_type == RoomType.BOSS:
                return coords
        return None

    def __repr__(self) -> str:
        return (
            f"Dungeon({self.width}x{self.height}, "
            f"{len(self.rooms)} rooms, "
            f"start={self.start_position}, boss={self.boss_position})"
        )

    def render_ascii(self) -> str:
        """Simple ASCII grid view — useful for debugging and screenshots."""
        glyphs = {
            RoomType.START: "S",
            RoomType.BOSS: "B",
            RoomType.TREASURE: "T",
            RoomType.MERCHANT: "M",
            RoomType.TRAP: "X",
            RoomType.NORMAL: ".",
        }
        lines = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                room = self.rooms.get((x, y))
                row.append(glyphs[room.room_type] if room else " ")
            lines.append(" ".join(row))
        return "\n".join(lines)
