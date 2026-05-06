"""
Decision Tree for NPC Behavior

NPCs make decisions based on player state and game conditions. The trees
are hand-written (not learned) so the structure is explicit and inspectable.

Each :class:`DecisionNode` carries a human-readable ``label`` describing
what the node tests (for decision nodes) or which action it chooses
(for leaf nodes). The label is used by the GUI's NPC Brain tab to
render the tree as a graph and highlight the path taken for a given
player state.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

from enum import Enum

from models import NPC, NPCType, Player


class NPCAction(Enum):
    """Possible NPC actions"""
    ATTACK = "attack"
    FLEE = "flee"
    DEFEND = "defend"
    TALK = "talk"
    TRADE = "trade"
    HELP = "help"
    IDLE = "idle"
    STEAL = "steal"
    SURRENDER = "surrender"


class DecisionNode:
    """
    Node in the decision tree. Either a decision (condition + true/false
    branches) or an action (leaf).
    """

    def __init__(self,
                 condition: Optional[Callable] = None,
                 action: Optional[NPCAction] = None,
                 label: Optional[str] = None):
        """
        Args:
            condition: function (npc, player, game_state) -> bool
            action: NPCAction taken at a leaf
            label: human-readable description of the test or action
        """
        self.condition = condition
        self.action = action
        if label is None:
            if action is not None:
                label = action.value.upper()
            elif condition is not None:
                label = "<unlabelled condition>"
            else:
                label = "<empty>"
        self.label = label
        self.true_branch: Optional["DecisionNode"] = None
        self.false_branch: Optional["DecisionNode"] = None

    def is_leaf(self) -> bool:
        return self.action is not None

    def evaluate(self, npc: NPC, player: Player,
                 game_state: Dict[str, Any]) -> NPCAction:
        if self.is_leaf():
            return self.action
        if self.condition(npc, player, game_state):
            if self.true_branch:
                return self.true_branch.evaluate(npc, player, game_state)
            return NPCAction.IDLE
        else:
            if self.false_branch:
                return self.false_branch.evaluate(npc, player, game_state)
            return NPCAction.IDLE

    def evaluate_with_path(
        self, npc: NPC, player: Player, game_state: Dict[str, Any]
    ) -> Tuple[NPCAction, List["DecisionNode"], List[bool]]:
        """
        Return the chosen action plus:
          - ``path``     : the list of nodes visited in order (including the leaf)
          - ``branches`` : list of bool decisions taken at each non-leaf
                           (True if condition held, False otherwise).
                           ``len(branches) == len(path) - 1`` always.
        """
        path: List["DecisionNode"] = [self]
        branches: List[bool] = []
        node = self
        while not node.is_leaf():
            cond_value = bool(node.condition(npc, player, game_state))
            branches.append(cond_value)
            nxt = node.true_branch if cond_value else node.false_branch
            if nxt is None:
                # Mirror evaluate(): treat dangling as idle
                fallback = DecisionNode(action=NPCAction.IDLE,
                                        label="IDLE (fallback)")
                path.append(fallback)
                return NPCAction.IDLE, path, branches
            path.append(nxt)
            node = nxt
        return node.action, path, branches


# Tree builders
class NPCDecisionTree:
    """Decision tree builders for each NPC type."""

    # Helper for building leaves with sensible labels
    @staticmethod
    def _leaf(action: NPCAction) -> DecisionNode:
        return DecisionNode(action=action, label=action.value.upper())

    # -- enemy 

    @staticmethod
    def build_enemy_tree() -> DecisionNode:
        """
        Logic:
          - If NPC effective health < 20%, flee (or fight on if player is weak)
          - Else if player health < 30, attack aggressively
          - Else if NPC effective health < 50%, defend
          - Else attack
        """
        root = DecisionNode(
            condition=lambda npc, player, state:
                (npc.hp / (npc.hp + npc.defense * 10)) < 0.2,
            label="NPC HP < 20% effective",
        )

        low_health = DecisionNode(
            condition=lambda npc, player, state: player.hp > 50,
            label="Player HP > 50",
        )
        low_health.true_branch = NPCDecisionTree._leaf(NPCAction.FLEE)
        low_health.false_branch = NPCDecisionTree._leaf(NPCAction.ATTACK)
        root.true_branch = low_health

        normal_health = DecisionNode(
            condition=lambda npc, player, state: player.hp < 30,
            label="Player HP < 30",
        )
        normal_health.true_branch = NPCDecisionTree._leaf(NPCAction.ATTACK)

        npc_health_check = DecisionNode(
            condition=lambda npc, player, state:
                (npc.hp / (npc.hp + npc.defense * 10)) < 0.5,
            label="NPC HP < 50% effective",
        )
        npc_health_check.true_branch = NPCDecisionTree._leaf(NPCAction.DEFEND)
        npc_health_check.false_branch = NPCDecisionTree._leaf(NPCAction.ATTACK)
        normal_health.false_branch = npc_health_check
        root.false_branch = normal_health
        return root

    # -- merchant 

    @staticmethod
    def build_merchant_tree() -> DecisionNode:
        """If the player has gold or items the merchant offers a trade,
        otherwise just chats."""
        root = DecisionNode(
            condition=lambda npc, player, state: player.gold > 50,
            label="Player gold > 50",
        )
        root.true_branch = NPCDecisionTree._leaf(NPCAction.TRADE)

        has_items = DecisionNode(
            condition=lambda npc, player, state: len(player.inventory) > 0,
            label="Player has items",
        )
        has_items.true_branch = NPCDecisionTree._leaf(NPCAction.TRADE)
        has_items.false_branch = NPCDecisionTree._leaf(NPCAction.TALK)
        root.false_branch = has_items
        return root

    # -- friendly 

    @staticmethod
    def build_friendly_tree() -> DecisionNode:
        """Heal the wounded, otherwise greet first-time visitors."""
        root = DecisionNode(
            condition=lambda npc, player, state: player.hp < 50,
            label="Player HP < 50",
        )
        root.true_branch = NPCDecisionTree._leaf(NPCAction.HELP)

        first_encounter = DecisionNode(
            condition=lambda npc, player, state:
                state.get("npc_met", False) is False,
            label="First encounter",
        )
        first_encounter.true_branch = NPCDecisionTree._leaf(NPCAction.TALK)
        first_encounter.false_branch = NPCDecisionTree._leaf(NPCAction.IDLE)
        root.false_branch = first_encounter
        return root

    # -- neutral 

    @staticmethod
    def build_neutral_tree() -> DecisionNode:
        """Defend if attacked first; thieves steal from a stocked target;
        otherwise chat."""
        root = DecisionNode(
            condition=lambda npc, player, state:
                state.get("player_attacked", False),
            label="Player attacked first",
        )
        root.true_branch = NPCDecisionTree._leaf(NPCAction.ATTACK)

        is_thief = DecisionNode(
            condition=lambda npc, player, state:
                ("thief" in npc.name.lower() and len(player.inventory) > 0),
            label="Is thief AND player has items",
        )
        is_thief.true_branch = NPCDecisionTree._leaf(NPCAction.STEAL)
        is_thief.false_branch = NPCDecisionTree._leaf(NPCAction.TALK)
        root.false_branch = is_thief
        return root

    # -- boss 

    @staticmethod
    def build_boss_tree() -> DecisionNode:
        """Aggressive when low; defensive when player is fresh; otherwise
        attack normally."""
        root = DecisionNode(
            condition=lambda npc, player, state: (npc.hp / 100) < 0.3,
            label="Boss HP < 30%",
        )
        root.true_branch = NPCDecisionTree._leaf(NPCAction.ATTACK)  # desperate

        player_strong = DecisionNode(
            condition=lambda npc, player, state: player.hp > 70,
            label="Player HP > 70",
        )
        player_strong.true_branch = NPCDecisionTree._leaf(NPCAction.DEFEND)
        player_strong.false_branch = NPCDecisionTree._leaf(NPCAction.ATTACK)
        root.false_branch = player_strong
        return root

    # -- dispatch 

    @staticmethod
    def get_tree_for_npc(npc_type: NPCType) -> DecisionNode:
        if npc_type == NPCType.ENEMY:
            return NPCDecisionTree.build_enemy_tree()
        if npc_type == NPCType.MERCHANT:
            return NPCDecisionTree.build_merchant_tree()
        if npc_type == NPCType.FRIENDLY:
            return NPCDecisionTree.build_friendly_tree()
        if npc_type == NPCType.NEUTRAL:
            return NPCDecisionTree.build_neutral_tree()
        if npc_type == NPCType.BOSS:
            return NPCDecisionTree.build_boss_tree()
        return DecisionNode(action=NPCAction.IDLE, label="IDLE (default)")


# Behavior manager
class NPCBehaviorManager:
    """Manages NPC behavior using decision trees."""

    def __init__(self):
        self.decision_trees: Dict[NPCType, DecisionNode] = {}
        self._initialize_trees()

    def _initialize_trees(self):
        for npc_type in NPCType:
            self.decision_trees[npc_type] = NPCDecisionTree.get_tree_for_npc(npc_type)

    def get_npc_action(self, npc: NPC, player: Player,
                       game_state: Dict[str, Any]) -> NPCAction:
        tree = self.decision_trees.get(npc.npc_type)
        if tree:
            return tree.evaluate(npc, player, game_state)
        return NPCAction.IDLE

    def get_npc_action_with_path(
        self, npc: NPC, player: Player, game_state: Dict[str, Any]
    ) -> Tuple[NPCAction, List[DecisionNode], List[bool]]:
        """Same as :meth:`get_npc_action` but also returns the path through
        the tree, suitable for animation in the GUI."""
        tree = self.decision_trees.get(npc.npc_type)
        if tree:
            return tree.evaluate_with_path(npc, player, game_state)
        # No tree — synthesize a single-leaf path
        leaf = DecisionNode(action=NPCAction.IDLE, label="IDLE (no tree)")
        return NPCAction.IDLE, [leaf], []

    def get_npc_dialogue(self, npc: NPC, action: NPCAction) -> str:
        dialogue_map = {
            NPCAction.ATTACK: npc.dialogue[0] if npc.dialogue else "Prepare to fight!",
            NPCAction.FLEE: "I must retreat!",
            NPCAction.DEFEND: "I'll protect myself!",
            NPCAction.TALK: npc.dialogue[0] if npc.dialogue else "Greetings, traveler.",
            NPCAction.TRADE: "Would you like to trade?",
            NPCAction.HELP: "Let me help you.",
            NPCAction.STEAL: "What's this in your pocket?",
            NPCAction.SURRENDER: "I yield!",
            NPCAction.IDLE: "...",
        }
        return dialogue_map.get(action, "...")


# Smoke test
if __name__ == "__main__":
    from models import Item, ItemType  # noqa: F401

    print("Testing NPC Decision Trees")
    print("=" * 60)

    player = Player(name="Hero", hp=80, max_hp=100,
                    attack=15, defense=8, position=(0, 0), gold=100)

    behavior_mgr = NPCBehaviorManager()
    test_npcs = [
        NPC("Goblin", NPCType.ENEMY, hp=30, attack=10, defense=3,
            dialogue=["I'll destroy you!"]),
        NPC("Merchant", NPCType.MERCHANT, hp=50, attack=0, defense=5,
            dialogue=["Welcome to my shop!"]),
        NPC("Village Elder", NPCType.FRIENDLY, hp=40, attack=5, defense=5,
            dialogue=["Hello, young adventurer."]),
        NPC("Mysterious Thief", NPCType.NEUTRAL, hp=45, attack=12, defense=6,
            dialogue=["..."]),
        NPC("Dragon", NPCType.BOSS, hp=100, attack=25, defense=10,
            dialogue=["Face me, mortal!"]),
    ]

    game_state = {"turn_count": 1, "npc_met": False, "player_attacked": False}

    for npc in test_npcs:
        action, path, branches = behavior_mgr.get_npc_action_with_path(
            npc, player, game_state)
        print(f"\n{npc.name} ({npc.npc_type.value}) -> {action.value}")
        for i, node in enumerate(path):
            mark = "  ↳" if i > 0 else "  •"
            decision = ""
            if i < len(branches):
                decision = f"   [{'T' if branches[i] else 'F'}]"
            print(f"{mark} {node.label}{decision}")
