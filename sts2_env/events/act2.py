"""Act 2 specific events.

Events that only appear in Act 2 (act_index == 1), or require act > 0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sts2_env.cards.factory import create_card, eligible_registered_cards
from sts2_env.cards.enchantments import can_enchant_card
from sts2_env.cards.status import (
    make_debt,
    make_doubt,
    make_feeding_frenzy,
    make_normality,
    make_spore_mind,
)
from sts2_env.core.enums import CardRarity, CardType, PotionRarity
from sts2_env.events.shared import (
    _event_result_with_rewards,
    _downgrade_selected_cards,
    _event_potion_options,
    _obtain_random_relics,
    _roll_event_potion_id,
    _remove_selected_cards,
    _should_defer_event_rewards,
    _transform_n_cards,
    _transform_selected_cards,
    _upgrade_n_cards,
)
from sts2_env.potions.base import create_potion, normal_pool_models
from sts2_env.relics.base import RelicId, RelicRarity
from sts2_env.run.reward_objects import (
    AddCardsReward,
    GoldReward,
    CardReward,
    EnchantCardsReward,
    PotionReward,
    RelicReward,
    RemoveCardReward,
    TransformCardsReward,
)
from sts2_env.run.rewards import CARD_CREATION_SOURCE_OTHER
from sts2_env.run.rewards import (
    CardRewardGenerationOptions,
    generate_uniform_noncombat_reward_cards_with_options,
)
from sts2_env.run.crystal_sphere import (
    CrystalSphereItemType,
    CrystalSphereMinigame,
)
from sts2_env.run.events import EventModel, EventOption, EventResult, register_event

if TYPE_CHECKING:
    from sts2_env.run.run_state import PlayerState, RunState


# ── CrystalSphere ─────────────────────────────────────────────────────

class CrystalSphere(EventModel):
    """Uncover Future: Pay 51-99g for 3 Prophesize picks.
    Payment Plan: Gain Debt curse for 6 Prophesize picks.
    """

    event_id = "CrystalSphere"
    REQUIRED_GOLD = 100
    UNCOVER_FUTURE_BASE_COST = 50
    UNCOVER_FUTURE_RANDOM_MIN = 1
    UNCOVER_FUTURE_RANDOM_MAX = 50
    UNCOVER_FUTURE_PROPHESIZE_COUNT = 3
    PAYMENT_PLAN_PROPHESIZE_COUNT = 6

    def __init__(self) -> None:
        self._cost = self.UNCOVER_FUTURE_BASE_COST
        self._minigame: CrystalSphereMinigame | None = None
        self._awaiting_proceed = False

    def is_allowed(self, run_state: RunState) -> bool:
        return (
            all(player.gold >= self.REQUIRED_GOLD for player in run_state.players)
            and run_state.current_act_index > 0
        )

    def calculate_vars(self, run_state: RunState) -> None:
        extra = self.get_rng(run_state).next_int_exclusive(
            self.UNCOVER_FUTURE_RANDOM_MIN,
            self.UNCOVER_FUTURE_RANDOM_MAX,
        )
        self._cost = self.UNCOVER_FUTURE_BASE_COST + extra

    def before_event_started(self, run_state: RunState) -> None:
        # Registered event models are reused across runs; a previous game's
        # minigame must not make the next event's pay/debt screen look like a grid.
        self._minigame = None
        self._awaiting_proceed = False

    def generate_initial_options(self, run_state: RunState) -> list[EventOption]:
        self.ensure_vars_calculated(run_state)
        return [
            EventOption(
                "pay",
                f"Uncover Future ({self._cost}g)",
                f"{self.UNCOVER_FUTURE_PROPHESIZE_COUNT} Prophesize picks",
            ),
            EventOption(
                "debt",
                "Payment Plan",
                f"{self.PAYMENT_PLAN_PROPHESIZE_COUNT} picks, gain Debt curse",
            ),
        ]

    def choose(self, run_state: RunState, option_id: str) -> EventResult:
        if option_id == "pay":
            run_state.player.lose_gold(self._cost)
            return self._start_minigame(
                run_state,
                self.UNCOVER_FUTURE_PROPHESIZE_COUNT,
                f"Paid {self._cost}g.",
            )
        if option_id == "debt":
            run_state.player.add_card_instance_to_deck(make_debt())
            return self._start_minigame(
                run_state,
                self.PAYMENT_PLAN_PROPHESIZE_COUNT,
                "Gained Debt curse.",
            )
        if option_id.startswith("divine:") and self._minigame is not None:
            _, x_text, y_text = option_id.split(":")
            revealed = self._minigame.click(int(x_text), int(y_text))
            for item_index in revealed:
                item = self._minigame.items[item_index]
                if item.item_type is CrystalSphereItemType.CURSE:
                    run_state.player.add_card_instance_to_deck(make_doubt())
            if not self._minigame.is_finished:
                return EventResult(
                    finished=False,
                    description="Divined a Crystal Sphere cell.",
                    next_options=self._minigame_options(),
                )
            self._awaiting_proceed = True
            return EventResult(
                finished=False,
                description="Crystal Sphere divination complete.",
                next_options=[EventOption(
                    "proceed",
                    "Proceed",
                    metadata={"minigame_action": "proceed"},
                )],
                rewards={
                    "reward_objects": self._reward_objects(run_state),
                },
                preserve_reward_order=True,
            )
        if option_id == "proceed" and self._awaiting_proceed:
            self._awaiting_proceed = False
            return EventResult(
                finished=True,
                description="Finished consulting the Crystal Sphere.",
            )
        return EventResult(finished=True)

    @property
    def minigame(self) -> CrystalSphereMinigame | None:
        return self._minigame

    def _start_minigame(
        self,
        run_state: RunState,
        divinations: int,
        description: str,
    ) -> EventResult:
        self._minigame = CrystalSphereMinigame(
            self.get_rng(run_state),
            divinations,
        )
        return EventResult(
            finished=False,
            description=description,
            next_options=self._minigame_options(),
        )

    def _minigame_options(self) -> list[EventOption]:
        assert self._minigame is not None
        return [
            EventOption(
                option_id=f"divine:{x}:{y}",
                label=f"Divine ({x}, {y})",
                metadata={
                    "minigame_action": "divine_cell",
                    "x": x,
                    "y": y,
                    "entity_id": f"crystal-cell:{x}:{y}",
                    "affected_cell_ids": [
                        f"crystal-cell:{cell_x}:{cell_y}"
                        for cell_x, cell_y
                        in self._minigame.affected_cells(x, y)
                    ],
                },
            )
            for x, y in self._minigame.hidden_coordinates()
        ]

    def _reward_objects(self, run_state: RunState) -> list:
        assert self._minigame is not None
        player = run_state.player
        rewards = []
        for index in self._minigame.revealed_indices:
            item = self._minigame.items[index]
            if item.item_type is CrystalSphereItemType.CURSE:
                continue
            if item.item_type is CrystalSphereItemType.GOLD:
                rewards.append(GoldReward.fixed(
                    player.player_id,
                    30 if item.is_big_gold else 10,
                ))
                continue
            if item.item_type is CrystalSphereItemType.POTION:
                models = [
                    model
                    for model in normal_pool_models(
                        in_combat=False,
                        character_id=player.character_id,
                    )
                    if model.rarity.name.title() == item.rarity
                ]
                potion_id = self._minigame.rng.choice(models).potion_id
                reward = PotionReward(player.player_id, potion_id)
                reward.is_populated = True
                rewards.append(reward)
                continue
            if item.item_type is CrystalSphereItemType.RELIC:
                rarity = player._roll_relic_rarity(self._minigame.rng)
                relic_id = player.pull_next_relic_reward_id(rarity=rarity)
                reward = RelicReward(player.player_id, relic_id=relic_id)
                reward.is_populated = True
                rewards.append(reward)
                continue
            if item.item_type is CrystalSphereItemType.CARD_REWARD:
                rarity = CardRarity[item.rarity.upper()]
                options = CardRewardGenerationOptions(
                    num_cards=3,
                    use_default_character_pool=True,
                    card_creation_source=CARD_CREATION_SOURCE_OTHER,
                    card_pool_rarity_filter=rarity,
                    generation_context=None,
                )
                cards = generate_uniform_noncombat_reward_cards_with_options(
                    run_state,
                    options,
                    default_character_id=player.character_id,
                    rng=self._minigame.rng,
                )
                reward = CardReward(
                    player.player_id,
                    cards=cards,
                    generation_context=None,
                    card_creation_source=CARD_CREATION_SOURCE_OTHER,
                )
                reward.is_populated = True
                rewards.append(reward)
        return rewards


register_event(CrystalSphere())


# ── DollRoom ──────────────────────────────────────────────────────────

class DollRoom(EventModel):
    """Multi-page: Choose how to pick a doll (relic).

    Random: Get a random doll relic.
    Take Some Time: Take 5 damage, choose from 2 doll relics.
    Examine: Take 15 damage, choose from all 3 doll relics.
    Dolls: Daughter of the Wind, Mr Struggles, Bing Bong.
    """

    event_id = "DollRoom"
    TAKE_TIME_HP_LOSS = 5
    TAKE_TIME_DOLL_COUNT = 2
    EXAMINE_HP_LOSS = 15
    EXAMINE_DOLL_COUNT = 3

    def __init__(self) -> None:
        self._doll_choices: dict[str, str] = {}

    _DOLLS = (
        ("DAUGHTER_OF_THE_WIND", "Daughter of the Wind"),
        ("MR_STRUGGLES", "Mr Struggles"),
        ("BING_BONG", "Bing Bong"),
    )

    def is_allowed(self, run_state: RunState) -> bool:
        return run_state.current_act_index == 1

    def generate_initial_options(self, run_state: RunState) -> list[EventOption]:
        self._doll_choices = {}
        return [
            EventOption("random", "Random", "Get a random doll relic"),
            EventOption("take_time", "Take Some Time",
                         f"Take {self.TAKE_TIME_HP_LOSS} damage, choose from {self.TAKE_TIME_DOLL_COUNT} dolls"),
            EventOption("examine", "Examine",
                         f"Take {self.EXAMINE_HP_LOSS} damage, choose from {self.EXAMINE_DOLL_COUNT} dolls"),
        ]

    def choose(self, run_state: RunState, option_id: str) -> EventResult:
        if option_id == "random":
            relic_id, _ = run_state.rng.niche.choice(list(self._DOLLS))
            if _should_defer_event_rewards(run_state):
                return _event_result_with_rewards(
                    "Got a random doll relic.",
                    [RelicReward(run_state.player.player_id, relic_id=relic_id)],
                )
            run_state.player.obtain_relic(relic_id)
            return EventResult(finished=True,
                               description="Got a random doll relic.")
        if option_id == "take_time":
            run_state.player.lose_hp(self.TAKE_TIME_HP_LOSS)
            dolls = sorted(self._DOLLS)
            self.get_rng(run_state).shuffle(dolls)
            shown = dolls[: self.TAKE_TIME_DOLL_COUNT]
            self._doll_choices = {f"doll_{i + 1}": relic_id for i, (relic_id, _) in enumerate(shown)}
            return EventResult(
                finished=False,
                description="Took 5 damage, examining dolls.",
                next_options=[
                    EventOption(option_id, label, f"Gain {label} relic")
                    for option_id, (relic_id, label) in zip(self._doll_choices, shown)
                ],
            )
        if option_id == "examine":
            run_state.player.lose_hp(self.EXAMINE_HP_LOSS)
            dolls = sorted(self._DOLLS)
            self.get_rng(run_state).shuffle(dolls)
            self._doll_choices = {f"doll_{i + 1}": relic_id for i, (relic_id, _) in enumerate(dolls)}
            return EventResult(
                finished=False,
                description="Took 15 damage, all dolls revealed.",
                next_options=[
                    EventOption(option_id, label, f"Gain {label} relic")
                    for option_id, (relic_id, label) in zip(self._doll_choices, dolls)
                ],
            )
        relic_id = self._doll_choices.get(option_id)
        if relic_id is not None:
            if _should_defer_event_rewards(run_state):
                return _event_result_with_rewards(
                    "Obtained a doll relic.",
                    [RelicReward(run_state.player.player_id, relic_id=relic_id)],
                )
            run_state.player.obtain_relic(relic_id)
        return EventResult(finished=True,
                           description="Obtained a doll relic.")


register_event(DollRoom())


# ── EndlessConveyor ───────────────────────────────────────────────────

class EndlessConveyor(EventModel):
    """Conveyor belt sushi bar. Pay 35g per grab for random benefits.

    Dishes: Caviar (+4 Max HP), Clam Roll (Heal 10), Spicy Snappy (Upgrade 1),
    Jelly Liver (Transform 1), Fried Eel (Colorless card), Golden Fysh (+75g),
    Seapunk Salad (Feeding Frenzy card), Suspicious Condiment (Potion).
    Observe Chef: Upgrade 1 card (free).
    """

    event_id = "EndlessConveyor"
    OPTION_GRAB = "grab"
    OPTION_OBSERVE = "observe"
    OPTION_LEAVE = "leave"
    DISH_CAVIAR = "caviar"
    DISH_CLAM_ROLL = "clam_roll"
    DISH_SPICY_SNAPPY = "spicy_snappy"
    DISH_JELLY_LIVER = "jelly_liver"
    DISH_FRIED_EEL = "fried_eel"
    DISH_GOLDEN_FYSH = "golden_fysh"
    DISH_SEAPUNK_SALAD = "seapunk_salad"
    DISH_SUSPICIOUS_CONDIMENT = "suspicious_condiment"
    OBSERVE_CHEF_UPGRADE_COUNT = 1
    SPICY_SNAPPY_UPGRADE_COUNT = 1
    JELLY_LIVER_TRANSFORM_COUNT = 1
    GRAB_GOLD = 35
    REQUIRED_GOLD = 105
    GOLDEN_FYSH_GOLD = 75
    CLAM_ROLL_HEAL = 10
    CAVIAR_MAX_HP = 4
    FORCED_SEAPUNK_INTERVAL = 5
    CAVIAR_WEIGHT = 6.0
    SPICY_SNAPPY_WEIGHT = 3.0
    JELLY_LIVER_WEIGHT = 3.0
    FRIED_EEL_WEIGHT = 3.0
    SUSPICIOUS_CONDIMENT_WEIGHT = 3.0
    CLAM_ROLL_WEIGHT = 6.0
    GOLDEN_FYSH_WEIGHT = 1.0
    BASE_DISH_WEIGHTS = (
        (DISH_CAVIAR, CAVIAR_WEIGHT),
        (DISH_SPICY_SNAPPY, SPICY_SNAPPY_WEIGHT),
        (DISH_JELLY_LIVER, JELLY_LIVER_WEIGHT),
        (DISH_FRIED_EEL, FRIED_EEL_WEIGHT),
    )

    def __init__(self) -> None:
        self._grabs = 0
        self._current_dish: str = ""
        self._last_dish: str = ""

    def is_allowed(self, run_state: RunState) -> bool:
        return all(player.gold >= self.REQUIRED_GOLD for player in run_state.players)

    def generate_initial_options(self, run_state: RunState) -> list[EventOption]:
        self._grabs = 0
        self._roll_dish(run_state)
        return [
            self._grab_option(run_state, initial=True),
            EventOption(self.OPTION_OBSERVE, "Observe Chef",
                         "Upgrade 1 random card (free)"),
        ]

    def choose(self, run_state: RunState, option_id: str) -> EventResult:
        if option_id == self.OPTION_OBSERVE:
            _upgrade_n_cards(run_state, self.OBSERVE_CHEF_UPGRADE_COUNT, rng=self.get_rng(run_state))
            return EventResult(finished=True,
                               description="Observed chef, upgraded a random card.")
        if option_id == self.OPTION_LEAVE:
            return EventResult(finished=True, description="Left the conveyor.")

        if run_state.player.gold >= self.GRAB_GOLD:
            if self._current_dish != self.DISH_GOLDEN_FYSH:
                run_state.player.lose_gold(self.GRAB_GOLD)
            should_roll_after_dish = (
                self._current_dish != self.DISH_JELLY_LIVER
                or not _should_defer_event_rewards(run_state)
            )
            self._apply_dish(run_state)
            grabbed_dish_number = self._grabs
            if should_roll_after_dish:
                self._roll_dish(run_state)

            next_opts = [
                self._grab_option(run_state, initial=False),
                EventOption(self.OPTION_LEAVE, "Leave", "Done eating"),
            ]

            return EventResult(
                finished=False,
                description=f"Grabbed dish #{grabbed_dish_number} from the conveyor.",
                next_options=next_opts,
            )
        return EventResult(finished=True, description="Cannot afford more food.")

    def _grab_option(self, run_state: RunState, *, initial: bool) -> EventOption:
        label = f"Grab Something ({self.GRAB_GOLD}g)" if initial else f"Grab Another ({self.GRAB_GOLD}g)"
        if run_state.player.gold >= self.GRAB_GOLD:
            return EventOption(self.OPTION_GRAB, label, "Random dish")
        return EventOption(self.OPTION_GRAB, "Locked", "Cannot afford another dish", enabled=False)

    def _roll_dish(self, run_state: RunState) -> None:
        self._grabs += 1
        if self._grabs % self.FORCED_SEAPUNK_INTERVAL == 0:
            self._last_dish = self.DISH_SEAPUNK_SALAD
            self._current_dish = self.DISH_SEAPUNK_SALAD
            return
        weighted = list(self.BASE_DISH_WEIGHTS)
        if len(run_state.player.held_potions()) < run_state.player.max_potion_slots:
            weighted.append((self.DISH_SUSPICIOUS_CONDIMENT, self.SUSPICIOUS_CONDIMENT_WEIGHT))
        if run_state.player.current_hp != run_state.player.max_hp:
            weighted.append((self.DISH_CLAM_ROLL, self.CLAM_ROLL_WEIGHT))
        if self._grabs > 1:
            weighted.append((self.DISH_GOLDEN_FYSH, self.GOLDEN_FYSH_WEIGHT))
        weighted = [(dish, weight) for dish, weight in weighted if dish != self._last_dish]
        total = sum(weight for _, weight in weighted)
        roll = self.get_rng(run_state).next_float() * total
        cumulative = 0.0
        for dish, weight in weighted:
            cumulative += weight
            if roll < cumulative:
                self._last_dish = dish
                self._current_dish = dish
                return
        self._last_dish = weighted[-1][0]
        self._current_dish = weighted[-1][0]

    def _apply_dish(self, run_state: RunState) -> None:
        dish = self._current_dish
        if dish == self.DISH_CAVIAR:
            run_state.player.gain_max_hp(self.CAVIAR_MAX_HP)
        elif dish == self.DISH_CLAM_ROLL:
            run_state.player.heal(self.CLAM_ROLL_HEAL)
        elif dish == self.DISH_SPICY_SNAPPY:
            _upgrade_n_cards(run_state, self.SPICY_SNAPPY_UPGRADE_COUNT, rng=self.get_rng(run_state))
        elif dish == self.DISH_JELLY_LIVER:
            if _should_defer_event_rewards(run_state):
                candidates = run_state.player.transformable_deck_cards()
                run_state.pending_rewards.append(
                    TransformCardsReward(
                        run_state.player.player_id,
                        count=min(self.JELLY_LIVER_TRANSFORM_COUNT, len(candidates)),
                        cards=candidates,
                        rng_override=self.get_rng(run_state),
                        after_selected=lambda: self._roll_dish(run_state),
                    )
                )
            else:
                _transform_n_cards(run_state, self.JELLY_LIVER_TRANSFORM_COUNT, rng=self.get_rng(run_state))
        elif dish == self.DISH_FRIED_EEL:
            reward = CardReward(
                run_state.player.player_id,
                option_count=1,
                generation_context=None,
                roll_upgrade=False,
                use_default_character_pool=False,
                include_colorless=True,
                card_creation_source=CARD_CREATION_SOURCE_OTHER,
            )
            reward.populate(run_state, None)
            if reward.cards:
                run_state.player.add_card_instance_to_deck(reward.cards[0], source=reward)
        elif dish == self.DISH_GOLDEN_FYSH:
            run_state.player.gain_gold(self.GOLDEN_FYSH_GOLD)
        elif dish == self.DISH_SEAPUNK_SALAD:
            if _should_defer_event_rewards(run_state):
                run_state.pending_rewards.append(AddCardsReward(run_state.player.player_id, [make_feeding_frenzy()]))
            else:
                run_state.player.add_card_instance_to_deck(make_feeding_frenzy())
        elif dish == self.DISH_SUSPICIOUS_CONDIMENT:
            potion_id = _roll_event_potion_id(run_state)
            if potion_id is not None:
                run_state.pending_rewards.append(PotionReward(run_state.player.player_id, potion_id=potion_id))


register_event(EndlessConveyor())


# ── FakeMerchant ──────────────────────────────────────────────────────

class FakeMerchant(EventModel):
    """Custom merchant event: buy fake relics for 50g each.

    Can throw Foul Potion to start a fight for real rewards.
    """

    event_id = "FakeMerchant"
    is_shared = True
    FAKE_RELIC_COST = 50
    MIN_ENTRY_GOLD = 100
    INVENTORY_SIZE = 6
    FOUL_POTION_ID = "FoulPotion"
    FIGHT_REWARD_RUG_ID = RelicId.FAKE_MERCHANTS_RUG.name
    OPTION_BUY = "buy"
    OPTION_THROW_FOUL = "throw_foul"
    OPTION_LEAVE = "leave"
    BUY_OPTION_PREFIX = "buy_"

    _INVENTORY_RELICS = (
        RelicId.FAKE_ANCHOR.name,
        RelicId.FAKE_BLOOD_VIAL.name,
        RelicId.FAKE_HAPPY_FLOWER.name,
        RelicId.FAKE_LEES_WAFFLE.name,
        RelicId.FAKE_MANGO.name,
        RelicId.FAKE_ORICHALCUM.name,
        RelicId.FAKE_SNECKO_EYE.name,
        RelicId.FAKE_STRIKE_DUMMY.name,
        RelicId.FAKE_VENERABLE_TEA_SET.name,
    )

    def __init__(self) -> None:
        self._inventories: dict[int, list[str]] = {}

    def _inventory_for(self, run_state: RunState) -> list[str]:
        key = id(run_state)
        inventory = self._inventories.get(key)
        if inventory is None:
            pool = list(self._INVENTORY_RELICS)
            self.get_rng(run_state).shuffle(pool)
            inventory = pool[: self.INVENTORY_SIZE]
            self._inventories[key] = inventory
        return inventory

    def _has_foul_potion(self, run_state: RunState) -> bool:
        return any(
            potion.potion_id == self.FOUL_POTION_ID
            for potion in run_state.player.held_potions()
        )

    def _fake_relic_option(self, index: int, relic_id: str) -> EventOption:
        return EventOption(
            f"{self.BUY_OPTION_PREFIX}{index}",
            relic_id.replace("_", " ").title(),
            "Buy this fake relic",
            enabled=True,
        )

    def _post_buy_options(self, run_state: RunState, inventory: list[str]) -> list[EventOption]:
        options: list[EventOption] = []
        if inventory:
            options.extend(
                self._fake_relic_option(index, relic_id)
                for index, relic_id in enumerate(inventory)
            )
        if self._has_foul_potion(run_state):
            options.append(EventOption(self.OPTION_THROW_FOUL, "Throw Foul Potion", "Fight for real rewards"))
        options.append(EventOption(self.OPTION_LEAVE, "Leave", "Leave the merchant"))
        return options

    def is_allowed(self, run_state: RunState) -> bool:
        if run_state.current_act_index < 1:
            return False
        if len(run_state.players) > 1:
            return False
        return run_state.player.gold >= self.MIN_ENTRY_GOLD or self._has_foul_potion(run_state)

    def before_event_started(self, run_state: RunState) -> None:
        run_state.player.can_remove_potions = False

    def on_event_finished(self, run_state: RunState) -> None:
        run_state.player.can_remove_potions = True

    def generate_initial_options(self, run_state: RunState) -> list[EventOption]:
        self._inventories[id(run_state)] = self._inventory_for(run_state)
        options = [
            EventOption(
                self.OPTION_BUY,
                f"Buy a Relic ({self.FAKE_RELIC_COST}g)",
                "Purchase a fake relic",
            ),
        ]
        if self._has_foul_potion(run_state):
            options.append(EventOption(self.OPTION_THROW_FOUL, "Throw Foul Potion",
                                       "Fight for real rewards"))
        options.append(EventOption(self.OPTION_LEAVE, "Leave", "Leave the merchant"))
        return options

    def choose(self, run_state: RunState, option_id: str) -> EventResult:
        inventory = self._inventory_for(run_state)
        if option_id == self.OPTION_BUY:
            if not inventory:
                return EventResult(finished=True, description="Nothing happened.")
            return EventResult(
                finished=False,
                description="Choose a fake relic to buy.",
                next_options=self._post_buy_options(run_state, inventory),
            )
        if option_id.startswith(self.BUY_OPTION_PREFIX):
            try:
                idx = int(option_id.removeprefix(self.BUY_OPTION_PREFIX))
            except ValueError:
                return EventResult(finished=True, description="Nothing happened.")
            if 0 <= idx < len(inventory) and run_state.player.gold >= self.FAKE_RELIC_COST:
                run_state.player.lose_gold(self.FAKE_RELIC_COST)
                purchased = inventory.pop(idx)
                if _should_defer_event_rewards(run_state):
                    return EventResult(
                        finished=False,
                        description=f"Bought a fake relic for {self.FAKE_RELIC_COST}g.",
                        next_options=self._post_buy_options(run_state, inventory),
                        rewards={"reward_objects": [RelicReward(run_state.player.player_id, relic_id=purchased)]},
                    )
                run_state.player.obtain_relic(purchased)
                return EventResult(
                    finished=False,
                    description=f"Bought a fake relic for {self.FAKE_RELIC_COST}g.",
                    next_options=self._post_buy_options(run_state, inventory),
                )
            return EventResult(
                finished=False,
                description="Could not buy that relic.",
                next_options=self._post_buy_options(run_state, inventory),
            )
        if option_id == self.OPTION_THROW_FOUL:
            for idx, potion in enumerate(list(run_state.player.held_potions())):
                if potion.potion_id == self.FOUL_POTION_ID:
                    run_state.player.remove_potion(potion.slot_index)
                    break
            stocked_fake_relics = [self.FIGHT_REWARD_RUG_ID, *inventory]
            rewards = [RelicReward(run_state.player.player_id, relic_id=relic_id) for relic_id in stocked_fake_relics]
            return EventResult(
                finished=True,
                description="Threw Foul Potion and started a fight for real rewards.",
                rewards={"reward_objects": rewards},
                event_combat_setup="fake_merchant",
            )
        return EventResult(finished=True, description="Left the fake merchant.")


register_event(FakeMerchant())


# ── FieldOfManSizedHoles ──────────────────────────────────────────────

class FieldOfManSizedHoles(EventModel):
    """Resist: Remove 2 cards, gain Normality curse.
    Enter Your Hole: Enchant a card with Perfect Fit.
    """

    event_id = "FieldOfManSizedHoles"

    def is_allowed(self, run_state: RunState) -> bool:
        return all(
            any(can_enchant_card(card, "PerfectFit") for card in player.deck)
            for player in run_state.players
        )

    def generate_initial_options(self, run_state: RunState) -> list[EventOption]:
        return [
            EventOption("resist", "Resist",
                         "Remove 2 cards, gain Normality curse"),
            EventOption("enter", "Enter Your Hole",
                         "Enchant a card with Perfect Fit"),
        ]

    def choose(self, run_state: RunState, option_id: str) -> EventResult:
        if option_id == "resist":
            candidates = run_state.player.removable_deck_cards()
            if _should_defer_event_rewards(run_state):
                return _event_result_with_rewards(
                    "Removed 2 cards, gained Normality curse.",
                    [
                        RemoveCardReward(
                            run_state.player.player_id,
                            count=min(2, len(candidates)),
                            cards=candidates,
                            after_selected=lambda: run_state.player.add_card_instance_to_deck(make_normality()),
                        ),
                    ],
                )
            return self.request_card_choice(
                prompt="Choose 2 cards to remove",
                cards=candidates,
                source_pile="deck",
                resolver=lambda selected: (
                    _remove_selected_cards(selected, run_state),
                    run_state.player.add_card_instance_to_deck(make_normality()),
                    EventResult(finished=True, description="Removed 2 cards, gained Normality curse."),
                )[-1],
                allow_skip=False,
                min_count=min(2, len(candidates)),
                max_count=min(2, len(candidates)),
                description="Choose 2 cards to remove.",
            )
        candidates = [card for card in run_state.player.deck if can_enchant_card(card, "PerfectFit")]
        if not candidates:
            return EventResult(finished=True, description="No valid card for Perfect Fit.")
        if _should_defer_event_rewards(run_state):
            return _event_result_with_rewards(
                "Enchanted a card with Perfect Fit.",
                [
                    EnchantCardsReward(
                        run_state.player.player_id,
                        enchantment="PerfectFit",
                        amount=1,
                        count=1,
                        cards=candidates,
                    )
                ],
            )
        return self.request_card_choice(
            prompt="Choose a card to enchant with Perfect Fit",
            cards=candidates,
            source_pile="deck",
            resolver=lambda selected: (
                selected and selected[0].add_enchantment("PerfectFit", 1),
                EventResult(finished=True, description="Enchanted a card with Perfect Fit."),
            )[-1],
            description="Choose a card to enchant.",
        )


register_event(FieldOfManSizedHoles())


# ── JungleMazeAdventure ──────────────────────────────────────────────

class JungleMazeAdventure(EventModel):
    """Solo Quest: Take 18 damage, gain ~150 gold.
    Join Forces: Gain ~50 gold (safe).
    """

    event_id = "JungleMazeAdventure"
    is_shared = True
    SOLO_GOLD = 150
    SOLO_HP_LOSS = 18
    JOIN_FORCES_GOLD = 50
    GOLD_VARIANCE_MIN = -15
    GOLD_VARIANCE_MAX = 15

    def __init__(self) -> None:
        self._solo_gold = self.SOLO_GOLD
        self._join_gold = self.JOIN_FORCES_GOLD

    def calculate_vars(self, run_state: RunState) -> None:
        rng = self.get_rng(run_state)
        self._solo_gold = self.SOLO_GOLD + rng.next_float_range(
            self.GOLD_VARIANCE_MIN,
            self.GOLD_VARIANCE_MAX,
        )
        self._join_gold = self.JOIN_FORCES_GOLD + rng.next_float_range(
            self.GOLD_VARIANCE_MIN,
            self.GOLD_VARIANCE_MAX,
        )

    def generate_initial_options(self, run_state: RunState) -> list[EventOption]:
        self.ensure_vars_calculated(run_state)
        return [
            EventOption(
                "solo",
                "Solo Quest",
                f"Take {self.SOLO_HP_LOSS} damage, gain {self._solo_gold} gold",
            ),
            EventOption(
                "join",
                "Join Forces",
                f"Gain {self._join_gold} gold",
            ),
        ]

    def choose(self, run_state: RunState, option_id: str) -> EventResult:
        if option_id == "solo":
            run_state.player.lose_hp(self.SOLO_HP_LOSS)
            run_state.player.gain_gold(self._solo_gold)
            return EventResult(finished=True,
                               description=(
                                   f"Took {self.SOLO_HP_LOSS} damage, "
                                   f"gained {self._solo_gold} gold."
                               ))
        run_state.player.gain_gold(self._join_gold)
        return EventResult(finished=True,
                           description=f"Gained {self._join_gold} gold.")


register_event(JungleMazeAdventure())


# ── LuminousChoir ─────────────────────────────────────────────────────

class LuminousChoir(EventModel):
    """Reach Into the Flesh: Remove 2 cards, gain Spore Mind curse.
    Offer Tribute: Pay ~100-149g, gain a relic.
    """

    event_id = "LuminousChoir"
    ENTRY_GOLD_COST = 149
    TRIBUTE_GOLD_VARIANCE_ROLL = 50

    def __init__(self) -> None:
        self._cost = self.ENTRY_GOLD_COST

    def calculate_vars(self, run_state: RunState) -> None:
        self._cost = self.ENTRY_GOLD_COST - self.get_rng(run_state).next_int_exclusive(
            0,
            self.TRIBUTE_GOLD_VARIANCE_ROLL,
        )

    def is_allowed(self, run_state: RunState) -> bool:
        return all(
            player.gold >= self.ENTRY_GOLD_COST and player.has_available_relics()
            for player in run_state.players
        )

    def generate_initial_options(self, run_state: RunState) -> list[EventOption]:
        self.ensure_vars_calculated(run_state)
        return [
            EventOption("reach", "Reach Into the Flesh",
                         "Remove 2 cards, gain Spore Mind curse"),
            EventOption(
                "tribute",
                f"Offer Tribute ({self._cost}g)" if run_state.player.gold >= self._cost else "Offer Tribute",
                "Gain a relic",
                enabled=run_state.player.gold >= self._cost and run_state.player.has_available_relics(),
            ),
        ]

    def choose(self, run_state: RunState, option_id: str) -> EventResult:
        if option_id == "reach":
            candidates = run_state.player.removable_deck_cards()
            if _should_defer_event_rewards(run_state):
                return _event_result_with_rewards(
                    "Removed 2 cards, gained Spore Mind curse.",
                    [
                        RemoveCardReward(
                            run_state.player.player_id,
                            count=min(2, len(candidates)),
                            cards=candidates,
                            after_selected=lambda: run_state.player.add_card_instance_to_deck(make_spore_mind()),
                        ),
                    ],
                )
            return self.request_card_choice(
                prompt="Choose 2 cards to remove",
                cards=candidates,
                source_pile="deck",
                resolver=lambda selected: (
                    _remove_selected_cards(selected, run_state),
                    run_state.player.add_card_instance_to_deck(make_spore_mind()),
                    EventResult(finished=True, description="Removed 2 cards, gained Spore Mind curse."),
                )[-1],
                min_count=min(2, len(candidates)),
                max_count=min(2, len(candidates)),
                description="Choose 2 cards to remove.",
            )
        run_state.player.lose_gold(self._cost)
        _obtain_random_relics(run_state, 1)
        return EventResult(finished=True,
                           description=f"Paid {self._cost}g, gained a relic.")


register_event(LuminousChoir())


# ── MorphicGrove ──────────────────────────────────────────────────────

class MorphicGrove(EventModel):
    """Group: Lose 100g, transform 2 cards. Loner: Gain 5 Max HP."""

    event_id = "MorphicGrove"

    def is_allowed(self, run_state: RunState) -> bool:
        return all(player.gold >= 100 for player in run_state.players)

    def generate_initial_options(self, run_state: RunState) -> list[EventOption]:
        return [
            EventOption("group", "Group",
                         "Lose 100g, transform 2 cards"),
            EventOption("loner", "Loner", "Gain 5 Max HP"),
        ]

    def choose(self, run_state: RunState, option_id: str) -> EventResult:
        if option_id == "group":
            run_state.player.lose_gold(100)
            candidates = run_state.player.transformable_deck_cards()
            if _should_defer_event_rewards(run_state):
                return _event_result_with_rewards(
                    "Lost 100g, transformed 2 cards.",
                    [
                        TransformCardsReward(
                            run_state.player.player_id,
                            count=min(2, len(candidates)),
                            cards=candidates,
                        )
                    ],
                )
            return self.request_card_choice(
                prompt="Choose 2 cards to transform",
                cards=candidates,
                source_pile="deck",
                resolver=lambda selected: (
                    _transform_selected_cards(selected, run_state),
                    EventResult(finished=True, description="Lost 100g, transformed 2 cards."),
                )[-1],
                min_count=min(2, len(candidates)),
                max_count=min(2, len(candidates)),
                description="Choose 2 cards to transform.",
            )
        run_state.player.gain_max_hp(5)
        return EventResult(finished=True, description="Gained 5 Max HP.")


register_event(MorphicGrove())


# ── PotionCourier ────────────────────────────────────────────────────

class PotionCourier(EventModel):
    """Grab Potions: Gain 3 Foul Potions. Ransack: Gain 1 uncommon potion."""

    event_id = "PotionCourier"
    FOUL_POTION_ID = "FoulPotion"
    FOUL_POTION_COUNT = 3
    RANSACK_POTION_COUNT = 1

    def is_allowed(self, run_state: RunState) -> bool:
        return run_state.current_act_index > 0

    def generate_initial_options(self, run_state: RunState) -> list[EventOption]:
        return [
            EventOption(
                "grab",
                "Grab Potions",
                f"Gain {self.FOUL_POTION_COUNT} Foul Potions",
            ),
            EventOption(
                "ransack",
                "Ransack",
                f"Gain {self.RANSACK_POTION_COUNT} uncommon potion",
            ),
        ]

    def choose(self, run_state: RunState, option_id: str) -> EventResult:
        if option_id == "grab":
            return EventResult(
                finished=True,
                description=f"Gained {self.FOUL_POTION_COUNT} Foul Potions.",
                rewards={
                    "reward_objects": [
                        PotionReward(run_state.player.player_id, potion_id=self.FOUL_POTION_ID)
                        for _ in range(self.FOUL_POTION_COUNT)
                    ]
                },
            )
        uncommon_models = [model for model in _event_potion_options(run_state) if model.rarity == PotionRarity.UNCOMMON]
        if uncommon_models:
            model = run_state.rng.rewards.choice(uncommon_models)
            return EventResult(
                finished=True,
                description="Gained an uncommon potion.",
                rewards={"reward_objects": [PotionReward(run_state.player.player_id, potion_id=model.potion_id)]},
            )
        return EventResult(finished=True, description="Gained an uncommon potion.")


register_event(PotionCourier())


# ── RanwidTheElder ────────────────────────────────────────────────────

class RanwidTheElder(EventModel):
    """Give Potion: Lose a potion, gain a relic.
    Give Gold: Lose 100g, gain a relic.
    Give Relic: Lose a relic, gain 2 relics.
    """

    event_id = "RanwidTheElder"
    ENTRY_GOLD_COST = 100

    def __init__(self) -> None:
        self._potion_slot: int | None = None
        self._relic_id: str | None = None

    @staticmethod
    def _tradable_relics(player: PlayerState) -> list[str]:
        return player.tradable_relics()

    def is_allowed(self, run_state: RunState) -> bool:
        return (
            run_state.current_act_index > 0
            and all(player.gold >= self.ENTRY_GOLD_COST for player in run_state.players)
            and all(len(player.held_potions()) > 0 for player in run_state.players)
            and all(bool(self._tradable_relics(player)) for player in run_state.players)
        )

    def before_event_started(self, run_state: RunState) -> None:
        run_state.player.can_remove_potions = False

    def on_event_finished(self, run_state: RunState) -> None:
        run_state.player.can_remove_potions = True

    def generate_initial_options(self, run_state: RunState) -> list[EventOption]:
        tradable_relics = self._tradable_relics(run_state.player)
        held_potions = run_state.player.held_potions()
        rng = self.get_rng(run_state)
        chosen_potion = rng.choice(held_potions) if held_potions else None
        chosen_relic = rng.choice(tradable_relics) if tradable_relics else None
        self._potion_slot = chosen_potion.slot_index if chosen_potion is not None else None
        self._relic_id = chosen_relic
        return [
            EventOption("potion", "Give a Potion", "Lose a potion, gain a relic", enabled=bool(run_state.player.held_potions())),
            EventOption("gold", f"Give {self.ENTRY_GOLD_COST} Gold", f"Lose {self.ENTRY_GOLD_COST}g, gain a relic"),
            EventOption("relic", "Give a Relic", "Lose a relic, gain 2 relics", enabled=bool(tradable_relics)),
        ]

    def choose(self, run_state: RunState, option_id: str) -> EventResult:
        if option_id == "potion":
            if self._potion_slot is not None:
                run_state.player.remove_potion(self._potion_slot)
            if _should_defer_event_rewards(run_state):
                return _event_result_with_rewards(
                    "Gave a potion, gained a relic.",
                    _roll_random_relic_rewards(run_state, 1),
                )
            _obtain_random_relics(run_state, 1)
            return EventResult(finished=True,
                               description="Gave a potion, gained a relic.")
        if option_id == "gold":
            run_state.player.lose_gold(self.ENTRY_GOLD_COST)
            if _should_defer_event_rewards(run_state):
                return _event_result_with_rewards(
                    f"Paid {self.ENTRY_GOLD_COST}g, gained a relic.",
                    _roll_random_relic_rewards(run_state, 1),
                )
            _obtain_random_relics(run_state, 1)
            return EventResult(finished=True,
                               description=f"Paid {self.ENTRY_GOLD_COST}g, gained a relic.")
        tradable_relics = self._tradable_relics(run_state.player)
        if self._relic_id is not None and self._relic_id in tradable_relics:
            to_remove = self._relic_id
            if to_remove in run_state.player.relics:
                index = run_state.player.relics.index(to_remove)
                run_state.player.relics.pop(index)
                if index < len(run_state.player.relic_objects):
                    run_state.player.relic_objects.pop(index)
        if _should_defer_event_rewards(run_state):
            return _event_result_with_rewards(
                "Gave a relic, gained 2 relics.",
                _roll_random_relic_rewards(run_state, 2),
            )
        _obtain_random_relics(run_state, 2)
        return EventResult(finished=True,
                           description="Gave a relic, gained 2 relics.")


register_event(RanwidTheElder())


# ── RelicTrader ───────────────────────────────────────────────────────

class RelicTrader(EventModel):
    """Trade an owned relic for a new random relic (up to 3 trade options)."""

    event_id = "RelicTrader"

    def __init__(self) -> None:
        self._owned_relic_choices: list[str] = []
        self._new_relic_choices: list[str] = []

    @staticmethod
    def _tradable_relics(player: PlayerState) -> list[str]:
        return player.tradable_relics()

    def is_allowed(self, run_state: RunState) -> bool:
        return (
            run_state.current_act_index > 0
            and all(len(self._tradable_relics(player)) >= 5 for player in run_state.players)
        )

    def generate_initial_options(self, run_state: RunState) -> list[EventOption]:
        owned = sorted(self._tradable_relics(run_state.player))
        self.get_rng(run_state).shuffle(owned)
        self._owned_relic_choices = owned[:3]
        self._new_relic_choices = []
        for _ in self._owned_relic_choices:
            reward = RelicReward(run_state.player.player_id)
            reward.populate(run_state, None)
            self._new_relic_choices.append(reward.relic_id or "CIRCLET")
        options = []
        for i in range(len(self._owned_relic_choices)):
            options.append(EventOption(f"trade_{i}", f"Trade Relic {i+1}",
                                        "Swap an owned relic for a new one"))
        return options

    def choose(self, run_state: RunState, option_id: str) -> EventResult:
        if option_id.startswith("trade_"):
            index = int(option_id.split("_")[1])
            if 0 <= index < len(self._owned_relic_choices):
                old = self._owned_relic_choices[index]
                new = self._new_relic_choices[index]
                if old in run_state.player.relics:
                    old_index = run_state.player.relics.index(old)
                    run_state.player.relics.pop(old_index)
                    if old_index < len(run_state.player.relic_objects):
                        run_state.player.relic_objects.pop(old_index)
                if _should_defer_event_rewards(run_state):
                    return _event_result_with_rewards(
                        "Traded a relic for a new one.",
                        [RelicReward(run_state.player.player_id, relic_id=new)],
                    )
                run_state.player.obtain_relic(new)
        return EventResult(finished=True,
                           description="Traded a relic for a new one.")


register_event(RelicTrader())


# ── SlipperyBridge ────────────────────────────────────────────────────

class SlipperyBridge(EventModel):
    """Multi-page: Overcome (lose a random card) or Hold On (take escalating damage).

    Hold On damage starts at 3 and increases by 1 each time.
    Each Hold On rerolls which card would be lost.
    """

    event_id = "SlipperyBridge"

    def __init__(self) -> None:
        self._hold_ons = 0
        self._random_card_to_lose = None

    def is_allowed(self, run_state: RunState) -> bool:
        return (
            run_state.total_floor > 6
            and all(any(card.is_removable for card in player.deck) for player in run_state.players)
        )

    def _roll_random_card_to_lose(self, run_state: RunState) -> None:
        if self._random_card_to_lose is None:
            candidates = [
                card for card in run_state.player.deck
                if card.rarity != CardRarity.BASIC and card.is_removable
            ]
        else:
            candidates = [
                card for card in run_state.player.deck
                if card.card_id != self._random_card_to_lose.card_id and card.is_removable
            ]
        if not candidates:
            candidates = [card for card in run_state.player.deck if card.is_removable]
        self._random_card_to_lose = self.get_rng(run_state).choice(candidates) if candidates else None

    def generate_initial_options(self, run_state: RunState) -> list[EventOption]:
        self._hold_ons = 0
        self._random_card_to_lose = None
        self._roll_random_card_to_lose(run_state)
        dmg = 3
        return [
            EventOption("overcome", "Overcome", "Lose a random card"),
            EventOption("hold_on", "Hold On", f"Take {dmg} damage, reroll card"),
        ]

    def choose(self, run_state: RunState, option_id: str) -> EventResult:
        if option_id == "overcome":
            if self._random_card_to_lose is None:
                self._roll_random_card_to_lose(run_state)
            if self._random_card_to_lose is not None:
                _remove_selected_cards([self._random_card_to_lose], run_state)
            return EventResult(finished=True,
                               description="Lost a random card to cross the bridge.")

        # hold_on
        dmg = 3 + self._hold_ons
        run_state.player.lose_hp(dmg)
        self._hold_ons += 1
        self._roll_random_card_to_lose(run_state)
        next_dmg = 3 + self._hold_ons

        return EventResult(
            finished=False,
            description=f"Took {dmg} damage, holding on.",
            next_options=[
                EventOption("overcome", "Overcome", "Lose a random card"),
                EventOption("hold_on", "Hold On",
                             f"Take {next_dmg} damage, reroll card"),
            ],
        )


register_event(SlipperyBridge())


# ── SpiralingWhirlpool ────────────────────────────────────────────────

class SpiralingWhirlpool(EventModel):
    """Observe: Enchant a card with Spiral. Drink: Heal 33% of Max HP."""

    event_id = "SpiralingWhirlpool"
    HEAL_MULTIPLIER = 0.33

    @classmethod
    def heal_amount(cls, run_state: RunState) -> int:
        return int(run_state.player.max_hp * cls.HEAL_MULTIPLIER)

    def is_allowed(self, run_state: RunState) -> bool:
        return all(
            any(can_enchant_card(card, "Spiral") for card in player.deck)
            for player in run_state.players
        )

    def generate_initial_options(self, run_state: RunState) -> list[EventOption]:
        heal = self.heal_amount(run_state)
        return [
            EventOption("observe", "Observe the Spiral",
                         "Enchant a card with Spiral"),
            EventOption("drink", "Drink", f"Heal {heal} HP"),
        ]

    def choose(self, run_state: RunState, option_id: str) -> EventResult:
        if option_id == "observe":
            candidates = [card for card in run_state.player.deck if can_enchant_card(card, "Spiral")]
            if not candidates:
                return EventResult(finished=True, description="No card could be enchanted with Spiral.")
            if _should_defer_event_rewards(run_state):
                return _event_result_with_rewards(
                    "Enchanted a card with Spiral.",
                    [
                        EnchantCardsReward(
                            run_state.player.player_id,
                            enchantment="Spiral",
                            amount=1,
                            count=1,
                            cards=candidates,
                        )
                    ],
                )
            return self.request_card_choice(
                prompt="Choose a card to enchant with Spiral",
                cards=candidates,
                source_pile="deck",
                resolver=lambda selected: (
                    selected and selected[0].add_enchantment("Spiral", 1),
                    EventResult(finished=True, description="Enchanted a card with Spiral."),
                )[-1],
                description="Choose a card to enchant.",
            )
        heal = self.heal_amount(run_state)
        run_state.player.heal(heal)
        return EventResult(finished=True, description=f"Healed {heal} HP.")


register_event(SpiralingWhirlpool())


# ── StoneOfAllTime ────────────────────────────────────────────────────

STONE_OF_ALL_TIME_DRINK_MAX_HP_GAIN = 10
STONE_OF_ALL_TIME_PUSH_HP_LOSS = 6
STONE_OF_ALL_TIME_VIGOROUS_AMOUNT = 8
STONE_OF_ALL_TIME_POST_CHOICE_RNG_BOUND = 100


class StoneOfAllTime(EventModel):
    """Lift: Discard a potion, gain 10 Max HP.
    Push: Take 6 damage, enchant a card with Vigorous +8.
    """

    event_id = "StoneOfAllTime"

    def __init__(self) -> None:
        self._lift_potion_slot: int | None = None

    def is_allowed(self, run_state: RunState) -> bool:
        return (
            run_state.current_act_index == 1
            and all(len(player.held_potions()) >= 1 for player in run_state.players)
        )

    def before_event_started(self, run_state: RunState) -> None:
        run_state.player.can_remove_potions = False

    def on_event_finished(self, run_state: RunState) -> None:
        run_state.player.can_remove_potions = True

    def _consume_post_choice_rng(self, run_state: RunState) -> None:
        self.get_rng(run_state).next_int(0, STONE_OF_ALL_TIME_POST_CHOICE_RNG_BOUND - 1)

    def generate_initial_options(self, run_state: RunState) -> list[EventOption]:
        held = run_state.player.held_potions()
        lift_potion = self.get_rng(run_state).choice(held) if held else None
        self._lift_potion_slot = lift_potion.slot_index if lift_potion is not None else None
        has_lift = lift_potion is not None
        has_push = any(can_enchant_card(card, "Vigorous") for card in run_state.player.deck)
        return [
            EventOption("lift", "Lift", "Discard a potion, gain 10 Max HP", enabled=has_lift),
            EventOption("push", "Push", "Take 6 damage, enchant a card with Vigorous +8", enabled=has_push),
        ]

    def choose(self, run_state: RunState, option_id: str) -> EventResult:
        if option_id == "lift":
            if self._lift_potion_slot is None:
                held = run_state.player.held_potions()
                lift_potion = self.get_rng(run_state).choice(held) if held else None
                self._lift_potion_slot = lift_potion.slot_index if lift_potion is not None else None
            if self._lift_potion_slot is not None:
                run_state.player.remove_potion(self._lift_potion_slot)
            run_state.player.gain_max_hp(STONE_OF_ALL_TIME_DRINK_MAX_HP_GAIN)
            self._consume_post_choice_rng(run_state)
            return EventResult(finished=True,
                               description="Discarded a potion, gained 10 Max HP.")
        run_state.player.lose_hp(STONE_OF_ALL_TIME_PUSH_HP_LOSS)
        candidates = [card for card in run_state.player.deck if can_enchant_card(card, "Vigorous")]
        if not candidates:
            return EventResult(finished=True, description="Took 6 damage, but had no card to enchant.")
        if _should_defer_event_rewards(run_state):
            return _event_result_with_rewards(
                "Took 6 damage, enchanted a card with Vigorous +8.",
                [
                    EnchantCardsReward(
                        run_state.player.player_id,
                        enchantment="Vigorous",
                        amount=STONE_OF_ALL_TIME_VIGOROUS_AMOUNT,
                        count=1,
                        cards=candidates,
                        after_selected=lambda: self._consume_post_choice_rng(run_state),
                    )
                ],
            )
        return self.request_card_choice(
            prompt="Choose a card to enchant with Vigorous",
            cards=candidates,
            source_pile="deck",
            resolver=lambda selected: (
                selected and selected[0].add_enchantment("Vigorous", STONE_OF_ALL_TIME_VIGOROUS_AMOUNT),
                self._consume_post_choice_rng(run_state),
                EventResult(finished=True, description="Took 6 damage, enchanted a card with Vigorous +8."),
            )[-1],
            description="Choose a card to enchant.",
        )


register_event(StoneOfAllTime())


# ── Symbiote ──────────────────────────────────────────────────────────

class Symbiote(EventModel):
    """Approach: Enchant a card with Corrupted.
    Kill with Fire: Transform 1 card.
    """

    event_id = "Symbiote"

    def is_allowed(self, run_state: RunState) -> bool:
        return run_state.current_act_index > 0

    def generate_initial_options(self, run_state: RunState) -> list[EventOption]:
        approach_enabled = any(can_enchant_card(card, "Corrupted") for card in run_state.player.deck)
        return [
            EventOption("approach", "Approach", "Enchant a card with Corrupted", enabled=approach_enabled),
            EventOption("kill_fire", "Kill with Fire", "Transform 1 card"),
        ]

    def choose(self, run_state: RunState, option_id: str) -> EventResult:
        if option_id == "approach":
            candidates = [card for card in run_state.player.deck if can_enchant_card(card, "Corrupted")]
            if _should_defer_event_rewards(run_state):
                return _event_result_with_rewards(
                    "Enchanted a card with Corrupted.",
                    [
                        EnchantCardsReward(
                            run_state.player.player_id,
                            enchantment="Corrupted",
                            amount=1,
                            count=1,
                            cards=candidates,
                        )
                    ],
                )
            return self.request_card_choice(
                prompt="Choose a card to enchant with Corrupted",
                cards=candidates,
                source_pile="deck",
                resolver=lambda selected: (
                    selected and selected[0].add_enchantment("Corrupted", 1),
                    EventResult(finished=True, description="Enchanted a card with Corrupted."),
                )[-1],
                description="Choose a card to enchant.",
            )
        candidates = run_state.player.transformable_deck_cards()
        if _should_defer_event_rewards(run_state):
            return _event_result_with_rewards(
                "Transformed 1 card.",
                [
                    TransformCardsReward(
                        run_state.player.player_id,
                        count=min(1, len(candidates)),
                        cards=candidates,
                        rng_override=self.get_rng(run_state),
                    )
                ],
            )
        return self.request_card_choice(
            prompt="Choose a card to transform",
            cards=candidates,
            source_pile="deck",
            resolver=lambda selected: (
                _transform_selected_cards(selected, run_state, rng=self.get_rng(run_state)),
                EventResult(finished=True, description="Transformed 1 card."),
            )[-1],
            description="Choose a card to transform.",
        )


register_event(Symbiote())


# ── TheFutureOfPotions ────────────────────────────────────────────────

class TheFutureOfPotions(EventModel):
    """Trade a potion for upgraded card rewards matching rarity."""

    event_id = "TheFutureOfPotions"
    MAX_VISIBLE_POTION_OPTIONS = 3
    CARD_REWARD_OPTION_COUNT = 3
    TRADE_OPTION_PREFIX = "trade_"
    FULL_CARD_TYPE_POOL = (CardType.ATTACK, CardType.SKILL, CardType.POWER)
    COMMON_TOKEN_CARD_TYPE_POOL = (CardType.ATTACK, CardType.SKILL)

    def __init__(self) -> None:
        self._trade_choices: list[tuple[int, CardRarity, CardType]] = []

    def is_allowed(self, run_state: RunState) -> bool:
        return all(len(player.held_potions()) >= 2 for player in run_state.players)

    def before_event_started(self, run_state: RunState) -> None:
        run_state.player.can_remove_potions = False

    def on_event_finished(self, run_state: RunState) -> None:
        run_state.player.can_remove_potions = True

    def generate_initial_options(self, run_state: RunState) -> list[EventOption]:
        self._trade_choices = []
        potions = run_state.player.held_potions()
        potion_card_types = {
            p.slot_index: self._roll_card_type_for_potion(run_state, p)
            for p in potions
        }
        options = []
        for i, p in enumerate(potions[: self.MAX_VISIBLE_POTION_OPTIONS]):
            chosen_type = potion_card_types[p.slot_index]
            self._trade_choices.append((p.slot_index, self._target_card_rarity(p.rarity), chosen_type))
            options.append(
                EventOption(f"{self.TRADE_OPTION_PREFIX}{i}", f"Trade {p.potion_id}",
                             "Discard potion for upgraded card reward")
            )
        return options

    def choose(self, run_state: RunState, option_id: str) -> EventResult:
        if not option_id.startswith(self.TRADE_OPTION_PREFIX):
            return EventResult(finished=True, description="Nothing happened.")
        try:
            choice_idx = int(option_id.removeprefix(self.TRADE_OPTION_PREFIX))
        except ValueError:
            return EventResult(finished=True, description="Nothing happened.")
        if choice_idx < 0 or choice_idx >= len(self._trade_choices):
            return EventResult(finished=True, description="Nothing happened.")

        slot_index, target_rarity, target_type = self._trade_choices[choice_idx]
        run_state.player.remove_potion(slot_index)
        rewards = CardReward(
            run_state.player.player_id,
            option_count=self.CARD_REWARD_OPTION_COUNT,
            character_ids=(run_state.player.character_id,),
            generation_context=None,
            roll_upgrade=False,
            card_creation_source=CARD_CREATION_SOURCE_OTHER,
            use_default_character_pool=False,
            card_type=target_type,
            allow_rarity_modifications=False,
            card_pool_rarity_filter=target_rarity,
            use_uniform_noncombat_odds=True,
            upgrade_after_generation=True,
        )
        return EventResult(finished=True,
                           description="Traded a potion for upgraded card rewards.",
                           rewards={"reward_objects": [rewards]})

    @staticmethod
    def _target_card_rarity(potion_rarity: PotionRarity) -> CardRarity:
        if potion_rarity in {PotionRarity.RARE, PotionRarity.EVENT}:
            return CardRarity.RARE
        if potion_rarity == PotionRarity.UNCOMMON:
            return CardRarity.UNCOMMON
        return CardRarity.COMMON

    def _roll_card_type_for_potion(self, run_state: RunState, potion) -> CardType:
        card_types = self.FULL_CARD_TYPE_POOL
        if potion.rarity in {PotionRarity.COMMON, PotionRarity.TOKEN}:
            card_types = self.COMMON_TOKEN_CARD_TYPE_POOL
        return self.get_rng(run_state).choice(card_types)

register_event(TheFutureOfPotions())


# ── WaterloggedScriptorium ────────────────────────────────────────────

class WaterloggedScriptorium(EventModel):
    """Bloody Ink: Gain 6 Max HP.
    Tentacle Quill: Pay 65g, enchant 1 card with Steady.
    Prickly Sponge: Pay 155g, enchant 2 cards with Steady.
    """

    event_id = "WaterloggedScriptorium"
    SPAWN_GOLD_REQUIREMENT = 65
    BLOODY_INK_MAX_HP_GAIN = 6
    TENTACLE_QUILL_COST = 65
    TENTACLE_QUILL_CARDS = 1
    PRICKLY_SPONGE_COST = 155
    PRICKLY_SPONGE_CARDS = 2
    STEADY_AMOUNT = 1
    STEADY_ENCHANTMENT = "Steady"

    def is_allowed(self, run_state: RunState) -> bool:
        return all(
            player.gold >= self.SPAWN_GOLD_REQUIREMENT
            for player in run_state.players
        )

    def generate_initial_options(self, run_state: RunState) -> list[EventOption]:
        gold = run_state.player.gold
        return [
            EventOption(
                "bloody_ink",
                "Bloody Ink",
                f"Gain {self.BLOODY_INK_MAX_HP_GAIN} Max HP",
            ),
            EventOption(
                "tentacle_quill",
                f"Tentacle Quill ({self.TENTACLE_QUILL_COST}g)"
                if gold >= self.TENTACLE_QUILL_COST else "Tentacle Quill",
                f"Enchant {self.TENTACLE_QUILL_CARDS} card with Steady",
                enabled=gold >= self.TENTACLE_QUILL_COST,
            ),
            EventOption(
                "prickly_sponge",
                f"Prickly Sponge ({self.PRICKLY_SPONGE_COST}g)"
                if gold >= self.PRICKLY_SPONGE_COST else "Prickly Sponge",
                f"Enchant {self.PRICKLY_SPONGE_CARDS} cards with Steady",
                enabled=gold >= self.PRICKLY_SPONGE_COST,
            ),
        ]

    def choose(self, run_state: RunState, option_id: str) -> EventResult:
        if option_id == "bloody_ink":
            run_state.player.gain_max_hp(self.BLOODY_INK_MAX_HP_GAIN)
            return EventResult(
                finished=True,
                description=f"Gained {self.BLOODY_INK_MAX_HP_GAIN} Max HP.",
            )
        if option_id == "tentacle_quill":
            run_state.player.lose_gold(self.TENTACLE_QUILL_COST)
            candidates = [
                card for card in run_state.player.deck
                if can_enchant_card(card, self.STEADY_ENCHANTMENT)
            ]
            if not candidates:
                return EventResult(
                    finished=True,
                    description=f"Paid {self.TENTACLE_QUILL_COST}g, but had no card to enchant.",
                )
            if _should_defer_event_rewards(run_state):
                return _event_result_with_rewards(
                    (
                        f"Paid {self.TENTACLE_QUILL_COST}g, enchanted "
                        f"{self.TENTACLE_QUILL_CARDS} card with Steady."
                    ),
                    [
                        EnchantCardsReward(
                            run_state.player.player_id,
                            enchantment=self.STEADY_ENCHANTMENT,
                            amount=self.STEADY_AMOUNT,
                            count=self.TENTACLE_QUILL_CARDS,
                            cards=candidates,
                        )
                    ],
                )
            return self.request_card_choice(
                prompt="Choose a card to enchant with Steady",
                cards=candidates,
                source_pile="deck",
                resolver=lambda selected: (
                    selected and selected[0].add_enchantment(
                        self.STEADY_ENCHANTMENT,
                        self.STEADY_AMOUNT,
                    ),
                    EventResult(
                        finished=True,
                        description=(
                            f"Paid {self.TENTACLE_QUILL_COST}g, enchanted "
                            f"{self.TENTACLE_QUILL_CARDS} card with Steady."
                        ),
                    ),
                )[-1],
                description="Choose a card to enchant.",
            )
        run_state.player.lose_gold(self.PRICKLY_SPONGE_COST)
        candidates = [
            card for card in run_state.player.deck
            if can_enchant_card(card, self.STEADY_ENCHANTMENT)
        ]
        if not candidates:
            return EventResult(
                finished=True,
                description=f"Paid {self.PRICKLY_SPONGE_COST}g, but had no cards to enchant.",
            )
        if _should_defer_event_rewards(run_state):
            return _event_result_with_rewards(
                (
                    f"Paid {self.PRICKLY_SPONGE_COST}g, enchanted "
                    f"{self.PRICKLY_SPONGE_CARDS} cards with Steady."
                ),
                [
                    EnchantCardsReward(
                        run_state.player.player_id,
                        enchantment=self.STEADY_ENCHANTMENT,
                        amount=self.STEADY_AMOUNT,
                        count=min(self.PRICKLY_SPONGE_CARDS, len(candidates)),
                        cards=candidates,
                    )
                ],
            )
        return self.request_card_choice(
            prompt="Choose 2 cards to enchant with Steady",
            cards=candidates,
            source_pile="deck",
            resolver=lambda selected: (
                [
                    card.add_enchantment(self.STEADY_ENCHANTMENT, self.STEADY_AMOUNT)
                    for card in selected
                ],
                EventResult(
                    finished=True,
                    description=(
                        f"Paid {self.PRICKLY_SPONGE_COST}g, enchanted "
                        f"{self.PRICKLY_SPONGE_CARDS} cards with Steady."
                    ),
                ),
            )[-1],
            min_count=min(self.PRICKLY_SPONGE_CARDS, len(candidates)),
            max_count=min(self.PRICKLY_SPONGE_CARDS, len(candidates)),
            description="Choose 2 cards to enchant.",
        )


register_event(WaterloggedScriptorium())


# ── WelcomeToWongos ───────────────────────────────────────────────────

class WelcomeToWongos(EventModel):
    """Bargain Bin: 100g for common relic. Featured Item: 200g for rare relic.
    Mystery Box: 300g for mystery ticket. Leave: Downgrade a random card.
    """

    event_id = "WelcomeToWongos"
    BARGAIN_BIN_COST = 100
    FEATURED_ITEM_COST = 200
    MYSTERY_BOX_COST = 300
    BARGAIN_BIN_WONGO_POINTS = 32
    FEATURED_ITEM_WONGO_POINTS = 16
    MYSTERY_BOX_WONGO_POINTS = 8
    WONGO_POINTS_FOR_BADGE = 2000
    BADGE_RELIC_ID = RelicId.WONGO_CUSTOMER_APPRECIATION_BADGE.name
    MYSTERY_TICKET_RELIC_ID = RelicId.WONGOS_MYSTERY_TICKET.name

    def __init__(self) -> None:
        self._featured_relic_id: str | None = None

    def is_allowed(self, run_state: RunState) -> bool:
        return run_state.current_act_index == 1 and all(
            player.gold >= self.BARGAIN_BIN_COST
            for player in run_state.players
        )

    def generate_initial_options(self, run_state: RunState) -> list[EventOption]:
        gold = run_state.player.gold
        featured = RelicReward(run_state.player.player_id, rarity=RelicRarity.RARE)
        featured.populate(run_state, None)
        self._featured_relic_id = featured.relic_id
        options: list[EventOption] = [
            EventOption(
                "bargain_bin",
                f"Bargain Bin ({self.BARGAIN_BIN_COST}g)" if gold >= self.BARGAIN_BIN_COST else "Bargain Bin",
                "Common relic",
                enabled=gold >= self.BARGAIN_BIN_COST,
            )
        ]
        if gold >= self.FEATURED_ITEM_COST:
            options.append(EventOption("featured", f"Featured Item ({self.FEATURED_ITEM_COST}g)",
                                        "Rare relic"))
        else:
            options.append(EventOption("featured", "Featured Item", "Rare relic", enabled=False))
        if gold >= self.MYSTERY_BOX_COST:
            options.append(EventOption("mystery", f"Mystery Box ({self.MYSTERY_BOX_COST}g)",
                                        "Mystery ticket relic"))
        else:
            options.append(EventOption("mystery", "Mystery Box", "Mystery ticket relic", enabled=False))
        options.append(EventOption("leave", "Leave",
                                    "Downgrade a random upgraded card"))
        return options

    def _finish_purchase(
        self,
        run_state: RunState,
        *,
        description: str,
        points: int,
        reward_objects: list[object] | None = None,
    ) -> EventResult:
        previous_points = run_state.player.wongo_points
        run_state.player.wongo_points += points
        run_state.extra_fields["wongo_points_earned"] = points
        if (
            run_state.player.wongo_points // self.WONGO_POINTS_FOR_BADGE
            > previous_points // self.WONGO_POINTS_FOR_BADGE
        ):
            badge_reward = RelicReward(run_state.player.player_id, relic_id=self.BADGE_RELIC_ID)
            if not _should_defer_event_rewards(run_state):
                run_state.player.obtain_relic(self.BADGE_RELIC_ID)
            elif reward_objects is None:
                reward_objects = [badge_reward]
            else:
                reward_objects = [*reward_objects, badge_reward]
        if reward_objects and _should_defer_event_rewards(run_state):
            return _event_result_with_rewards(description, reward_objects)
        return EventResult(finished=True, description=description)

    def choose(self, run_state: RunState, option_id: str) -> EventResult:
        if option_id == "bargain_bin":
            run_state.player.lose_gold(self.BARGAIN_BIN_COST)
            reward = RelicReward(run_state.player.player_id, rarity=RelicRarity.COMMON)
            reward.populate(run_state, None)
            rewards: list[object] | None = None
            if reward.relic_id is not None:
                if _should_defer_event_rewards(run_state):
                    rewards = [RelicReward(run_state.player.player_id, relic_id=reward.relic_id)]
                else:
                    run_state.player.obtain_relic(reward.relic_id)
            return self._finish_purchase(
                run_state,
                description=f"Bought common relic for {self.BARGAIN_BIN_COST}g.",
                points=self.BARGAIN_BIN_WONGO_POINTS,
                reward_objects=rewards,
            )
        if option_id == "featured":
            run_state.player.lose_gold(self.FEATURED_ITEM_COST)
            reward_id = self._featured_relic_id
            rewards = None
            if reward_id is not None:
                if _should_defer_event_rewards(run_state):
                    rewards = [RelicReward(run_state.player.player_id, relic_id=reward_id)]
                else:
                    run_state.player.obtain_relic(reward_id)
            return self._finish_purchase(
                run_state,
                description=f"Bought rare relic for {self.FEATURED_ITEM_COST}g.",
                points=self.FEATURED_ITEM_WONGO_POINTS,
                reward_objects=rewards,
            )
        if option_id == "mystery":
            run_state.player.lose_gold(self.MYSTERY_BOX_COST)
            rewards = None
            if _should_defer_event_rewards(run_state):
                rewards = [RelicReward(run_state.player.player_id, relic_id=self.MYSTERY_TICKET_RELIC_ID)]
            else:
                run_state.player.obtain_relic(self.MYSTERY_TICKET_RELIC_ID)
            return self._finish_purchase(
                run_state,
                description=f"Bought mystery box for {self.MYSTERY_BOX_COST}g.",
                points=self.MYSTERY_BOX_WONGO_POINTS,
                reward_objects=rewards,
            )
        upgraded_cards = [card for card in run_state.player.deck if card.upgraded]
        if upgraded_cards:
            _downgrade_selected_cards([self.get_rng(run_state).choice(upgraded_cards)], run_state)
        return EventResult(finished=True,
                           description="Left Wongo's, downgraded a card.")


register_event(WelcomeToWongos())


# ── WhisperingHollow ──────────────────────────────────────────────────

class WhisperingHollow(EventModel):
    """Gold: Pay 50g, gain 2 potions. Hug: Take 9 damage, transform 1 card."""

    event_id = "WhisperingHollow"
    GOLD_COST = 50
    GOLD_POTION_REWARD_COUNT = 2
    HUG_DAMAGE = 9
    OPTION_GOLD = "gold"
    OPTION_HUG = "hug"

    def is_allowed(self, run_state: RunState) -> bool:
        return all(player.gold >= self.GOLD_COST for player in run_state.players)

    def generate_initial_options(self, run_state: RunState) -> list[EventOption]:
        return [
            EventOption(
                self.OPTION_GOLD,
                f"Pay Gold ({self.GOLD_COST}g)",
                f"Gain {self.GOLD_POTION_REWARD_COUNT} potions",
            ),
            EventOption(
                self.OPTION_HUG,
                "Hug",
                f"Take {self.HUG_DAMAGE} damage, transform 1 card",
            ),
        ]

    def choose(self, run_state: RunState, option_id: str) -> EventResult:
        if option_id == self.OPTION_GOLD:
            run_state.player.lose_gold(self.GOLD_COST)
            rewards = [
                PotionReward(run_state.player.player_id)
                for _ in range(self.GOLD_POTION_REWARD_COUNT)
            ]
            return EventResult(
                finished=True,
                description=f"Paid {self.GOLD_COST}g, gained {self.GOLD_POTION_REWARD_COUNT} potions.",
                rewards={"reward_objects": rewards},
            )
        candidates = run_state.player.transformable_deck_cards()
        if _should_defer_event_rewards(run_state):
            return _event_result_with_rewards(
                f"Took {self.HUG_DAMAGE} damage, transformed 1 card.",
                [
                    TransformCardsReward(
                        run_state.player.player_id,
                        count=min(1, len(candidates)),
                        cards=candidates,
                        rng_override=self.get_rng(run_state),
                        after_selected=lambda: run_state.player.lose_hp(self.HUG_DAMAGE),
                    )
                ],
            )
        return self.request_card_choice(
            prompt="Choose a card to transform",
            cards=candidates,
            source_pile="deck",
            resolver=lambda selected: (
                _transform_selected_cards(selected, run_state, rng=self.get_rng(run_state)),
                run_state.player.lose_hp(self.HUG_DAMAGE),
                EventResult(finished=True, description=f"Took {self.HUG_DAMAGE} damage, transformed 1 card."),
            )[-1],
            description="Choose a card to transform.",
        )


register_event(WhisperingHollow())
