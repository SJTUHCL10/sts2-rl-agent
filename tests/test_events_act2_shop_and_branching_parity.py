"""Parity tests for Act 2 shop and branching events."""

import sts2_env.events.act2  # noqa: F401
import sts2_env.events.act3  # noqa: F401

from sts2_env.cards.factory import create_card
from sts2_env.cards.ironclad import create_ironclad_starter_deck
from sts2_env.cards.status import make_spore_mind
from sts2_env.core.enums import CardId, ValueProp
from sts2_env.events.act2 import (
    CrystalSphere,
    EndlessConveyor,
    FakeMerchant,
    FieldOfManSizedHoles,
    WelcomeToWongos,
)
from sts2_env.events.act3 import TheArchitect
from sts2_env.potions.base import create_potion
from sts2_env.run.modifiers import ModifierModel
from sts2_env.run.rewards import CardRewardGenerationOptions
from sts2_env.run.run_manager import RunManager
from sts2_env.run.run_state import PlayerState, RunState

FIRST_DECK_CHOICE_INDEX = 0
ENDLESS_CONVEYOR_OBSERVE_RUN_MANAGER_SEED = 90521
ENDLESS_CONVEYOR_SPICY_SNAPPY_RUN_MANAGER_SEED = 90522
ENDLESS_CONVEYOR_JELLY_LIVER_RUN_MANAGER_SEED = 90523
ENDLESS_CONVEYOR_STARTING_GOLD = 200
ENDLESS_CONVEYOR_POST_INITIAL_ROLL_GRABS = 1


def _make_run_state(seed: int) -> RunState:
    run_state = RunState(seed=seed, character_id="Ironclad")
    run_state.initialize_run()
    run_state.player.deck = create_ironclad_starter_deck()
    return run_state


class _SwapFirstTwoRng:
    def __init__(self) -> None:
        self.shuffle_calls = 0

    def shuffle(self, seq) -> None:
        self.shuffle_calls += 1
        seq[0], seq[1] = seq[1], seq[0]

    def choice(self, seq):
        return seq[0]

    def next_float(self) -> float:
        return 0.0


class _LastChoiceRng:
    def __init__(self) -> None:
        self.choice_calls = 0

    def choice(self, seq):
        self.choice_calls += 1
        return seq[-1]


class _FirstChoiceCountingRng:
    def __init__(self) -> None:
        self.choice_calls = 0

    def choice(self, seq):
        self.choice_calls += 1
        return seq[0]


class _LastFloatRng:
    def next_float(self) -> float:
        return 0.999


class _FixedFloatRng:
    def __init__(self, value: float) -> None:
        self.value = value

    def next_float(self) -> float:
        return self.value


class _EndlessConveyorFriedEelRewardModifier(ModifierModel):
    def __init__(self) -> None:
        super().__init__("endless_conveyor_fried_eel_reward")
        self.creation_options_seen = False
        self.options_seen = False

    def modify_card_reward_creation_options(self, player, options, reward, room, run_state):
        self.creation_options_seen = True
        return CardRewardGenerationOptions(
            context=options.context,
            num_cards=options.num_cards,
            character_ids=options.character_ids,
            forced_rarities=options.forced_rarities,
            include_colorless=options.include_colorless,
            use_default_character_pool=options.use_default_character_pool,
            generation_context=options.generation_context,
            roll_upgrade=options.roll_upgrade,
            card_type=options.card_type,
            card_creation_source=options.card_creation_source,
            allow_card_pool_modifications=options.allow_card_pool_modifications,
            allow_rarity_modifications=options.allow_rarity_modifications,
            allow_hook_upgrades=options.allow_hook_upgrades,
            has_custom_card_pool=True,
            custom_card_ids=(CardId.DARK_SHACKLES,),
            card_pool_rarity_filter=options.card_pool_rarity_filter,
        )

    def modify_card_reward_options(self, player, cards, reward, room, run_state):
        self.options_seen = True
        return cards


def test_the_architect_is_pool_disabled_and_has_single_proceed_option():
    run_state = _make_run_state(901)
    event = TheArchitect()

    assert event.is_allowed(run_state) is False
    options = event.generate_initial_options(run_state)
    assert [option.option_id for option in options] == ["proceed"]

    result = event.choose(run_state, "proceed")
    assert result.finished
    assert "architect" in result.description.lower()
    assert result.event_combat_setup == "the_architect"
    assert result.post_combat_phase == "RUN_OVER"


def test_run_manager_enters_the_architect_combat_and_ends_run_after_victory():
    mgr = RunManager(seed=901, character_id="Ironclad")
    mgr._phase = RunManager.PHASE_EVENT
    event = TheArchitect()
    mgr._event_model = event
    mgr._event_options = event.generate_initial_options(mgr.run_state)

    result = mgr._do_event_choice({"option_id": "proceed"})
    assert result["phase"] == RunManager.PHASE_COMBAT

    combat = mgr._combat  # noqa: SLF001
    enemy = combat.enemies[0]
    combat.deal_damage(combat.player, enemy, 99999, ValueProp.UNPOWERED)
    resolved = mgr._resolve_combat_end()  # noqa: SLF001
    assert resolved["phase"] == RunManager.PHASE_RUN_OVER
    assert mgr.run_state.is_over is True
    assert mgr.run_state.player_won is True


def test_crystal_sphere_thresholds_and_choices_apply_cost_or_debt():
    blocked_state = _make_run_state(902)
    event = CrystalSphere()
    assert event.is_allowed(blocked_state) is False

    blocked_state.current_act_index = 1
    blocked_state.player.gold = 99
    assert event.is_allowed(blocked_state) is False

    run_state = _make_run_state(903)
    run_state.current_act_index = 1
    run_state.player.gold = 180
    options = event.generate_initial_options(run_state)
    assert [option.option_id for option in options] == ["pay", "debt"]
    assert (
        event.UNCOVER_FUTURE_BASE_COST + event.UNCOVER_FUTURE_RANDOM_MIN
        <= event._cost  # noqa: SLF001
        < event.UNCOVER_FUTURE_BASE_COST + event.UNCOVER_FUTURE_RANDOM_MAX
    )

    gold_before = run_state.player.gold
    pay = event.choose(run_state, "pay")
    assert not pay.finished
    assert len(pay.next_options) > 4
    assert all(option.option_id.startswith("divine:") for option in pay.next_options)
    assert event.minigame is not None
    assert event.minigame.divination_count == 3
    assert run_state.player.gold == gold_before - event._cost  # noqa: SLF001

    debt_state = _make_run_state(904)
    debt_state.current_act_index = 1
    debt_state.player.gold = 200
    debt_event = CrystalSphere()
    debt_event.generate_initial_options(debt_state)
    deck_before = len(debt_state.player.deck)

    debt = debt_event.choose(debt_state, "debt")
    assert not debt.finished
    assert debt_event.minigame is not None
    assert debt_event.minigame.divination_count == 6
    assert len(debt_state.player.deck) == deck_before + 1
    assert debt_state.player.deck[-1].card_id == CardId.DEBT


def test_crystal_sphere_requires_all_players_to_have_entry_gold():
    run_state = _make_run_state(9041)
    run_state.current_act_index = 1
    run_state.player.gold = 100
    ally = run_state.add_player(PlayerState(player_id=2, character_id="Silent", gold=99))
    event = CrystalSphere()

    assert event.is_allowed(run_state) is False

    ally.gold = 100
    assert event.is_allowed(run_state) is True


def test_endless_conveyor_observe_and_targeted_dishes_apply_expected_effects():
    observe_state = _make_run_state(905)
    observe_state.player.gold = 200
    observe = EndlessConveyor()
    observe.generate_initial_options(observe_state)
    observed = observe.choose(observe_state, "observe")
    assert observed.finished
    assert any(card.upgraded for card in observe_state.player.deck)

    run_state = _make_run_state(906)
    run_state.player.gold = 200
    event = EndlessConveyor()

    event._current_dish = "golden_fysh"  # noqa: SLF001
    gold_before = run_state.player.gold
    golden = event.choose(run_state, "grab")
    assert not golden.finished
    assert run_state.player.gold == gold_before + EndlessConveyor.GOLDEN_FYSH_GOLD

    event._current_dish = "caviar"  # noqa: SLF001
    run_state.player.gold = EndlessConveyor.GRAB_GOLD
    max_hp_before = run_state.player.max_hp
    caviar = event.choose(run_state, "grab")
    assert not caviar.finished
    assert run_state.player.max_hp == max_hp_before + EndlessConveyor.CAVIAR_MAX_HP
    assert run_state.player.gold == 0
    assert [option.option_id for option in caviar.next_options] == ["grab", "leave"]
    assert [option.enabled for option in caviar.next_options] == [False, True]

    event._current_dish = "seapunk_salad"  # noqa: SLF001
    run_state.player.gold = 100
    deck_before = len(run_state.player.deck)
    seapunk = event.choose(run_state, "grab")
    assert not seapunk.finished
    assert len(run_state.player.deck) == deck_before + 1
    assert run_state.player.deck[-1].card_id == CardId.FEEDING_FRENZY_CARD

    event._current_dish = "fried_eel"  # noqa: SLF001
    deck_before = len(run_state.player.deck)
    eel = event.choose(run_state, "grab")
    assert not eel.finished
    assert len(run_state.player.deck) == deck_before + 1


def test_endless_conveyor_fried_eel_uses_card_reward_hooks():
    run_state = _make_run_state(90601)
    run_state.player.gold = ENDLESS_CONVEYOR_STARTING_GOLD
    modifier = _EndlessConveyorFriedEelRewardModifier()
    run_state.modifiers = [modifier]
    event = EndlessConveyor()
    event._current_dish = EndlessConveyor.DISH_FRIED_EEL  # noqa: SLF001

    result = event.choose(run_state, EndlessConveyor.OPTION_GRAB)

    assert result.finished is False
    assert modifier.creation_options_seen is True
    assert modifier.options_seen is True
    assert run_state.player.deck[-1].card_id == CardId.DARK_SHACKLES


def test_endless_conveyor_requires_all_players_to_have_initial_gold():
    run_state = _make_run_state(9061)
    run_state.player.gold = EndlessConveyor.REQUIRED_GOLD
    ally = run_state.add_player(PlayerState(player_id=2, character_id="Silent", gold=EndlessConveyor.REQUIRED_GOLD - 1))
    event = EndlessConveyor()

    assert event.is_allowed(run_state) is False

    ally.gold = EndlessConveyor.REQUIRED_GOLD
    assert event.is_allowed(run_state) is True


def test_endless_conveyor_excludes_the_dish_it_just_rolled_next_time():
    run_state = _make_run_state(9062)
    run_state.player.gold = 200
    event = EndlessConveyor()
    event.rng = _SwapFirstTwoRng()

    event._roll_dish(run_state)
    first_dish = event._current_dish
    event._roll_dish(run_state)

    assert first_dish == "caviar"
    assert event._current_dish == "spicy_snappy"


def test_endless_conveyor_weight_boundary_uses_reference_strict_less_than():
    run_state = _make_run_state(90621)
    run_state.player.gold = 200
    run_state.player.current_hp = run_state.player.max_hp
    run_state.player.potions = [None] * run_state.player.max_potion_slots
    event = EndlessConveyor()
    event.rng = _FixedFloatRng(EndlessConveyor.BASE_DISH_WEIGHTS[0][1] / sum(weight for _, weight in EndlessConveyor.BASE_DISH_WEIGHTS))

    event._roll_dish(run_state)

    assert event._current_dish == "spicy_snappy"


def test_endless_conveyor_golden_fysh_can_appear_after_first_grab():
    run_state = _make_run_state(9063)
    run_state.player.gold = 200
    event = EndlessConveyor()
    event.rng = _LastFloatRng()
    event._grabs = 1

    event._roll_dish(run_state)

    assert event._current_dish == "golden_fysh"


def test_endless_conveyor_forced_seapunk_counts_as_last_dish():
    run_state = _make_run_state(9064)
    run_state.player.gold = 200
    event = EndlessConveyor()
    event.rng = _SwapFirstTwoRng()
    event._grabs = EndlessConveyor.FORCED_SEAPUNK_INTERVAL - 1

    event._roll_dish(run_state)
    assert event._current_dish == "seapunk_salad"
    event._grabs = EndlessConveyor.FORCED_SEAPUNK_INTERVAL
    event._roll_dish(run_state)

    assert event._current_dish != "seapunk_salad"


def test_endless_conveyor_roll_dish_increments_grab_counter_like_reference():
    run_state = _make_run_state(90641)
    run_state.player.gold = 200
    event = EndlessConveyor()
    event.rng = _SwapFirstTwoRng()

    event._roll_dish(run_state)

    assert event._grabs == 1


def test_endless_conveyor_fourth_grab_roll_forces_fifth_dish_to_seapunk_salad():
    run_state = _make_run_state(90642)
    run_state.player.gold = 500
    event = EndlessConveyor()
    event.rng = _SwapFirstTwoRng()
    event.generate_initial_options(run_state)

    for _ in range(4):
        result = event.choose(run_state, "grab")
        assert result.finished is False

    assert event._grabs == EndlessConveyor.FORCED_SEAPUNK_INTERVAL
    assert event._current_dish == "seapunk_salad"


def test_endless_conveyor_observe_uses_event_rng_for_upgrade_selection():
    run_state = _make_run_state(9051)
    run_state.player.gold = 200
    first = create_card(CardId.DEFEND_IRONCLAD)
    second = create_card(CardId.STRIKE_IRONCLAD)
    run_state.player.deck = [first, second]
    event = EndlessConveyor()
    event.rng = _SwapFirstTwoRng()
    rewards_counter = run_state.rng.rewards.counter

    result = event.choose(run_state, "observe")

    assert result.finished
    assert event.rng.shuffle_calls == 1
    assert run_state.rng.rewards.counter == rewards_counter
    assert first.upgraded is False
    assert second.upgraded is True


def test_endless_conveyor_spicy_snappy_uses_event_rng_for_upgrade_selection():
    run_state = _make_run_state(9052)
    run_state.player.gold = 200
    first = create_card(CardId.DEFEND_IRONCLAD)
    second = create_card(CardId.STRIKE_IRONCLAD)
    run_state.player.deck = [first, second]
    event = EndlessConveyor()
    event.rng = _SwapFirstTwoRng()
    event._current_dish = "spicy_snappy"  # noqa: SLF001
    rewards_counter = run_state.rng.rewards.counter

    result = event.choose(run_state, "grab")

    assert result.finished is False
    assert event.rng.shuffle_calls == 1
    assert run_state.rng.rewards.counter == rewards_counter
    assert first.upgraded is False
    assert second.upgraded is True


def test_endless_conveyor_observe_does_not_open_run_manager_card_choice():
    mgr = RunManager(seed=ENDLESS_CONVEYOR_OBSERVE_RUN_MANAGER_SEED, character_id="Ironclad")
    mgr._phase = RunManager.PHASE_EVENT
    mgr.run_state.player.gold = ENDLESS_CONVEYOR_STARTING_GOLD
    mgr.run_state.player.deck = [
        create_card(CardId.DEFEND_IRONCLAD),
        create_card(CardId.STRIKE_IRONCLAD),
    ]
    conveyor = EndlessConveyor()
    mgr._event_model = conveyor
    mgr._event_started = True
    mgr._event_options = conveyor.generate_initial_options(mgr.run_state)

    result = mgr._do_event_choice({"option_id": EndlessConveyor.OPTION_OBSERVE})

    assert result["phase"] == RunManager.PHASE_MAP_CHOICE
    assert mgr.run_state.pending_choice is None
    assert any(card.upgraded for card in mgr.run_state.player.deck)


def test_endless_conveyor_spicy_snappy_does_not_open_run_manager_card_choice():
    mgr = RunManager(seed=ENDLESS_CONVEYOR_SPICY_SNAPPY_RUN_MANAGER_SEED, character_id="Ironclad")
    mgr._phase = RunManager.PHASE_EVENT
    mgr.run_state.player.gold = ENDLESS_CONVEYOR_STARTING_GOLD
    mgr.run_state.player.deck = [
        create_card(CardId.DEFEND_IRONCLAD),
        create_card(CardId.STRIKE_IRONCLAD),
    ]
    conveyor = EndlessConveyor()
    conveyor._current_dish = EndlessConveyor.DISH_SPICY_SNAPPY  # noqa: SLF001
    mgr._event_model = conveyor
    mgr._event_started = True

    result = mgr._do_event_choice({"option_id": EndlessConveyor.OPTION_GRAB})

    assert result["phase"] == RunManager.PHASE_EVENT
    assert mgr.run_state.pending_choice is None
    assert any(card.upgraded for card in mgr.run_state.player.deck)


def test_endless_conveyor_jelly_liver_rolls_next_dish_after_deferred_transform_choice():
    mgr = RunManager(seed=ENDLESS_CONVEYOR_JELLY_LIVER_RUN_MANAGER_SEED, character_id="Ironclad")
    mgr._phase = RunManager.PHASE_EVENT
    mgr.run_state.player.gold = ENDLESS_CONVEYOR_STARTING_GOLD
    mgr.run_state.player.deck = [
        create_card(CardId.DEFEND_IRONCLAD),
        create_card(CardId.STRIKE_IRONCLAD),
    ]
    conveyor = EndlessConveyor()
    conveyor.rng = _SwapFirstTwoRng()
    conveyor._grabs = ENDLESS_CONVEYOR_POST_INITIAL_ROLL_GRABS  # noqa: SLF001
    conveyor._current_dish = EndlessConveyor.DISH_JELLY_LIVER  # noqa: SLF001
    mgr._event_model = conveyor
    mgr._event_started = True

    result = mgr._do_event_choice({"option_id": EndlessConveyor.OPTION_GRAB})

    assert result["phase"] == RunManager.PHASE_CARD_REWARD
    assert mgr.run_state.pending_choice is not None
    assert conveyor._current_dish == EndlessConveyor.DISH_JELLY_LIVER  # noqa: SLF001

    final = mgr.take_action({"action": "choose", "index": FIRST_DECK_CHOICE_INDEX})

    assert final["phase"] == RunManager.PHASE_EVENT
    assert mgr.run_state.pending_choice is None
    assert conveyor._current_dish == EndlessConveyor.DISH_CAVIAR  # noqa: SLF001


def test_endless_conveyor_suspicious_condiment_rolls_specific_potion_reward():
    run_state = _make_run_state(9053)
    run_state.player.gold = 200
    run_state.rng.rewards = _FirstChoiceCountingRng()
    event = EndlessConveyor()
    event._current_dish = "suspicious_condiment"  # noqa: SLF001

    result = event.choose(run_state, "grab")

    assert result.finished is False
    assert len(run_state.pending_rewards) == 1
    reward = run_state.pending_rewards[0]
    assert reward.potion_id == "BloodPotion"
    assert run_state.rng.rewards.choice_calls == 1

    reward.populate(run_state, None)

    assert reward.potion_id == "BloodPotion"
    assert run_state.rng.rewards.choice_calls == 1


def test_endless_conveyor_jelly_liver_uses_event_rng_for_transform_selection():
    run_state = _make_run_state(9061)
    run_state.player.gold = 200
    first = create_card(CardId.STRIKE_IRONCLAD)
    second = create_card(CardId.DEFEND_IRONCLAD)
    run_state.player.deck = [first, second]
    event = EndlessConveyor()
    event.rng = _SwapFirstTwoRng()
    event._current_dish = "jelly_liver"  # noqa: SLF001
    niche_counter = run_state.rng.niche.counter

    result = event.choose(run_state, "grab")

    assert result.finished is False
    assert run_state.rng.niche.counter == niche_counter
    assert run_state.player.deck[0].card_id == CardId.STRIKE_IRONCLAD
    assert run_state.player.deck[1].card_id != CardId.DEFEND_IRONCLAD


def test_fake_merchant_gating_options_and_choices_reflect_foul_potion_and_gold():
    blocked_state = _make_run_state(907)
    blocked_state.current_act_index = 1
    blocked_state.player.gold = 99
    event = FakeMerchant()
    assert event.is_allowed(blocked_state) is False

    multiplayer_state = _make_run_state(9071)
    multiplayer_state.current_act_index = 1
    multiplayer_state.player.gold = 200
    multiplayer_state.players.append(multiplayer_state.player)
    assert event.is_allowed(multiplayer_state) is False

    option_state = _make_run_state(908)
    option_state.current_act_index = 1
    option_state.player.gold = 120
    up_front_counter = option_state.rng.up_front.counter
    options = event.generate_initial_options(option_state)
    assert [option.option_id for option in options] == ["buy", "leave"]
    assert option_state.rng.up_front.counter == up_front_counter

    foul_state = _make_run_state(909)
    foul_state.current_act_index = 1
    foul_state.player.gold = 0
    foul_state.player.add_potion(create_potion("FoulPotion"))
    assert event.is_allowed(foul_state)
    foul_options = event.generate_initial_options(foul_state)
    assert [option.option_id for option in foul_options] == ["buy", "throw_foul", "leave"]

    gold_before = option_state.player.gold
    buy = event.choose(option_state, "buy")
    assert not buy.finished
    assert buy.next_options[-1].option_id == "leave"
    purchase = event.choose(option_state, buy.next_options[0].option_id)
    assert purchase.finished is False
    assert option_state.player.gold == gold_before - 50
    assert len(event._inventories[id(option_state)]) == 5  # noqa: SLF001
    assert any(option.option_id == "leave" for option in purchase.next_options)

    throw_foul = event.choose(foul_state, "throw_foul")
    assert throw_foul.finished
    assert throw_foul.event_combat_setup == "fake_merchant"
    assert len(throw_foul.rewards["reward_objects"]) == 7
    assert all(p.potion_id != "FoulPotion" for p in foul_state.player.held_potions())


def test_fake_merchant_throw_foul_only_rewards_current_stock():
    run_state = _make_run_state(919)
    run_state.current_act_index = 1
    run_state.player.gold = 200
    run_state.player.add_potion(create_potion("FoulPotion"))
    event = FakeMerchant()
    event.generate_initial_options(run_state)

    buy = event.choose(run_state, "buy")
    assert not buy.finished
    event.choose(run_state, buy.next_options[0].option_id)

    throw_foul = event.choose(run_state, "throw_foul")
    rewards = throw_foul.rewards["reward_objects"]
    assert len(rewards) == 6


def test_fake_merchant_failed_fake_relic_purchase_keeps_foul_potion_fight_available():
    run_state = _make_run_state(9191)
    run_state.current_act_index = 1
    run_state.player.gold = FakeMerchant.FAKE_RELIC_COST - 1
    run_state.player.add_potion(create_potion(FakeMerchant.FOUL_POTION_ID))
    event = FakeMerchant()
    event.generate_initial_options(run_state)
    inventory_before = list(event._inventories[id(run_state)])  # noqa: SLF001

    buy = event.choose(run_state, FakeMerchant.OPTION_BUY)
    assert buy.finished is False

    failed = event.choose(run_state, buy.next_options[0].option_id)

    assert failed.finished is False
    assert run_state.player.gold == FakeMerchant.FAKE_RELIC_COST - 1
    assert event._inventories[id(run_state)] == inventory_before  # noqa: SLF001
    assert any(option.option_id == FakeMerchant.OPTION_THROW_FOUL for option in failed.next_options)

    throw_foul = event.choose(run_state, FakeMerchant.OPTION_THROW_FOUL)

    assert throw_foul.finished
    assert throw_foul.event_combat_setup == "fake_merchant"
    assert len(throw_foul.rewards["reward_objects"]) == FakeMerchant.INVENTORY_SIZE + 1
    assert all(potion.potion_id != FakeMerchant.FOUL_POTION_ID for potion in run_state.player.held_potions())


def test_run_manager_enters_combat_for_fake_merchant_throw_foul():
    mgr = RunManager(seed=909, character_id="Ironclad")
    mgr.run_state.current_act_index = 1
    mgr.run_state.player.deck = create_ironclad_starter_deck()
    mgr.run_state.player.gold = 0
    mgr.run_state.player.add_potion(create_potion("FoulPotion"))
    mgr._phase = RunManager.PHASE_EVENT
    event = FakeMerchant()
    mgr._event_model = event
    mgr._event_options = event.generate_initial_options(mgr.run_state)

    result = mgr._do_event_choice({"option_id": "throw_foul"})
    assert result["phase"] == RunManager.PHASE_COMBAT


def test_welcome_to_wongos_thresholds_purchase_effects_and_leave_downgrade():
    threshold_state = _make_run_state(910)
    threshold_state.current_act_index = 1
    event = WelcomeToWongos()

    threshold_state.player.gold = 100
    options = event.generate_initial_options(threshold_state)
    assert [option.option_id for option in options] == [
        "bargain_bin",
        "featured",
        "mystery",
        "leave",
    ]
    assert [option.enabled for option in options] == [True, False, False, True]

    threshold_state.player.gold = 200
    options = event.generate_initial_options(threshold_state)
    assert [option.option_id for option in options] == [
        "bargain_bin",
        "featured",
        "mystery",
        "leave",
    ]
    assert [option.enabled for option in options] == [True, True, False, True]

    threshold_state.player.gold = 300
    options = event.generate_initial_options(threshold_state)
    assert [option.option_id for option in options] == [
        "bargain_bin",
        "featured",
        "mystery",
        "leave",
    ]
    assert [option.enabled for option in options] == [True, True, True, True]

    bargain_state = _make_run_state(911)
    bargain_state.current_act_index = 1
    bargain_state.player.gold = 400
    bargain_event = WelcomeToWongos()
    relics_before = len(bargain_state.player.relics)
    gold_before = bargain_state.player.gold
    points_before = bargain_state.player.wongo_points
    bargain = bargain_event.choose(bargain_state, "bargain_bin")
    assert bargain.finished
    assert bargain_state.player.gold == gold_before - 100
    assert len(bargain_state.player.relics) == relics_before + 1
    assert bargain_state.player.wongo_points == points_before + 32

    featured_state = _make_run_state(912)
    featured_state.current_act_index = 1
    featured_state.player.gold = 400
    featured_event = WelcomeToWongos()
    featured_event.generate_initial_options(featured_state)
    featured_id = featured_event._featured_relic_id
    relics_before = len(featured_state.player.relics)
    gold_before = featured_state.player.gold
    points_before = featured_state.player.wongo_points
    featured = featured_event.choose(featured_state, "featured")
    assert featured.finished
    assert featured_state.player.gold == gold_before - 200
    assert len(featured_state.player.relics) == relics_before + 1
    assert featured_state.player.relics[-1] == featured_id
    assert featured_state.player.wongo_points == points_before + 16

    mystery_state = _make_run_state(913)
    mystery_state.current_act_index = 1
    mystery_state.player.gold = 400
    mystery_event = WelcomeToWongos()
    relics_before = len(mystery_state.player.relics)
    gold_before = mystery_state.player.gold
    points_before = mystery_state.player.wongo_points
    mystery = mystery_event.choose(mystery_state, "mystery")
    assert mystery.finished
    assert mystery_state.player.gold == gold_before - 300
    assert len(mystery_state.player.relics) == relics_before + 1
    assert "WONGOS_MYSTERY_TICKET" in mystery_state.player.relics
    assert mystery_state.player.wongo_points == points_before + 8

    badge_state = _make_run_state(9131)
    badge_state.current_act_index = 1
    badge_state.player.gold = 400
    badge_state.player.wongo_points = 1992
    badge_event = WelcomeToWongos()
    badge = badge_event.choose(badge_state, "mystery")
    assert badge.finished
    assert badge_state.player.wongo_points == 2000
    assert "WONGO_CUSTOMER_APPRECIATION_BADGE" in badge_state.player.relics
    assert not badge.rewards

    mgr = RunManager(seed=9132, character_id="Ironclad")
    mgr.run_state.current_act_index = 1
    mgr.run_state.player.gold = 400
    mgr.run_state.player.wongo_points = 1992
    mgr._phase = RunManager.PHASE_EVENT
    mgr._event_model = WelcomeToWongos()
    mgr._event_options = mgr._event_model.generate_initial_options(mgr.run_state)
    result = mgr._do_event_choice({"option_id": "mystery"})
    assert result["phase"] == RunManager.PHASE_CARD_REWARD
    reward_ids = [getattr(mgr._current_reward, "relic_id", None)]
    reward_ids.extend(reward.relic_id for reward in mgr._pending_rewards if hasattr(reward, "relic_id"))
    assert "WONGOS_MYSTERY_TICKET" in reward_ids
    assert "WONGO_CUSTOMER_APPRECIATION_BADGE" in reward_ids

    leave_state = _make_run_state(914)
    leave_state.current_act_index = 1
    leave_state.player.gold = 300
    leave_state.player.deck = [
        create_card(CardId.BASH, upgraded=True),
        create_card(CardId.STRIKE_IRONCLAD),
    ]
    leave_event = WelcomeToWongos()
    up_front_counter = leave_state.rng.up_front.counter
    leave = leave_event.choose(leave_state, "leave")
    assert leave.finished
    assert leave_state.player.deck[0].upgraded is False
    assert leave_state.rng.up_front.counter == up_front_counter


def test_welcome_to_wongos_requires_all_players_to_have_initial_gold():
    run_state = _make_run_state(9140)
    run_state.current_act_index = 1
    run_state.player.gold = 100
    ally = run_state.add_player(PlayerState(player_id=2, character_id="Silent", gold=99))
    event = WelcomeToWongos()

    assert event.is_allowed(run_state) is False

    ally.gold = 100
    assert event.is_allowed(run_state) is True


def test_welcome_to_wongos_leave_uses_single_event_rng_choice():
    run_state = _make_run_state(9141)
    run_state.current_act_index = 1
    run_state.player.gold = 300
    first = create_card(CardId.BASH, upgraded=True)
    second = create_card(CardId.STRIKE_IRONCLAD, upgraded=True)
    run_state.player.deck = [first, second]
    event = WelcomeToWongos()
    event.rng = _LastChoiceRng()
    up_front_counter = run_state.rng.up_front.counter
    niche_counter = run_state.rng.niche.counter
    rewards_counter = run_state.rng.rewards.counter

    result = event.choose(run_state, "leave")

    assert result.finished
    assert event.rng.choice_calls == 1
    assert run_state.rng.up_front.counter == up_front_counter
    assert run_state.rng.niche.counter == niche_counter
    assert run_state.rng.rewards.counter == rewards_counter
    assert first.upgraded is True
    assert second.upgraded is False


def test_field_of_man_sized_holes_gate_resist_and_enter_behaviors():
    blocked_state = _make_run_state(915)
    blocked_state.player.deck = [make_spore_mind()]
    event = FieldOfManSizedHoles()
    assert event.is_allowed(blocked_state) is False

    resist_state = _make_run_state(916)
    deck_before = len(resist_state.player.deck)
    resist = event.choose(resist_state, "resist")
    assert resist.finished is False
    assert event.pending_choice is not None
    event.resolve_pending_choice(0)
    event.resolve_pending_choice(1)
    final = event.resolve_pending_choice(None)
    assert final.finished
    assert len(resist_state.player.deck) == deck_before - 1
    assert any(card.card_id == CardId.NORMALITY for card in resist_state.player.deck)

    enter_state = _make_run_state(917)
    enter = event.choose(enter_state, "enter")
    assert not enter.finished
    assert event.pending_choice is not None
    target = event.pending_choice.options[0].card
    resolved = event.resolve_pending_choice(0)
    assert resolved.finished
    assert target.enchantments.get("PerfectFit") == 1


def test_field_of_man_sized_holes_requires_all_players_to_have_enchantable_cards():
    run_state = _make_run_state(9171)
    ally = run_state.add_player(PlayerState(player_id=2, character_id="Silent"))
    ally.deck = [make_spore_mind()]
    event = FieldOfManSizedHoles()

    assert event.is_allowed(run_state) is False

    ally.deck = [create_card(CardId.STRIKE_IRONCLAD)]
    assert event.is_allowed(run_state) is True
