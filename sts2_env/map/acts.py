"""Per-act configuration: room counts, encounter pools, event pools, boss pools.

Based on decompiled MegaCrit.Sts2.Core.Models.Acts source.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ActConfig:
    """Configuration for a single act."""

    act_index: int
    num_rooms: int  # Number of room rows (used as mapLength input)
    act_id: str = ""
    num_weak_encounters: int = 3  # C# NumberOfWeakEncounters (3 for Acts 0/3, 2 for Acts 1/2)
    boss_ids: list[str] = field(default_factory=list)
    elite_ids: list[str] = field(default_factory=list)
    weak_encounter_ids: list[str] = field(default_factory=list)
    strong_encounter_ids: list[str] = field(default_factory=list)
    event_ids: list[str] = field(default_factory=list)
    events_visited: int = 0

    def to_mutable(self) -> "ActConfig":
        return ActConfig(
            act_index=self.act_index,
            num_rooms=self.num_rooms,
            act_id=self.act_id,
            num_weak_encounters=self.num_weak_encounters,
            boss_ids=list(self.boss_ids),
            elite_ids=list(self.elite_ids),
            weak_encounter_ids=list(self.weak_encounter_ids),
            strong_encounter_ids=list(self.strong_encounter_ids),
            event_ids=list(self.event_ids),
            events_visited=self.events_visited,
        )


# ── Act definitions ───────────────────────────────────────────────────

ACT_0 = ActConfig(
    act_index=0,
    num_rooms=15,
    act_id="Overgrowth",
    boss_ids=["TheLich"],
    elite_ids=["SentryAndSentry", "GremlinNob", "BookOfStabbing"],
    weak_encounter_ids=[
        "TwoLouses", "ThreeJawWorms", "SmallSlimes",
        "Cultist", "GremlinGang",
    ],
    strong_encounter_ids=[
        "BlueSlaver", "RedSlaver", "FungiBeast",
        "LooterGroup", "ExordiumWildlife", "LotOfSlimes",
    ],
    event_ids=[
        "AromaOfChaos", "ByrdonisNest", "DenseVegetation",
        "JungleMazeAdventure", "LuminousChoir", "MorphicGrove",
        "SapphireSeed", "SunkenStatue", "TabletOfTruth", "UnrestSite",
        "Wellspring", "WhisperingHollow", "WoodCarvings",
    ],
)

ACT_0_UNDERDOCKS = ActConfig(
    act_index=0,
    num_rooms=15,
    act_id="Underdocks",
    num_weak_encounters=3,
    event_ids=[
        "AbyssalBaths", "DrowningBeacon", "EndlessConveyor", "PunchOff",
        "SpiralingWhirlpool", "SunkenStatue", "SunkenTreasury",
        "DoorsOfLightAndDark", "TrashHeap", "WaterloggedScriptorium",
    ],
)

ACT_1 = ActConfig(
    act_index=1,
    num_rooms=14,  # C# Hive.BaseNumberOfRooms = 14
    act_id="Hive",
    num_weak_encounters=2,  # C# Hive.NumberOfWeakEncounters = 2
    boss_ids=["TheCollector", "Automaton", "Champ"],
    elite_ids=["TaskMaster", "SphericGuardian", "Snecko"],
    weak_encounter_ids=[
        "SnakePlant", "Centurion", "ThreeByrds",
    ],
    strong_encounter_ids=[
        "SlaverGroup", "BookOfStabbing", "MushroomGroup",
    ],
    event_ids=[
        "Amalgamator", "Bugslayer", "ColorfulPhilosophers",
        "ColossalFlower", "FieldOfManSizedHoles", "InfestedAutomaton",
        "LostWisp", "SpiritGrafter", "TheLanternKey", "ZenWeaver",
    ],
)

ACT_2 = ActConfig(
    act_index=2,
    num_rooms=13,  # C# Glory.BaseNumberOfRooms = 13
    act_id="Glory",
    num_weak_encounters=2,  # C# Glory.NumberOfWeakEncounters = 2
    boss_ids=["AwakenedOne", "TimeEater", "DonuAndDeca"],
    elite_ids=["GiantHead", "Nemesis", "Reptomancer"],
    weak_encounter_ids=[
        "Darkling", "OrbWalker",
    ],
    strong_encounter_ids=[
        "WrithingMass", "Transient", "Maw",
    ],
    event_ids=[
        "BattlewornDummy", "GraveOfTheForgotten", "HungryForMushrooms",
        "Reflections", "RoundTeaParty", "Trial", "TinkerTime",
    ],
)

ALL_ACTS = [ACT_0, ACT_1, ACT_2]


def get_act_config(act_index: int) -> ActConfig:
    if 0 <= act_index < len(ALL_ACTS):
        return ALL_ACTS[act_index]
    raise ValueError(f"Invalid act index: {act_index}")
