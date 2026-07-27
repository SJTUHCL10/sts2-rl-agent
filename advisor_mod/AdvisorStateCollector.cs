using System.Text.Json;
using System.Text.Json.Nodes;
using System.Reflection;
using Godot;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Creatures;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.MonsterMoves.Intents;
using MegaCrit.Sts2.Core.Nodes;
using MegaCrit.Sts2.Core.Nodes.Cards.Holders;
using MegaCrit.Sts2.Core.Nodes.Screens.CardSelection;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Runs;
using STS2AgentShared;

namespace STS2AdvisorMod;

internal static class AdvisorStateCollector
{
    public static bool TryCollect(out Dictionary<string, object?> state)
    {
        state = null!;
        try
        {
            if (TryCollectCardReward(out state)) return true;
            if (TryCollectCombat(out state)) return true;
            if (TryCollectMap(out state)) return true;
        }
        catch (Exception ex)
        {
            AdvisorLog.Info($"State collection failed: {ex.Message}");
        }
        return false;
    }

    private static bool TryCollectCombat(out Dictionary<string, object?> state)
    {
        state = null!;
        CombatManager? manager = CombatManager.Instance;
        if (manager == null || !manager.IsInProgress || manager.PlayerActionsDisabled) return false;
        ICombatState? combat = manager.DebugOnlyGetState();
        if (combat == null) return false;
        Player? player = LocalContext.GetMe(combat);
        PlayerCombatState? pcs = player?.PlayerCombatState;
        if (player == null || pcs?.Phase != PlayerTurnPhase.Play) return false;

        var playerObj = new Dictionary<string, object?>
        {
            ["hp"] = player.Creature.CurrentHp,
            ["max_hp"] = player.Creature.MaxHp,
            ["block"] = player.Creature.Block,
            ["energy"] = pcs.Energy,
            ["max_energy"] = pcs.MaxEnergy,
            ["powers"] = SerializePowers(player.Creature),
        };
        var hand = pcs.Hand.Cards
            .Select((card, index) => SerializeCard(card, "hand", index))
            .ToList();
        var enemies = combat.Enemies.Select(SerializeEnemy).ToList();
        var potions = SerializePotions(player);
        RunState? run = SafeRunState();

        state = new Dictionary<string, object?>
        {
            ["type"] = "combat_action",
            ["player"] = playerObj,
            ["hand"] = hand,
            ["cards"] = SerializeCombatCards(pcs),
            ["enemies"] = enemies,
            ["potions"] = potions,
            ["available_actions"] = potions.Count > 0
                ? new[] { "PLAY", "END_TURN", "POTION" }
                : new[] { "PLAY", "END_TURN" },
            ["draw_pile_count"] = pcs.DrawPile.Cards.Count,
            ["discard_pile_count"] = pcs.DiscardPile.Cards.Count,
            ["exhaust_pile_count"] = pcs.ExhaustPile.Cards.Count,
            ["round"] = combat.RoundNumber,
            ["floor"] = run?.TotalFloor ?? 0,
            ["act"] = (run?.CurrentActIndex ?? 0) + 1,
        };
        return true;
    }

    private static bool TryCollectCardReward(out Dictionary<string, object?> state)
    {
        state = null!;
        if (NOverlayStack.Instance == null || NOverlayStack.Instance.ScreenCount == 0) return false;
        if (NOverlayStack.Instance.Peek() is not NCardRewardSelectionScreen screen || !screen.IsVisibleInTree())
            return false;
        List<CardModel> optionCards = GetStableCardRewardOptions(screen);
        if (optionCards.Count == 0) return false;
        var cards = optionCards.Select((card, index) => SerializeRewardCard(card, index)).ToList();
        AddRunFields(out Dictionary<string, object?> runFields);
        state = new Dictionary<string, object?>(runFields)
        {
            ["type"] = "card_reward",
            ["cards"] = cards,
            ["can_skip"] = true,
        };
        return true;
    }

    private static bool TryCollectMap(out Dictionary<string, object?> state)
    {
        state = null!;
        SceneTree tree = (SceneTree)Engine.GetMainLoop();
        NRun? runNode = tree.Root.GetNodeOrNull<NRun>("/root/Game/RootSceneContainer/Run");
        NMapScreen? screen = runNode?.GlobalUi?.MapScreen;
        if (screen == null || !screen.IsVisibleInTree()) return false;
        RunState? run = SafeRunState();
        if (run == null) return false;
        List<NMapPoint> points = FindAll<NMapPoint>(screen);
        List<NMapPoint> available;
        if (run.VisitedMapCoords.Count == 0)
        {
            available = points.Where(point => point.Point.coord.row == 0 && point.IsEnabled).ToList();
        }
        else
        {
            MapCoord last = run.VisitedMapCoords[^1];
            NMapPoint? previous = points.FirstOrDefault(point => point.Point.coord.Equals(last));
            if (previous == null) return false;
            var children = previous.Point.Children.Select(child => child.coord).ToHashSet();
            available = points.Where(point => children.Contains(point.Point.coord) && point.IsEnabled).ToList();
        }
        if (available.Count == 0) return false;
        var nodes = available.Select((point, index) => new Dictionary<string, object?>
        {
            ["index"] = index,
            ["type"] = point.Point.PointType.ToString(),
            ["row"] = point.Point.coord.row,
            ["col"] = point.Point.coord.col,
            ["enabled"] = true,
        }).ToList();
        AddRunFields(out Dictionary<string, object?> runFields);
        state = new Dictionary<string, object?>(runFields)
        {
            ["type"] = "map_select",
            ["nodes"] = nodes,
        };
        return true;
    }

    private static List<Dictionary<string, object?>> SerializeCombatCards(
        PlayerCombatState pcs)
    {
        var cards = new List<Dictionary<string, object?>>();
        AddPile(cards, pcs.Hand.Cards, "hand");
        AddPile(cards, pcs.DrawPile.Cards, "draw");
        AddPile(cards, pcs.DiscardPile.Cards, "discard");
        AddPile(cards, pcs.ExhaustPile.Cards, "exhaust");
        return cards;
    }

    private static void AddPile(
        List<Dictionary<string, object?>> output,
        IReadOnlyList<CardModel> cards,
        string zone)
    {
        for (int index = 0; index < cards.Count; index++)
            output.Add(SerializeCard(cards[index], zone, index));
    }

    private static Dictionary<string, object?> SerializeCard(
        CardModel card,
        string zone = "hand",
        int zoneIndex = -1)
    {
        int cost;
        try { cost = card.EnergyCost.GetWithModifiers(CostModifiers.All); }
        catch { cost = card.EnergyCost.Canonical; }
        UnplayableReason reason;
        AbstractModel? preventer;
        var result = new Dictionary<string, object?>
        {
            ["id"] = card.Id.Entry,
            ["entity_id"] = $"card:player:local:{zone}:{zoneIndex}:{card.Id.Entry}",
            ["zone"] = zone,
            ["zone_index"] = zoneIndex,
            ["cost"] = cost,
            ["type"] = card.Type.ToString(),
            ["target"] = card.TargetType.ToString(),
            ["playable"] = card.CanPlay(out reason, out preventer),
            ["upgraded"] = card.IsUpgraded,
        };
        foreach (KeyValuePair<string, JsonNode?> feature in ProtocolV2.SerializeCardFeatures(card))
            result[feature.Key] = feature.Value;
        AddCardModifiers(result, card);
        return result;
    }

    private static Dictionary<string, object?> SerializeRewardCard(CardModel card, int index)
    {
        var result = new Dictionary<string, object?>
        {
            ["index"] = index,
            ["id"] = card.Id.Entry,
            ["type"] = card.Type.ToString(),
            ["cost"] = card.EnergyCost.Canonical,
            ["upgraded"] = card.IsUpgraded,
        };
        foreach (KeyValuePair<string, JsonNode?> feature in ProtocolV2.SerializeCardFeatures(card))
            result[feature.Key] = feature.Value;
        AddCardModifiers(result, card);
        return result;
    }

    private static void AddCardModifiers(Dictionary<string, object?> result, CardModel card)
    {
        if (card.Enchantment != null)
        {
            result["enchantment"] = new Dictionary<string, object?>
            {
                ["id"] = card.Enchantment.Id.Entry,
                ["amount"] = card.Enchantment.Amount,
            };
        }
        if (card.Affliction != null)
        {
            result["affliction"] = new Dictionary<string, object?>
            {
                ["id"] = card.Affliction.Id.Entry,
                ["amount"] = card.Affliction.Amount,
            };
        }
    }

    private static List<CardModel> GetStableCardRewardOptions(NCardRewardSelectionScreen screen)
    {
        // The screen's semantic option list is stable while hover animation may
        // create or rearrange visual card-holder nodes. Prefer the model list so
        // mouse movement cannot change the advisor observation or action mask.
        FieldInfo? field = typeof(NCardRewardSelectionScreen).GetField(
            "_options", BindingFlags.Instance | BindingFlags.NonPublic);
        if (field?.GetValue(screen) is IReadOnlyList<CardCreationResult> options)
            return options.Select(option => option.Card).ToList();

        return FindAll<NGridCardHolder>(screen)
            .Select(holder => holder.CardModel)
            .Where(card => card != null)
            .Cast<CardModel>()
            .ToList();
    }

    private static Dictionary<string, object?> SerializeEnemy(Creature enemy)
    {
        var result = new Dictionary<string, object?>
        {
            ["id"] = enemy.IsMonster ? enemy.Monster!.Id.Entry : "UNKNOWN",
            ["hp"] = enemy.CurrentHp,
            ["max_hp"] = enemy.MaxHp,
            ["block"] = enemy.Block,
            ["is_alive"] = enemy.IsAlive,
            ["powers"] = SerializePowers(enemy),
        };
        try
        {
            var move = enemy.Monster?.NextMove;
            AbstractIntent? intent = move?.Intents.FirstOrDefault();
            if (intent != null)
            {
                result["intent"] = intent.IntentType.ToString();
                result["intent_move_id"] = move!.Id;
                if (intent is AttackIntent attack)
                {
                    ICombatState? combat = enemy.CombatState;
                    if (combat != null)
                    {
                        result["intent_damage"] = attack.GetSingleDamage(combat.PlayerCreatures, enemy);
                        result["intent_hits"] = attack.Repeats > 0 ? attack.Repeats : 1;
                    }
                }
            }
        }
        catch { result["intent"] = "UNKNOWN"; }
        return result;
    }

    private static List<Dictionary<string, object?>> SerializePowers(Creature creature) =>
        creature.Powers.Select(power =>
        {
            var result = new Dictionary<string, object?>
            {
                ["id"] = power.Id.Entry,
                ["amount"] = power.Amount,
            };
            foreach (KeyValuePair<string, JsonNode?> feature in ProtocolV2.SerializePowerState(power))
                result[feature.Key] = feature.Value;
            return result;
        }).ToList();

    private static List<Dictionary<string, object?>> SerializePotions(Player player)
    {
        var result = new List<Dictionary<string, object?>>();
        int slot = 0;
        foreach (dynamic? potion in player.PotionSlots)
        {
            if (potion != null)
            {
                string target = "Self";
                bool canUse = true;
                try { target = potion.TargetType?.ToString() ?? "Self"; } catch { }
                try { canUse = !string.Equals(potion.Usage?.ToString(), "Automatic", StringComparison.OrdinalIgnoreCase); } catch { }
                result.Add(new Dictionary<string, object?>
                {
                    ["slot"] = slot,
                    ["id"] = potion.Id.Entry,
                    ["can_use"] = canUse,
                    ["target"] = target,
                    ["requires_target"] = target == "AnyEnemy",
                    ["target_type"] = target,
                });
            }
            slot++;
        }
        return result;
    }

    private static RunState? SafeRunState()
    {
        try { return RunManager.Instance?.DebugOnlyGetState(); }
        catch { return null; }
    }

    private static void AddRunFields(out Dictionary<string, object?> fields)
    {
        RunState? run = SafeRunState();
        Player? player = run == null ? null : LocalContext.GetMe(run);
        fields = new Dictionary<string, object?>
        {
            ["floor"] = run?.TotalFloor ?? 0,
            ["act"] = (run?.CurrentActIndex ?? 0) + 1,
            ["player"] = new Dictionary<string, object?>
            {
                ["hp"] = player?.Creature.CurrentHp ?? 0,
                ["max_hp"] = player?.Creature.MaxHp ?? 1,
            },
        };
    }

    private static List<T> FindAll<T>(Node root) where T : Node
    {
        var matches = new List<T>();
        foreach (Node child in root.GetChildren())
        {
            if (child is T match) matches.Add(match);
            matches.AddRange(FindAll<T>(child));
        }
        return matches;
    }
}
