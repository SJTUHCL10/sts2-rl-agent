using System.Text.Json;
using System.Text.Json.Nodes;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;
using MegaCrit.Sts2.Core.Map;
using MegaCrit.Sts2.Core.Models;
using MegaCrit.Sts2.Core.Runs;

namespace STS2AgentShared;

/// <summary>
/// Shared, side-effect-free v2 protocol enrichment for Bridge and Advisor.
/// V1 fields are preserved so old Python clients and checkpoints still work.
/// </summary>
public static class ProtocolV2
{
    public const int ProtocolVersion = 2;
    public const string ObservationSchema = "sts2-entity-v2";
    public const string ActionSchema = "candidate-v2";
    public const string FeatureLayoutHash =
        "f30670de5d3a336cb9c4bbfb2ee2ad942958d98ac39260a0db4ba23fba922223";

    public static string EnrichStateJson(
        string stateJson,
        string? episodeId = null,
        string? decisionId = null)
    {
        JsonObject state = JsonNode.Parse(stateJson)?.AsObject()
            ?? throw new JsonException("State payload is not a JSON object.");
        state["protocol_version"] = ProtocolVersion;
        state["observation_schema"] = ObservationSchema;
        state["action_schema"] = ActionSchema;
        state["feature_layout_hash"] = FeatureLayoutHash;
        if (episodeId != null) state["episode_id"] = episodeId;
        if (decisionId != null) state["decision_id"] = decisionId;
        AddRunSnapshot(state);
        AddEntityIds(state);
        state["candidates"] = BuildCandidates(state);
        return state.ToJsonString();
    }

    /// <summary>
    /// Resolve a candidate response centrally, leaving all existing handlers
    /// on their proven legacy action payloads.
    /// </summary>
    public static string ResolveCandidateAction(string enrichedState, string actionJson)
    {
        JsonObject action = JsonNode.Parse(actionJson)?.AsObject()
            ?? throw new JsonException("Action payload is not a JSON object.");
        if (!string.Equals(
            action["action"]?.GetValue<string>(),
            "candidate",
            StringComparison.OrdinalIgnoreCase))
        {
            return actionJson;
        }
        string candidateId = action["candidate_id"]?.GetValue<string>()
            ?? throw new JsonException("Candidate action omitted candidate_id.");
        JsonObject state = JsonNode.Parse(enrichedState)!.AsObject();
        JsonArray candidates = state["candidates"]?.AsArray() ?? new JsonArray();
        JsonObject? match = candidates
            .OfType<JsonObject>()
            .SingleOrDefault(candidate =>
                candidate["enabled"]?.GetValue<bool>() != false
                && candidate["candidate_id"]?.GetValue<string>() == candidateId);
        if (match?["payload"] is not JsonObject payload)
            throw new JsonException($"Unknown or disabled candidate_id '{candidateId}'.");
        JsonObject resolved = (JsonObject)payload.DeepClone();
        if (action["request_id"] != null)
            resolved["request_id"] = action["request_id"]!.DeepClone();
        resolved["candidate_id"] = candidateId;
        return resolved.ToJsonString();
    }

    /// <summary>
    /// Public card features shared by the automatic Bridge and passive
    /// Advisor.  PreviewValue is the value currently shown to the player;
    /// BaseValue and EnchantedValue preserve the underlying card instance.
    /// </summary>
    public static JsonObject SerializeCardFeatures(CardModel card)
    {
        var result = new JsonObject
        {
            ["original_cost"] = card.EnergyCost.Canonical,
            ["upgrade_level"] = card.CurrentUpgradeLevel,
            ["dynamic_vars"] = SerializeDynamicVars(card.DynamicVars),
        };
        if (card.Enchantment != null)
        {
            result["enchantments"] = new JsonObject
            {
                [card.Enchantment.Id.Entry] = card.Enchantment.Amount,
            };
        }
        if (card.Affliction != null)
        {
            result["afflictions"] = new JsonObject
            {
                [card.Affliction.Id.Entry] = card.Affliction.Amount,
            };
        }
        return result;
    }

    public static JsonObject SerializePowerState(PowerModel power)
    {
        return new JsonObject
        {
            ["power_type"] = power.Type.ToString(),
            ["stack_type"] = power.StackType.ToString(),
            ["amount"] = power.Amount,
            ["display_amount"] = power.DisplayAmount,
            ["amount_on_turn_start"] = power.AmountOnTurnStart,
            ["skip_next_duration_tick"] = power.SkipNextDurationTick,
            ["is_visible"] = power.IsVisible,
            ["dynamic_vars"] = SerializeDynamicVars(power.DynamicVars),
            ["counters"] = new JsonObject
            {
                ["display_amount"] = power.DisplayAmount,
            },
        };
    }

    public static JsonObject SerializeRelicState(RelicModel relic)
    {
        return new JsonObject
        {
            ["rarity"] = relic.Rarity.ToString(),
            ["stack_count"] = relic.StackCount,
            ["status"] = relic.Status.ToString(),
            ["is_used_up"] = relic.IsUsedUp,
            ["is_melted"] = relic.IsMelted,
            ["is_wax"] = relic.IsWax,
            ["show_counter"] = relic.ShowCounter,
            ["display_amount"] = relic.DisplayAmount,
            ["floor_added"] = relic.FloorAddedToDeck,
            ["dynamic_vars"] = SerializeDynamicVars(relic.DynamicVars),
            ["counters"] = relic.ShowCounter
                ? new JsonObject { ["display_amount"] = relic.DisplayAmount }
                : new JsonObject(),
        };
    }

    private static JsonObject SerializeDynamicVars(
        MegaCrit.Sts2.Core.Localization.DynamicVars.DynamicVarSet vars)
    {
        var result = new JsonObject();
        foreach (var pair in vars)
        {
            result[pair.Key] = new JsonObject
            {
                ["base"] = pair.Value.BaseValue,
                ["enchanted"] = pair.Value.EnchantedValue,
                ["preview"] = pair.Value.PreviewValue,
            };
        }
        return result;
    }

    private static void AddRunSnapshot(JsonObject state)
    {
        try
        {
            RunState? run = RunManager.Instance?.DebugOnlyGetState();
            if (run == null) return;
            var players = new JsonArray();
            var cards = new JsonArray();
            var relics = new JsonArray();
            var potions = new JsonArray();
            foreach (Player player in run.Players)
            {
                string ownerId = $"player:{player.NetId}";
                players.Add(new JsonObject
                {
                    ["entity_id"] = ownerId,
                    ["player_id"] = player.NetId.ToString(),
                    ["character_id"] = player.Character.Id.Entry,
                    ["hp"] = player.Creature.CurrentHp,
                    ["max_hp"] = player.Creature.MaxHp,
                    ["gold"] = player.Gold,
                    ["max_energy"] = player.MaxEnergy,
                    ["max_potion_slots"] = player.MaxPotionCount,
                    ["base_orb_slot_count"] = player.BaseOrbSlotCount,
                });
                int cardIndex = 0;
                foreach (CardModel card in player.Deck.Cards)
                {
                    cards.Add(SerializeRunCard(card, ownerId, cardIndex++));
                }
                int relicIndex = 0;
                foreach (RelicModel relic in player.Relics)
                {
                    string relicId = relic.Id.Entry;
                    JsonObject relicState = SerializeRelicState(relic);
                    var serializedRelic = new JsonObject
                    {
                        ["entity_id"] = $"relic:{ownerId}:{relicIndex}:{relicId}",
                        ["owner_id"] = ownerId,
                        ["relic_id"] = relicId,
                        ["id"] = relicId,
                        ["state"] = relicState.DeepClone(),
                    };
                    foreach (var feature in relicState)
                        serializedRelic[feature.Key] = feature.Value?.DeepClone();
                    relics.Add(serializedRelic);
                    relicIndex++;
                }
                int slot = 0;
                foreach (dynamic? potion in player.PotionSlots)
                {
                    if (potion != null)
                    {
                        string potionId = potion.Id.Entry;
                        potions.Add(new JsonObject
                        {
                            ["entity_id"] = $"potion:{ownerId}:{slot}",
                            ["owner_id"] = ownerId,
                            ["slot"] = slot,
                            ["potion_id"] = potionId,
                            ["id"] = potionId,
                            ["usage"] = potion.Usage.ToString(),
                            ["target_type"] = potion.TargetType.ToString(),
                            ["can_use"] = player.CanUseOrRemovePotions,
                        });
                    }
                    slot++;
                }
            }

            var mapNodes = new JsonArray();
            var mapEdges = new JsonArray();
            HashSet<MapCoord> visited = run.VisitedMapCoords.ToHashSet();
            HashSet<MapCoord> reachable = run.CurrentMapPoint == null
                ? run.Map.GetAllMapPoints().Where(p => p.coord.row == 0).Select(p => p.coord).ToHashSet()
                : run.CurrentMapPoint.Children.Select(p => p.coord).ToHashSet();
            foreach (MapPoint point in run.Map.GetAllMapPoints().OrderBy(p => p.coord.row).ThenBy(p => p.coord.col))
            {
                string nodeId = $"map:{point.coord.col}:{point.coord.row}";
                mapNodes.Add(new JsonObject
                {
                    ["entity_id"] = nodeId,
                    ["row"] = point.coord.row,
                    ["col"] = point.coord.col,
                    ["node_type"] = point.PointType.ToString(),
                    ["visited"] = visited.Contains(point.coord),
                    ["reachable"] = reachable.Contains(point.coord),
                });
                foreach (MapPoint child in point.Children)
                {
                    mapEdges.Add(new JsonObject
                    {
                        ["source_id"] = nodeId,
                        ["target_id"] = $"map:{child.coord.col}:{child.coord.row}",
                    });
                }
            }

            Player? primary = LocalContext.GetMe(run);
            string primaryId = $"player:{primary?.NetId}";
            JsonNode? primaryPlayer = players
                .OfType<JsonObject>()
                .FirstOrDefault(player =>
                    player["entity_id"]?.GetValue<string>() == primaryId)
                ?.DeepClone()
                ?? (players.Count > 0 ? players[0]?.DeepClone() : null);
            state["run_state"] = new JsonObject
            {
                ["character_id"] = primary?.Character.Id.Entry ?? "",
                ["act"] = run.CurrentActIndex + 1,
                ["act_floor"] = run.ActFloor,
                ["floor"] = run.TotalFloor,
                ["ascension"] = run.AscensionLevel,
                ["room_type"] = run.CurrentMapPoint?.PointType.ToString() ?? "",
                // Compatibility projections for the frozen 151x157 adapter.
                ["player"] = primaryPlayer,
                ["gold"] = primary?.Gold ?? 0,
                ["max_potion_slots"] = primary?.MaxPotionCount ?? 0,
                ["players"] = players,
                ["cards"] = cards,
                ["deck"] = cards.DeepClone(),
                ["relics"] = relics,
                ["potions"] = potions,
                ["map_nodes"] = mapNodes,
                ["map_edges"] = mapEdges,
            };
        }
        catch
        {
            // Brief game transitions can have no valid RunState.  Presence of
            // the schema envelope still lets Python distinguish missing data
            // from numeric zero.
            state["run_state_available"] = false;
            return;
        }
        state["run_state_available"] = true;
    }

    private static JsonObject SerializeRunCard(CardModel card, string ownerId, int index)
    {
        int cost;
        try { cost = card.EnergyCost.GetWithModifiers(CostModifiers.All); }
        catch { cost = card.EnergyCost.Canonical; }
        var result = new JsonObject
        {
            ["entity_id"] = $"card:{ownerId}:deck:{index}",
            ["card_id"] = card.Id.Entry,
            ["id"] = card.Id.Entry,
            ["owner_id"] = ownerId,
            ["zone"] = "deck",
            ["zone_index"] = index,
            ["cost"] = cost,
            ["original_cost"] = card.EnergyCost.Canonical,
            ["card_type"] = card.Type.ToString(),
            ["type"] = card.Type.ToString(),
            ["target_type"] = card.TargetType.ToString(),
            ["target"] = card.TargetType.ToString(),
            ["upgraded"] = card.IsUpgraded,
        };
        foreach (var feature in SerializeCardFeatures(card))
        {
            result[feature.Key] = feature.Value?.DeepClone();
        }
        return result;
    }

    private static void AddEntityIds(JsonObject state)
    {
        AddIds(state["hand"] as JsonArray, "hand");
        AddIds(state["enemies"] as JsonArray, "enemy");
        AddIds(state["potions"] as JsonArray, "potion", "slot");
        AddIds(state["nodes"] as JsonArray, "map");
        AddIds(state["options"] as JsonArray, "option");
        AddIds(state["cards"] as JsonArray, "choice-card");
        AddIds(state["bundles"] as JsonArray, "bundle");
    }

    private static void AddIds(JsonArray? items, string prefix, string indexKey = "index")
    {
        if (items == null) return;
        for (int i = 0; i < items.Count; i++)
        {
            if (items[i] is not JsonObject item || item["entity_id"] != null) continue;
            string id = item["id"]?.ToString() ?? "unknown";
            string index = item[indexKey]?.ToString() ?? i.ToString();
            item["entity_id"] = $"{prefix}:{id}:{index}";
        }
    }

    private static JsonArray BuildCandidates(JsonObject state)
    {
        var candidates = new JsonArray();
        string type = state["type"]?.GetValue<string>() ?? "";
        if (type == "combat_action")
        {
            AddCandidate(candidates, "combat:end_turn", "END_TURN", null, null,
                new JsonObject { ["action"] = "end_turn" });
            JsonArray hand = state["hand"] as JsonArray ?? new JsonArray();
            JsonArray enemies = state["enemies"] as JsonArray ?? new JsonArray();
            for (int i = 0; i < hand.Count; i++)
            {
                if (hand[i] is not JsonObject card || card["playable"]?.GetValue<bool>() == false) continue;
                string source = card["entity_id"]!.GetValue<string>();
                string targetType = Canonical(card["target"]?.ToString());
                if (targetType is "anyenemy" or "randomenemy")
                {
                    for (int j = 0; j < enemies.Count; j++)
                    {
                        if (enemies[j] is not JsonObject enemy || enemy["is_alive"]?.GetValue<bool>() != true) continue;
                        string target = enemy["entity_id"]!.GetValue<string>();
                        AddCandidate(candidates, $"combat:play:{source}:{target}", "PLAY_CARD", source, target,
                            new JsonObject { ["action"] = "play", ["card_index"] = i, ["target_index"] = j });
                    }
                }
                else
                {
                    AddCandidate(candidates, $"combat:play:{source}:none", "PLAY_CARD", source, null,
                        new JsonObject { ["action"] = "play", ["card_index"] = i, ["target_index"] = -1 });
                }
            }
            JsonArray potions = state["potions"] as JsonArray ?? new JsonArray();
            for (int i = 0; i < potions.Count; i++)
            {
                if (potions[i] is not JsonObject potion || potion["can_use"]?.GetValue<bool>() == false) continue;
                int slot = potion["slot"]?.GetValue<int>() ?? i;
                string source = potion["entity_id"]!.GetValue<string>();
                bool targeted = potion["requires_target"]?.GetValue<bool>() == true
                    || Canonical(potion["target"]?.ToString()) == "anyenemy";
                if (targeted)
                {
                    for (int j = 0; j < enemies.Count; j++)
                    {
                        if (enemies[j] is not JsonObject enemy || enemy["is_alive"]?.GetValue<bool>() != true) continue;
                        string target = enemy["entity_id"]!.GetValue<string>();
                        AddCandidate(candidates, $"combat:potion:{source}:{target}", "USE_POTION", source, target,
                            new JsonObject { ["action"] = "potion", ["slot"] = slot, ["target_index"] = j });
                    }
                }
                else
                {
                    AddCandidate(candidates, $"combat:potion:{source}:none", "USE_POTION", source, null,
                        new JsonObject { ["action"] = "potion", ["slot"] = slot, ["target_index"] = -1 });
                }
            }
            return candidates;
        }

        string collection = state["nodes"] != null ? "nodes"
            : state["options"] != null ? "options"
            : state["cards"] != null ? "cards"
            : state["bundles"] != null ? "bundles" : "options";
        JsonArray items = state[collection] as JsonArray ?? new JsonArray();
        for (int i = 0; i < items.Count; i++)
        {
            if (items[i] is not JsonObject item || item["enabled"]?.GetValue<bool>() == false) continue;
            int index = item["index"]?.GetValue<int>() ?? i;
            string source = item["entity_id"]!.GetValue<string>();
            string actionType = (item["action"]?.ToString() ?? type switch
            {
                "map_select" => "MOVE",
                "card_reward" => "PICK_CARD",
                "card_select" => "SELECT_CARD",
                "card_bundle" => "PICK_CARD_BUNDLE",
                "boss_relic" => "PICK_RELIC",
                _ => "CHOOSE",
            }).ToUpperInvariant();
            string candidateId = $"{type}:{actionType.ToLowerInvariant()}:{source}";
            var features = new JsonObject();
            if (type == "crystal_sphere")
            {
                if (item["x"] != null && item["y"] != null)
                {
                    int x = item["x"]!.GetValue<int>();
                    int y = item["y"]!.GetValue<int>();
                    candidateId = $"crystal_sphere:divine:{x}:{y}";
                    features["x"] = x;
                    features["y"] = y;
                    features["affected_cell_ids"] =
                        item["affected_cell_ids"]?.DeepClone();
                }
                else
                {
                    candidateId = "crystal_sphere:proceed";
                }
            }
            AddCandidate(candidates, candidateId, actionType, source, null,
                new JsonObject { ["action"] = "choose", ["index"] = index },
                features);
        }
        bool optionalCardSelection = type == "card_select"
            && state["min_select"]?.GetValue<int>() == 0;
        if (state["can_skip"]?.GetValue<bool>() == true || optionalCardSelection)
        {
            AddCandidate(candidates, $"{type}:skip", "SKIP", null, null,
                new JsonObject { ["action"] = "skip" });
        }
        return candidates;
    }

    private static void AddCandidate(
        JsonArray output,
        string candidateId,
        string actionType,
        string? sourceId,
        string? targetId,
        JsonObject payload,
        JsonObject? features = null)
    {
        output.Add(new JsonObject
        {
            ["candidate_id"] = candidateId,
            ["action_type"] = actionType,
            ["source_id"] = sourceId,
            ["target_id"] = targetId,
            ["enabled"] = true,
            ["payload"] = payload,
            ["features"] = features ?? new JsonObject(),
        });
    }

    private static string Canonical(string? value) =>
        new string((value ?? "").Where(char.IsLetterOrDigit).ToArray()).ToLowerInvariant();
}
