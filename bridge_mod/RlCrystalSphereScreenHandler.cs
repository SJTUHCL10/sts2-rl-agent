// RlCrystalSphereScreenHandler.cs -- bridge-driven Crystal Sphere minigame handler.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Godot;
using MegaCrit.Sts2.Core.AutoSlay;
using MegaCrit.Sts2.Core.AutoSlay.Handlers;
using MegaCrit.Sts2.Core.AutoSlay.Helpers;
using MegaCrit.Sts2.Core.Events.Custom.CrystalSphereEvent;
using MegaCrit.Sts2.Core.Nodes.CommonUi;
using MegaCrit.Sts2.Core.Nodes.Events.Custom.CrystalSphere;
using MegaCrit.Sts2.Core.Nodes.GodotExtensions;
using MegaCrit.Sts2.Core.Nodes.Screens.Map;
using MegaCrit.Sts2.Core.Nodes.Screens.Overlays;
using MegaCrit.Sts2.Core.Random;
using MegaCrit.Sts2.Core.Runs;

namespace STS2BridgeMod;

public class RlCrystalSphereScreenHandler : IScreenHandler, IHandler
{
    private const int AgentTimeoutSeconds = 30;
    private const int HandlerTimeoutSeconds = 120;
    private const int InitialSettleDelayMs = 1000;
    private const int ClickSettleDelayMs = 500;
    private const int WaitForOutcomeTimeoutSeconds = 15;
    private const int ProceedCloseTimeoutSeconds = 10;
    private const int OverlayRemovalDelayMs = 100;
    private const int ActDisplayIndexOffset = 1;
    private const string ScreenLogName = "NCrystalSphereScreen";
    private static readonly TimeSpan AgentTimeout = TimeSpan.FromSeconds(AgentTimeoutSeconds);

    public Type ScreenType => typeof(NCrystalSphereScreen);
    public TimeSpan Timeout => TimeSpan.FromSeconds(HandlerTimeoutSeconds);

    public async Task HandleAsync(Rng random, CancellationToken ct)
    {
        AutoSlayLog.EnterScreen(ScreenLogName);
        NCrystalSphereScreen screen = AutoSlayer.GetCurrentScreen<NCrystalSphereScreen>();
        await Task.Delay(InitialSettleDelayMs, ct);

        while (GodotObject.IsInstanceValid(screen) && screen.IsVisibleInTree())
        {
            ct.ThrowIfCancellationRequested();

            IOverlayScreen overlayScreen = NOverlayStack.Instance?.Peek();
            if (overlayScreen != null && overlayScreen != screen)
            {
                AutoSlayLog.Info("Child screen appeared from Crystal Sphere, returning to drain loop");
                AutoSlayLog.ExitScreen(ScreenLogName);
                return;
            }

            NProceedButton? proceedButton = screen.GetNodeOrNull<NProceedButton>("%ProceedButton");
            List<NCrystalSphereCell> hiddenCells = HiddenCells(screen);
            CrystalSphereChoice choice = await ChooseCrystalSphereOption(
                screen, hiddenCells, proceedButton, random, ct);
            if (choice.ShouldProceed)
            {
                if (proceedButton != null)
                {
                    await ClickProceed(screen, proceedButton, ct);
                }
                AutoSlayLog.ExitScreen(ScreenLogName);
                return;
            }

            if (choice.Cell == null)
            {
                await WaitForOutcome(screen, ct);
                continue;
            }

            AutoSlayLog.Info(
                $"Clicking crystal sphere cell at ({choice.Cell.Entity.X}, {choice.Cell.Entity.Y})");
            choice.Cell.EmitSignal(NClickableControl.SignalName.Released, choice.Cell);
            await Task.Delay(ClickSettleDelayMs, ct);
        }

        AutoSlayLog.ExitScreen(ScreenLogName);
    }

    private static List<NCrystalSphereCell> HiddenCells(NCrystalSphereScreen screen)
    {
        return AllCells(screen)
            .Where(cell => cell.Visible && cell.Entity.IsHidden)
            .ToList();
    }

    private static List<NCrystalSphereCell> AllCells(NCrystalSphereScreen screen)
    {
        Control? cellsContainer = screen.GetNodeOrNull<Control>("%Cells");
        if (cellsContainer == null)
        {
            return new List<NCrystalSphereCell>();
        }
        return UiHelper.FindAll<NCrystalSphereCell>(cellsContainer)
            .OrderBy(cell => cell.Entity.X)
            .ThenBy(cell => cell.Entity.Y)
            .ToList();
    }

    private static async Task<CrystalSphereChoice> ChooseCrystalSphereOption(
        NCrystalSphereScreen screen,
        List<NCrystalSphereCell> hiddenCells,
        NProceedButton? proceedButton,
        Rng random,
        CancellationToken ct)
    {
        bool canProceed = proceedButton?.IsEnabled ?? false;
        if (!BridgeServer.Instance.IsClientConnected)
        {
            if (canProceed)
            {
                return CrystalSphereChoice.Proceed();
            }
            return hiddenCells.Count > 0
                ? CrystalSphereChoice.Click(random.NextItem(hiddenCells))
                : CrystalSphereChoice.Wait();
        }

        try
        {
            List<Dictionary<string, object>> options;
            if (canProceed)
            {
                // Match the native AutoSlay handler: once proceed is enabled,
                // divination is over and the only legal action is to continue.
                options = new List<Dictionary<string, object>>
                {
                    new()
                    {
                        ["index"] = 0,
                        ["action"] = NonCombatBridgeProtocol.ProceedAction,
                        ["enabled"] = true,
                    },
                };
            }
            else
            {
                options = hiddenCells
                    .Select((cell, index) => CellOption(cell, index))
                    .ToList();
            }

            if (options.Count == 0)
            {
                return CrystalSphereChoice.Wait();
            }

            RunState runState = RunManager.Instance.DebugOnlyGetState();
            Dictionary<string, object> minigame = SerializeMinigame(screen);
            string stateJson = JsonSerializer.Serialize(new Dictionary<string, object>
            {
                ["type"] = NonCombatBridgeProtocol.CrystalSphereState,
                ["options"] = options,
                ["minigame"] = minigame,
                ["crystal_cells"] = minigame["cells"],
                ["floor"] = runState.TotalFloor,
                ["act"] = runState.CurrentActIndex + ActDisplayIndexOffset,
            });
            string? responseJson = await BridgeServer.Instance.SendStateAndWaitForActionAsync(
                stateJson,
                AgentTimeout,
                ct);
            if (responseJson == null)
            {
                return hiddenCells.Count > 0
                    ? CrystalSphereChoice.Click(random.NextItem(hiddenCells))
                    : CrystalSphereChoice.Wait();
            }

            int chosenIndex = ReadChoiceIndex(responseJson);
            if (canProceed && chosenIndex == 0)
            {
                return CrystalSphereChoice.Proceed();
            }
            if (!canProceed && chosenIndex >= 0 && chosenIndex < hiddenCells.Count)
            {
                return CrystalSphereChoice.Click(hiddenCells[chosenIndex]);
            }
        }
        catch (Exception ex)
        {
            AutoSlayLog.Warn("[RlCrystalSphere] Agent error: " + ex.Message);
        }

        if (canProceed)
        {
            return CrystalSphereChoice.Proceed();
        }
        return hiddenCells.Count > 0
            ? CrystalSphereChoice.Click(random.NextItem(hiddenCells))
            : CrystalSphereChoice.Wait();
    }

    private static Dictionary<string, object> CellOption(NCrystalSphereCell cell, int index)
    {
        return new Dictionary<string, object>
        {
            ["index"] = index,
            ["id"] = $"divine:{cell.Entity.X}:{cell.Entity.Y}",
            ["entity_id"] = $"crystal-cell:{cell.Entity.X}:{cell.Entity.Y}",
            ["action"] = NonCombatBridgeProtocol.DivineCellAction,
            ["x"] = cell.Entity.X,
            ["y"] = cell.Entity.Y,
            ["affected_cell_ids"] = AdjacentCoordinates(cell.Entity.X, cell.Entity.Y)
                .Select(coord => $"crystal-cell:{coord.X}:{coord.Y}")
                .ToList(),
            ["enabled"] = true,
        };
    }

    private static Dictionary<string, object> SerializeMinigame(NCrystalSphereScreen screen)
    {
        FieldInfo? field = typeof(NCrystalSphereScreen).GetField(
            "_entity",
            BindingFlags.Instance | BindingFlags.NonPublic);
        CrystalSphereMinigame? game = field?.GetValue(screen) as CrystalSphereMinigame;
        IReadOnlyList<CrystalSphereItem> revealed = ReadRevealedItems(game);
        var revealedIds = revealed.ToDictionary(item => item, ItemId);
        var revealedItems = new List<Dictionary<string, object>>();
        foreach (CrystalSphereItem item in revealed)
        {
            SerializableCrystalSphereItem serialized = item.ToSerializable();
            var itemState = new Dictionary<string, object>
            {
                ["entity_id"] = revealedIds[item],
                ["item_type"] = serialized.type.ToString(),
                ["width"] = item.Size.X,
                ["height"] = item.Size.Y,
                ["revealed"] = true,
                ["is_good"] = item.IsGood,
            };
            if (serialized.type == CrystalSphereItemType.CardReward)
                itemState["rarity"] = serialized.cardRarity.ToString();
            else if (serialized.type == CrystalSphereItemType.Potion)
                itemState["rarity"] = serialized.potionRarity.ToString();
            else if (serialized.type == CrystalSphereItemType.Gold)
                itemState["amount"] = serialized.isBigGold ? 30 : 10;
            revealedItems.Add(itemState);
        }
        var cells = new List<Dictionary<string, object>>();
        foreach (NCrystalSphereCell node in AllCells(screen))
        {
            CrystalSphereCell cell = node.Entity;
            string? revealedItemId = cell.Item != null
                && revealedIds.TryGetValue(cell.Item, out string? itemId)
                    ? itemId
                    : null;
            cells.Add(new Dictionary<string, object>
            {
                ["entity_id"] = $"crystal-cell:{cell.X}:{cell.Y}",
                ["x"] = cell.X,
                ["y"] = cell.Y,
                ["hidden"] = cell.IsHidden,
                ["clickable"] = cell.IsHidden && game?.IsFinished != true,
                ["revealed_item_id"] = revealedItemId!,
            });
        }
        return new Dictionary<string, object>
        {
            ["type"] = NonCombatBridgeProtocol.CrystalSphereState,
            ["grid_width"] = game?.GridSize.X ?? 11,
            ["grid_height"] = game?.GridSize.Y ?? 11,
            ["divinations_remaining"] = game?.DivinationCount ?? 0,
            ["tool"] = game?.CrystalSphereTool.ToString() ?? "Unknown",
            ["finished"] = game?.IsFinished ?? false,
            ["placed_all_items"] = game?.PlacedAllItems ?? false,
            ["cells"] = cells,
            ["revealed_items"] = revealedItems,
        };
    }

    private static IReadOnlyList<CrystalSphereItem> ReadRevealedItems(
        CrystalSphereMinigame? game)
    {
        if (game == null)
            return Array.Empty<CrystalSphereItem>();
        FieldInfo? field = typeof(CrystalSphereMinigame).GetField(
            "_revealed",
            BindingFlags.Instance | BindingFlags.NonPublic);
        return field?.GetValue(game) as IReadOnlyList<CrystalSphereItem>
            ?? Array.Empty<CrystalSphereItem>();
    }

    private static string ItemId(CrystalSphereItem item)
    {
        SerializableCrystalSphereItem serialized = item.ToSerializable();
        return $"crystal-item:{item.Position.X}:{item.Position.Y}:{serialized.type}";
    }

    private static List<(int X, int Y)> AdjacentCoordinates(int x, int y)
    {
        var result = new List<(int X, int Y)>();
        foreach (int dx in new[] { -1, 1 })
        {
            int nx = x + dx;
            if (nx >= 0 && nx < 11)
                result.Add((nx, y));
        }
        foreach (int dy in new[] { -1, 1 })
        {
            int ny = y + dy;
            if (ny >= 0 && ny < 11)
                result.Add((x, ny));
        }
        foreach (int dx in new[] { -1, 1 })
        {
            foreach (int dy in new[] { -1, 1 })
            {
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < 11 && ny >= 0 && ny < 11)
                    result.Add((nx, ny));
            }
        }
        result.Add((x, y));
        return result;
    }

    private static int ReadChoiceIndex(string responseJson)
    {
        using JsonDocument doc = JsonDocument.Parse(responseJson);
        JsonElement root = doc.RootElement;
        string action = root.GetProperty("action").GetString() ?? "";
        if (action == NonCombatBridgeProtocol.ChooseAction &&
            root.TryGetProperty("index", out JsonElement indexProp))
        {
            return indexProp.GetInt32();
        }
        return -1;
    }

    private static async Task WaitForOutcome(NCrystalSphereScreen screen, CancellationToken ct)
    {
        await WaitHelper.Until(delegate
        {
            if (!GodotObject.IsInstanceValid(screen) || !screen.IsVisibleInTree())
            {
                return true;
            }
            NProceedButton? proceedButton = screen.GetNodeOrNull<NProceedButton>("%ProceedButton");
            if (proceedButton?.IsEnabled ?? false)
            {
                return true;
            }
            IOverlayScreen overlayScreen = NOverlayStack.Instance?.Peek();
            return overlayScreen != null && overlayScreen != screen;
        }, ct, TimeSpan.FromSeconds(WaitForOutcomeTimeoutSeconds), "Crystal Sphere did not produce proceed or reward screen");
    }

    private static async Task ClickProceed(NCrystalSphereScreen screen, NProceedButton proceedButton, CancellationToken ct)
    {
        AutoSlayLog.Action("Clicking Crystal Sphere proceed button");
        await UiHelper.Click(proceedButton);
        await WaitHelper.Until(
            () => !GodotObject.IsInstanceValid(screen)
                || !screen.IsVisibleInTree()
                || (NMapScreen.Instance?.IsVisibleInTree() ?? false),
            ct,
            TimeSpan.FromSeconds(ProceedCloseTimeoutSeconds),
            "Crystal Sphere screen did not close after clicking proceed");
        if (GodotObject.IsInstanceValid(screen) && screen.IsVisibleInTree())
        {
            NMapScreen? mapScreen = NMapScreen.Instance;
            if (mapScreen != null && mapScreen.IsVisibleInTree())
            {
                AutoSlayLog.Info("Map opened, manually removing Crystal Sphere screen from overlay stack");
                NOverlayStack.Instance?.Remove(screen);
                await Task.Delay(OverlayRemovalDelayMs, ct);
            }
        }
    }

    private readonly record struct CrystalSphereChoice(bool ShouldProceed, NCrystalSphereCell? Cell)
    {
        public static CrystalSphereChoice Proceed() => new(true, null);

        public static CrystalSphereChoice Click(NCrystalSphereCell cell) => new(false, cell);

        public static CrystalSphereChoice Wait() => new(false, null);
    }
}
