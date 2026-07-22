using Godot;
using MegaCrit.Sts2.Core.Modding;

namespace STS2AdvisorMod;

[ModInitializer(nameof(Initialize))]
public partial class AdvisorMainFile : Node
{
    public const string ModId = "STS2AdvisorMod";

    public static void Initialize()
    {
        AdvisorLog.Info("Initializing passive advisor (no automatic inputs).");
        SceneTree tree = (SceneTree)Engine.GetMainLoop();
        var controller = new AdvisorController { Name = "STS2AdvisorController" };
        tree.Root.CallDeferred(Node.MethodName.AddChild, controller);
    }
}

internal static class AdvisorLog
{
    public static void Info(string message) => GD.Print($"[STS2Advisor] {message}");
}

