// BridgeRuntimeConfig.cs -- machine-local runtime switches for the Bridge.

using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.Json;

namespace STS2BridgeMod;

public sealed class BridgeRuntimeConfig
{
    public const string FileName = "STS2BridgeMod.runtime.json";

    public bool ResumeExistingRun { get; init; }

    public static BridgeRuntimeConfig Load()
    {
        string? assemblyDirectory = Path.GetDirectoryName(
            typeof(BridgeRuntimeConfig).Assembly.Location);
        string[] candidates = new[]
        {
            string.IsNullOrWhiteSpace(assemblyDirectory)
                ? null
                : Path.Combine(assemblyDirectory, FileName),
            Path.Combine(
                AppContext.BaseDirectory,
                "mods",
                MainFile.ModId,
                FileName),
            Path.Combine(
                Directory.GetCurrentDirectory(),
                "mods",
                MainFile.ModId,
                FileName),
        }
            .Where(path => !string.IsNullOrWhiteSpace(path))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .Cast<string>()
            .ToArray();
        string? path = candidates.FirstOrDefault(File.Exists);
        if (path == null)
        {
            Logger.Log(
                $"[BridgeConfig] {FileName} not found in "
                + $"{string.Join(", ", candidates)}; starting a new run.");
            return new BridgeRuntimeConfig();
        }

        try
        {
            using JsonDocument document = JsonDocument.Parse(
                File.ReadAllText(path));
            JsonElement root = document.RootElement;
            bool resume = root.TryGetProperty(
                    "resume_existing_run",
                    out JsonElement value)
                && value.ValueKind is JsonValueKind.True;
            Logger.Log(
                $"[BridgeConfig] Loaded {path}; resume_existing_run={resume}.");
            return new BridgeRuntimeConfig { ResumeExistingRun = resume };
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException(
                $"Could not read Bridge runtime config '{path}'.", ex);
        }
    }
}
