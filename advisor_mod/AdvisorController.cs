using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Godot;

namespace STS2AdvisorMod;

internal partial class AdvisorController : Node
{
    private const int Port = 9003;
    private const double PollIntervalSeconds = 0.20;
    private AdvisorOverlay? _overlay;
    private double _elapsed;
    private bool _f8WasDown;
    private bool _wasConnected;
    private string? _activeFingerprint;
    private string? _activeRequestId;

    public override void _Ready()
    {
        ProcessMode = ProcessModeEnum.Always;
        _overlay = new AdvisorOverlay { Name = "STS2AdvisorOverlay" };
        AddChild(_overlay);
        AdvisorServer.Instance.Start(Port);
        AdvisorLog.Info("Ready. F8 toggles the advice panel.");
    }

    public override void _ExitTree() => AdvisorServer.Instance.Stop();

    public override void _Process(double delta)
    {
        bool f8Down = Input.IsKeyPressed(Key.F8);
        if (f8Down && !_f8WasDown) _overlay?.TogglePanel();
        _f8WasDown = f8Down;

        bool connected = AdvisorServer.Instance.IsClientConnected;
        if (connected != _wasConnected)
        {
            _overlay?.SetConnection(connected);
            _wasConnected = connected;
            _activeFingerprint = null;
            _activeRequestId = null;
            if (!connected)
            {
                _overlay?.SetIdle();
            }
        }

        DrainAdviceMessages();
        _elapsed += delta;
        if (_elapsed < PollIntervalSeconds) return;
        _elapsed = 0;
        PollDecision(connected);
    }

    private void PollDecision(bool connected)
    {
        if (!AdvisorStateCollector.TryCollect(out Dictionary<string, object?> state))
        {
            if (_activeFingerprint != null) _overlay?.SetIdle();
            _activeFingerprint = null;
            _activeRequestId = null;
            return;
        }

        string stateJson = JsonSerializer.Serialize(state);
        string fingerprint = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(stateJson)));
        if (fingerprint == _activeFingerprint) return;

        _activeFingerprint = fingerprint;
        _activeRequestId = Guid.NewGuid().ToString("N");
        string decisionType = state.TryGetValue("type", out object? value) ? value?.ToString() ?? "" : "";
        _overlay?.SetWaiting(decisionType);
        if (!connected) return;

        state["request_id"] = _activeRequestId;
        state["fingerprint"] = fingerprint;
        AdvisorServer.Instance.Send(JsonSerializer.Serialize(state));
    }

    private void DrainAdviceMessages()
    {
        while (AdvisorServer.Instance.TryDequeue(out string message))
        {
            try
            {
                using JsonDocument doc = JsonDocument.Parse(message);
                JsonElement root = doc.RootElement;
                string? requestId = root.TryGetProperty("request_id", out JsonElement request)
                    ? request.GetString() : null;
                string? fingerprint = root.TryGetProperty("fingerprint", out JsonElement fp)
                    ? fp.GetString() : null;
                if (requestId != _activeRequestId || fingerprint != _activeFingerprint)
                {
                    AdvisorLog.Info("Discarded stale advice response.");
                    continue;
                }

                string decisionType = root.TryGetProperty("decision_type", out JsonElement type)
                    ? type.GetString() ?? "" : "";
                string summary = root.TryGetProperty("summary", out JsonElement text)
                    ? text.GetString() ?? "无建议" : "无建议";
                string details = root.TryGetProperty("details", out JsonElement detail)
                    ? detail.GetString() ?? "" : "";
                _overlay?.SetAdvice(decisionType, summary, details);
            }
            catch (Exception ex)
            {
                AdvisorLog.Info($"Invalid advice response: {ex.Message}");
            }
        }
    }
}
