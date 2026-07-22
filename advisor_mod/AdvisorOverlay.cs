using Godot;

namespace STS2AdvisorMod;

internal partial class AdvisorOverlay : CanvasLayer
{
    private PanelContainer? _panel;
    private Label? _status;
    private Label? _advice;

    public override void _Ready()
    {
        Layer = 100;
        _panel = new PanelContainer
        {
            Name = "AdvisorPanel",
            MouseFilter = Control.MouseFilterEnum.Ignore,
            CustomMinimumSize = new Vector2(390, 0),
        };
        _panel.SetAnchorsPreset(Control.LayoutPreset.TopRight);
        _panel.OffsetLeft = -410;
        _panel.OffsetTop = 18;
        _panel.OffsetRight = -18;

        var margin = new MarginContainer { MouseFilter = Control.MouseFilterEnum.Ignore };
        margin.AddThemeConstantOverride("margin_left", 14);
        margin.AddThemeConstantOverride("margin_top", 10);
        margin.AddThemeConstantOverride("margin_right", 14);
        margin.AddThemeConstantOverride("margin_bottom", 10);

        var rows = new VBoxContainer { MouseFilter = Control.MouseFilterEnum.Ignore };
        var title = new Label { Text = "RL ADVISOR  [F8 显示/隐藏]" };
        title.AddThemeFontSizeOverride("font_size", 18);
        _status = new Label { Text = "等待 Python advisor runner…" };
        _status.Modulate = new Color(0.72f, 0.78f, 0.86f);
        _advice = new Label
        {
            Text = "所有操作仍由玩家完成。",
            AutowrapMode = TextServer.AutowrapMode.WordSmart,
            CustomMinimumSize = new Vector2(360, 0),
        };
        _advice.AddThemeFontSizeOverride("font_size", 17);

        rows.AddChild(title);
        rows.AddChild(_status);
        rows.AddChild(_advice);
        margin.AddChild(rows);
        _panel.AddChild(margin);
        AddChild(_panel);
    }

    public void SetConnection(bool connected)
    {
        if (_status != null)
            _status.Text = connected ? "模型已连接" : "等待 Python advisor runner…";
    }

    public void SetWaiting(string decisionType)
    {
        if (_advice != null)
            _advice.Text = $"{DecisionTitle(decisionType)}\n模型计算中…";
    }

    public void SetAdvice(string decisionType, string summary, string details)
    {
        if (_advice == null) return;
        string suffix = string.IsNullOrWhiteSpace(details) ? "" : $"\n{details}";
        _advice.Text = $"{DecisionTitle(decisionType)}\n建议：{summary}{suffix}";
    }

    public void SetIdle()
    {
        if (_advice != null)
            _advice.Text = "当前没有可建议的决策。\n所有操作仍由玩家完成。";
    }

    public void TogglePanel()
    {
        if (_panel != null) _panel.Visible = !_panel.Visible;
    }

    private static string DecisionTitle(string type) => type switch
    {
        "combat_action" => "战斗决策",
        "map_select" => "路线决策",
        "card_reward" => "选牌决策",
        _ => "决策建议",
    };
}

