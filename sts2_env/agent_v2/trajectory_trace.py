"""Compact, inspectable records of complete simulator runs."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

import numpy as np

from sts2_env.agent_v2.tensorizer import _candidate_slots

TRACE_FORMAT = "sts2-trajectory-v1"


def _name(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "name", value))


def _run_part(snapshot: dict[str, Any]) -> dict[str, Any]:
    run = snapshot.get("run_state")
    return run if isinstance(run, dict) else snapshot


def _compact_card(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": card.get("card_id") or card.get("id"),
        "cost": card.get("cost"),
        "index": card.get("zone_index", card.get("index")),
        "playable": card.get("playable"),
        "enchantments": card.get("enchantments") or {},
        "afflictions": card.get("afflictions") or {},
    }


def compact_state(snapshot: dict[str, Any], manager: Any) -> dict[str, Any]:
    """Keep decision-relevant, player-visible fields without the whole map."""
    run = _run_part(snapshot)
    combat_players = snapshot.get("players") or []
    run_players = run.get("players") or []
    player = combat_players[0] if combat_players else (run_players[0] if run_players else {})
    persistent_player = run_players[0] if run_players else player
    cards = snapshot.get("cards") or []
    deck = [
        _compact_card(card)
        for card in run.get("cards", []) or []
        if card.get("zone") == "deck"
    ]
    room = getattr(manager, "current_room", None)
    room_type = getattr(room, "room_type", None) or getattr(manager, "_current_room_type", None)
    event = getattr(manager, "_event_model", None)
    choice_key = next((key for key in ("options", "bundles", "cards", "nodes") if key in snapshot), None)
    choices = [
        {
            "id": item.get("id") or item.get("option_id") or item.get("type") or item.get("action"),
            "label": item.get("label"),
            "action": item.get("action"),
            "cost": item.get("cost"),
            "cards": [card.get("id") for card in item.get("cards", [])],
        }
        for item in (snapshot.get(choice_key, []) if choice_key else [])
        if isinstance(item, dict)
    ]
    if manager.phase == "EVENT":
        choices = [
            {
                "id": item.get("option_id") or item.get("card_id"),
                "label": item.get("label"),
                "action": item.get("action"),
                "cost": None,
                "cards": [],
            }
            for item in manager.get_available_actions()
            if item.get("action") in {"event_choice", "choose", "confirm_choice"}
        ]
    treasure_relic = None
    if _name(snapshot.get("phase")) == "TREASURE":
        collect = next(
            (item for item in manager.get_available_actions() if item.get("action") == "collect"),
            None,
        )
        treasure_relic = collect.get("relic_id") if collect else None
    enemies = snapshot.get("creatures") or snapshot.get("enemies") or []
    return {
        "phase": _name(snapshot.get("phase") or snapshot.get("global", {}).get("phase")),
        "floor": run.get("floor", 0),
        "act": run.get("act", 1),
        "act_id": run.get("act_id"),
        "room_type": _name(room_type),
        "event_id": getattr(event, "event_id", None) if manager.phase == "EVENT" else None,
        "treasure_relic": treasure_relic,
        "choices": choices,
        "round": snapshot.get("round"),
        "hp": player.get("hp", persistent_player.get("hp")),
        "max_hp": player.get("max_hp", persistent_player.get("max_hp")),
        "block": player.get("block", 0),
        "energy": player.get("energy"),
        "gold": persistent_player.get("gold"),
        "hand": [_compact_card(card) for card in cards if card.get("zone") == "hand"],
        "deck": deck,
        "enemies": [{
            "id": enemy.get("monster_id") or enemy.get("id"),
            "hp": enemy.get("hp"),
            "max_hp": enemy.get("max_hp"),
            "block": enemy.get("block"),
            "alive": enemy.get("alive", enemy.get("is_alive")),
            "intents": enemy.get("intents") or [],
        } for enemy in enemies],
        "powers": [{
            "id": power.get("power_id") or power.get("id"),
            "amount": power.get("amount"),
            "display_amount": power.get("display_amount"),
            "owner_id": power.get("owner_id"),
        } for power in snapshot.get("powers", []) or []],
        "relics": [relic.get("relic_id") or relic.get("id") for relic in run.get("relics", []) or []],
        "potions": [
            potion.get("potion_id") or potion.get("id")
            for potion in run.get("potions", []) or []
        ],
    }


def describe_action(
    snapshot: dict[str, Any], action_mask: np.ndarray, slot: int,
    *, legacy_v5: bool = False,
) -> dict[str, Any]:
    """Resolve a sampled action slot to its semantic candidate and entities."""
    candidate = _candidate_slots(snapshot, action_mask).get(slot)
    result: dict[str, Any] = {"slot": slot}
    if legacy_v5 and _candidate_slots(snapshot, action_mask, legacy_v5=True).get(slot) is None:
        result["legacy_model_candidate_missing"] = True
    if candidate is None:
        result["action_type"] = "UNRESOLVED"
        result["candidate_missing"] = True
        result["snapshot_type"] = snapshot.get("type")
        result["candidates"] = [
            {
                "action_type": item.get("action_type"),
                "payload": item.get("payload"),
            }
            for item in snapshot.get("candidates", []) or []
        ]
        return result
    lookup: dict[str, dict[str, Any]] = {}
    for container in (snapshot, _run_part(snapshot)):
        for key in (
            "players", "creatures", "enemies", "cards", "hand", "powers",
            "relics", "potions", "map_nodes", "options", "nodes", "bundles",
            "crystal_cells",
        ):
            for entity in container.get(key, []) or []:
                if isinstance(entity, dict) and entity.get("entity_id") is not None:
                    lookup[str(entity["entity_id"])] = entity

    def content(entity_id: Any) -> str | None:
        entity = lookup.get(str(entity_id), {})
        for key in (
            "card_id", "monster_id", "power_id", "relic_id", "potion_id",
            "node_type", "id", "type",
        ):
            if entity.get(key) is not None:
                return str(entity[key])
        return None

    payload = candidate.get("payload") or {}
    semantic_action = str(payload.get("action", "")).upper()
    result.update({
        "candidate_id": candidate.get("candidate_id"),
        "action_type": semantic_action if snapshot.get("reward_item_type") else candidate.get("action_type"),
        "candidate_action_type": candidate.get("action_type"),
        "source_id": candidate.get("source_id"),
        "source": content(candidate.get("source_id")) or (candidate.get("features") or {}).get("model_source_content"),
        "target_id": candidate.get("target_id"),
        "target": content(candidate.get("target_id")),
        "payload": payload,
    })
    return result


def record_step(
    *, index: int, before: dict[str, Any], action: dict[str, Any],
    after_snapshot: dict[str, Any],
    after_manager: Any, reward: float, info: dict[str, Any],
    terminated: bool, truncated: bool,
) -> dict[str, Any]:
    after = compact_state(after_snapshot, after_manager)
    return {
        "step": index,
        "floor": after["floor"] if after["floor"] != before["floor"] else before["floor"],
        "before": before,
        "action": action,
        "after": after,
        "reward": reward,
        "base_reward": info.get("base_reward"),
        "reward_components": info.get("reward_components") or {},
        "terminated": terminated,
        "truncated": truncated,
    }


def write_trace(trace: dict[str, Any], output_stem: Path) -> tuple[Path, Path]:
    """Write machine-readable JSON and an offline Chinese/English viewer."""
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_stem.with_suffix(".json")
    html_path = output_stem.with_suffix(".html")
    serialized = json.dumps(trace, ensure_ascii=False, indent=2)
    json_path.write_text(serialized + "\n", encoding="utf-8")
    safe_json = serialized.replace("<", "\\u003c").replace("&", "\\u0026")
    html_path.write_text(
        _HTML_TEMPLATE.replace("__TRACE_TITLE__", escape(f"STS2 · seed {trace['seed']}"))
        .replace("__TRACE_JSON__", safe_json),
        encoding="utf-8",
    )
    return json_path, html_path


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TRACE_TITLE__</title>
<style>
:root { color-scheme: dark; font: 15px/1.55 system-ui, sans-serif; background: #111827; color: #e5e7eb; }
body { max-width: 1100px; margin: auto; padding: 24px; }
header { display: flex; justify-content: space-between; gap: 16px; align-items: start; }
h1 { font-size: 1.55rem; margin: 0 0 4px; } h2 { font-size: 1.15rem; }
.muted { color: #9ca3af; } .summary { margin: 16px 0 24px; }
select, input { background: #1f2937; color: #f9fafb; border: 1px solid #4b5563; border-radius: 6px; padding: 6px 9px; }
.controls { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
details { border: 1px solid #374151; border-radius: 9px; margin: 9px 0; background: #1f2937; }
details.floor { background: #182331; } summary { cursor: pointer; padding: 11px 14px; }
.steps { padding: 4px 13px 14px; } .step { margin: 7px 0; }
.step summary { display: flex; gap: 10px; flex-wrap: wrap; align-items: baseline; }
.tag { color: #93c5fd; font-size: .85rem; } .danger { color: #fca5a5; }
.state { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 12px; padding: 0 13px 13px; }
.panel { background: #111827; border-radius: 7px; padding: 10px; overflow-wrap: anywhere; }
.panel h3 { font-size: .95rem; margin: 0 0 5px; } .panel p { margin: 5px 0; }
code { color: #fcd34d; } button { cursor: pointer; }
</style>
</head>
<body>
<header><div><h1 id="title"></h1><div class="muted" id="subtitle"></div></div>
<div class="controls"><label for="lang" id="language-label"></label><select id="lang"><option value="zh">中文</option><option value="en">English</option></select>
<label for="filter" id="filter-label"></label><input id="filter" type="search"></div></header>
<div class="summary" id="summary"></div><main id="floors"></main>
<script type="application/json" id="trace-data">__TRACE_JSON__</script>
<script>
const trace = JSON.parse(document.getElementById('trace-data').textContent);
const words = {
  zh: {title:'测试轨迹', language:'语言', filter:'筛选', filterHint:'阶段、动作、怪物、卡牌',
    seed:'种子', model:'模型', result:'结果', won:'胜利', lost:'失败', floor:'第 {n} 层',
    steps:'步', room:'房间', round:'回合', hp:'生命', gold:'金币', energy:'能量', block:'格挡',
    before:'行动前', after:'行动后', hand:'手牌', enemies:'敌人及意图', powers:'能力',
    deck:'牌组', relics:'遗物', potions:'药水', reward:'奖励', target:'目标',
    event:'事件', choices:'可选奖励／选项', treasureRelic:'宝箱遗物', gainedRelic:'获得遗物',
    empty:'无', noMatch:'没有匹配的动作', candidateMissing:'缺少语义候选', legacyMissing:'旧 v5 模型未见语义候选'},
  en: {title:'Test trajectory', language:'Language', filter:'Filter', filterHint:'Phase, action, enemy, card',
    seed:'Seed', model:'Model', result:'Result', won:'Win', lost:'Loss', floor:'Floor {n}',
    steps:'steps', room:'Room', round:'Turn', hp:'HP', gold:'Gold', energy:'Energy', block:'Block',
    before:'Before', after:'After', hand:'Hand', enemies:'Enemies and intents', powers:'Powers',
    deck:'Deck', relics:'Relics', potions:'Potions', reward:'Reward', target:'Target',
    event:'Event', choices:'Reward choices / options', treasureRelic:'Chest relic', gainedRelic:'Relic gained',
    empty:'None', noMatch:'No matching actions', candidateMissing:'Semantic candidate missing', legacyMissing:'Legacy v5 model saw no semantic candidate'}
};
const names = {zh:{COMBAT:'战斗',MAP_CHOICE:'选路',CARD_REWARD:'卡牌奖励',REST_SITE:'火堆',SHOP:'商店',EVENT:'事件',TREASURE:'宝箱',BOSS_RELIC:'Boss 遗物',PLAY_CARD:'打牌',END_TURN:'结束回合',USE_POTION:'使用药水',PICK_POTION:'领取药水',SKIP_POTION:'跳过药水',PICK_RELIC_REWARD:'领取遗物',SKIP_RELIC:'跳过遗物',MOVE:'移动',PICK_CARD:'选牌',SELECT_CARD:'选择卡牌',REST_OPTION:'选择火堆选项',EVENT_CHOICE:'选择事件选项',COLLECT:'领取',SKIP:'跳过'},
  en:{COMBAT:'Combat',MAP_CHOICE:'Map',CARD_REWARD:'Card reward',REST_SITE:'Rest site',SHOP:'Shop',EVENT:'Event',TREASURE:'Treasure',BOSS_RELIC:'Boss relic',PLAY_CARD:'Play card',END_TURN:'End turn',USE_POTION:'Use potion',PICK_POTION:'Collect potion',SKIP_POTION:'Skip potion',PICK_RELIC_REWARD:'Collect relic',SKIP_RELIC:'Skip relic',MOVE:'Move',PICK_CARD:'Pick card',SELECT_CARD:'Select card',REST_OPTION:'Rest option',EVENT_CHOICE:'Event choice',COLLECT:'Collect',SKIP:'Skip'}};
const el = (tag, text, cls) => {const x=document.createElement(tag); if(text !== undefined) x.textContent=String(text); if(cls) x.className=cls; return x;};
const fmt = x => x === null || x === undefined ? '—' : String(x);
const list = (xs, fn) => xs && xs.length ? xs.map(fn).join(' · ') : '—';
const term = (x, lang) => names[lang][x] || x || '—';
function statePanel(state, lang, title) {
  const w=words[lang], box=el('div',undefined,'panel'); box.append(el('h3',title));
  const lines=[
    `${w.hp}: ${fmt(state.hp)}/${fmt(state.max_hp)}  ${w.block}: ${fmt(state.block)}  ${w.energy}: ${fmt(state.energy)}  ${w.gold}: ${fmt(state.gold)}`,
    ...(state.event_id ? [`${w.event}: ${state.event_id}`] : []),
    ...(state.treasure_relic ? [`${w.treasureRelic}: ${state.treasure_relic}`] : []),
    `${w.hand}: ${list(state.hand,c=>`${c.id}[${fmt(c.cost)}]${c.playable===false?' ×':''}${Object.keys(c.afflictions||{}).length?' !'+Object.keys(c.afflictions).join(','):''}${Object.keys(c.enchantments||{}).length?' +'+Object.entries(c.enchantments).map(([k,v])=>k+':'+v).join(','):''}`)}`,
    `${w.powers}: ${list(state.powers,p=>`${p.id}(${fmt(p.amount)})`)}`,
    `${w.potions}: ${list(state.potions,p=>p)}`,
    `${w.relics}: ${list(state.relics,r=>r)}`,
    `${w.deck}: ${list(state.deck,c=>c.id)}`
  ];
  for(const line of lines) box.append(el('p',line));
  if(state.enemies && state.enemies.length) {
    box.append(el('p',`${w.enemies}:`));
    for(const e of state.enemies) box.append(el('div',`${e.id} ${fmt(e.hp)}/${fmt(e.max_hp)} ${list(e.intents,i=>`${i.intent_type} ${fmt(i.damage)}×${fmt(i.hits)}`)}`));
  }
  if(state.choices && state.choices.length) {
    box.append(el('p',`${w.choices}:`));
    for(const choice of state.choices) {
      const content=(choice.cards && choice.cards.length ? choice.cards.join(' + ') : choice.id) || choice.label || choice.action || '—';
      box.append(el('div',`${content}${choice.cost !== null && choice.cost !== undefined ? ` [${choice.cost}]` : ''}${choice.label && choice.label !== content ? ` · ${choice.label}` : ''}`));
    }
  }
  return box;
}
function render() {
  const lang=document.getElementById('lang').value, w=words[lang];
  const query=document.getElementById('filter').value.trim().toLowerCase();
  document.documentElement.lang=lang;
  document.getElementById('title').textContent=`${w.title} · ${w.seed} ${trace.seed}`;
  document.getElementById('language-label').textContent=w.language;
  document.getElementById('filter-label').textContent=w.filter;
  document.getElementById('filter').placeholder=w.filterHint;
  document.getElementById('subtitle').textContent=`${w.model}: ${trace.model_path || 'random'}`;
  document.getElementById('summary').textContent=`${w.result}: ${trace.summary.won?w.won:w.lost} · ${w.floor.replace('{n}',trace.summary.floor)} · ${trace.summary.steps} ${w.steps}`;
  const root=document.getElementById('floors'); root.replaceChildren();
  const groups=new Map(); for(const step of trace.steps) {const n=step.floor; if(!groups.has(n)) groups.set(n,[]); groups.get(n).push(step);}
  for(const [floor, steps] of groups) {
    const visible=steps.filter(s=>JSON.stringify([s.before.phase,s.before.room_type,s.before.event_id,s.before.choices,s.action,s.before.enemies,s.before.hand]).toLowerCase().includes(query));
    if(!visible.length) continue;
    const section=el('details',undefined,'floor'); section.open=floor===Math.max(...groups.keys());
    const first=steps[0].before, last=steps[steps.length-1].after;
    const room=steps.map(s=>s.before.phase==='MAP_CHOICE'?s.after.room_type:s.before.room_type).find(Boolean) || first.room_type;
    const encountered=[...new Set(steps.flatMap(s=>(s.before.enemies||[]).map(e=>e.id)).filter(Boolean))];
    const event=steps.map(s=>s.before.event_id).find(Boolean);
    const gainedRelics=(last.relics||[]).filter(r=>!(first.relics||[]).includes(r));
    section.append(el('summary',`${w.floor.replace('{n}',floor)}${first.act_id?' · '+first.act_id:''} · ${term(room,lang)}${event?' · '+event:''}${encountered.length?' · '+encountered.join(', '):''} · ${w.hp} ${fmt(first.hp)} → ${fmt(last.hp)} · ${w.gold} ${fmt(first.gold)} → ${fmt(last.gold)}${gainedRelics.length?' · '+w.gainedRelic+' '+gainedRelics.join(', '):''} · ${visible.length} ${w.steps}`));
    const body=el('div',undefined,'steps');
    for(const s of visible) {
      const d=el('details',undefined,'step'), a=s.action;
      const label=[`#${s.step}`,s.before.phase=== 'COMBAT' ? `${w.round} ${fmt(s.before.round)}` : term(s.before.phase,lang),term(a.action_type,lang),a.source||a.source_id,a.target?`→ ${a.target}`:null,a.candidate_missing?`⚠ ${w.candidateMissing}`:null,a.legacy_model_candidate_missing?`⚠ ${w.legacyMissing}`:null].filter(Boolean).join(' · ');
      d.append(el('summary',label));
      const states=el('div',undefined,'state'); states.append(statePanel(s.before,lang,w.before),statePanel(s.after,lang,w.after));
      const rewards=el('div',undefined,'panel'); rewards.append(el('h3',w.reward));
      rewards.append(el('p',`${fmt(s.reward)} · ${JSON.stringify(s.reward_components)}`));
      const goldDelta=(s.after.gold ?? 0)-(s.before.gold ?? 0);
      if(goldDelta) rewards.append(el('p',`${w.gold}: ${goldDelta>0?'+':''}${goldDelta}`));
      const gained=(s.after.relics||[]).filter(r=>!(s.before.relics||[]).includes(r));
      if(gained.length) rewards.append(el('p',`${w.gainedRelic}: ${gained.join(', ')}`));
      states.append(rewards);
      d.append(states); body.append(d);
    }
    section.append(body); root.append(section);
  }
  if(!root.children.length) root.append(el('p',w.noMatch,'muted'));
}
document.getElementById('lang').addEventListener('change',render);
document.getElementById('filter').addEventListener('input',render);
render();
</script>
</body></html>
"""
