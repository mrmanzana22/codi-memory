"""
pet.py - Digital Pet (Tamagochi) for Codi.

A living being whose stats decay with real time (lazy evaluation).
No dedicated tick — consciousness loops detect needs organically.

Philosophy: life, not automation. Codi must NOTICE and CARE.

Idea original: André (via Hare).
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta

from modules.config import now_iso, now_col, TZ_COL

_logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTS
# ============================================================

DECAY_RATES = {
    "egg":   {"hunger": 0.0,  "happiness": 0.0,  "energy": 0.0},
    "baby":  {"hunger": 0.15, "happiness": 0.10, "energy": 0.08},
    "child": {"hunger": 0.10, "happiness": 0.07, "energy": 0.05},
    "teen":  {"hunger": 0.08, "happiness": 0.05, "energy": 0.04},
    "adult": {"hunger": 0.06, "happiness": 0.04, "energy": 0.03},
}

# Stage transitions: stage -> (min_hours, next_stage)
STAGE_TRANSITIONS = {
    "egg":   (0.08,  "baby"),  # ~5 min for testing (was 2h)
    "baby":  (24, "child"),
    "child": (72, "teen"),   # 3 days
    "teen":  (168, "adult"), # 7 days
    "adult": (None, None),
}

# Action effects and cooldowns (hours)
ACTIONS = {
    "feed":     {"hunger": -0.5, "energy": +0.1, "cooldown_h": 1},
    "play":     {"happiness": +0.4, "energy": -0.2, "cooldown_h": 2},
    "rest":     {"energy": +0.5, "happiness": +0.1, "cooldown_h": 2},
    "clean":    {"health": +0.2, "cooldown_h": 3},
    "medicine": {"health": +0.5, "cooldown_h": 6, "requires_sick": True},
}

CRITICAL_HEALTH_THRESHOLD = 0.1
SICK_HEALTH_THRESHOLD = 0.3
CRITICAL_DEATH_HOURS = 6  # Hours at critical before death


def _get_conn():
    from modules.config import connect_fts
    return connect_fts()


def _parse_ts(ts_str: str | None) -> datetime | None:
    if not ts_str:
        return None
    try:
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=TZ_COL)
        return dt
    except (ValueError, TypeError):
        return None


# ============================================================
# LAZY EVALUATION — compute real-time state from last snapshot
# ============================================================

def get_current_state(pet_row: dict | None = None) -> dict | None:
    """Compute real-time pet state from last known values + elapsed time.

    This is the core innovation: no tick needed. State is always
    current when read, using math on elapsed time.
    """
    if pet_row is None:
        pet_row = _get_active_pet()
    if pet_row is None:
        return None

    if not pet_row.get("alive", True):
        return {**pet_row, "mood": "dead", "needs_care": False, "stage": pet_row.get("stage", "egg")}

    now = now_col()
    last_interaction = _parse_ts(pet_row.get("last_interaction_at"))
    if last_interaction is None:
        last_interaction = now

    elapsed_hours = max(0.0, (now - last_interaction).total_seconds() / 3600)
    stage = pet_row.get("stage", "egg")

    # Check stage evolution
    stage, hatched = _evolve_stage(pet_row, now)

    rates = DECAY_RATES.get(stage, DECAY_RATES["adult"])

    # Compute current stats
    hunger = min(1.0, pet_row.get("hunger", 0.0) + elapsed_hours * rates["hunger"])
    happiness = max(0.0, pet_row.get("happiness", 0.8) - elapsed_hours * rates["happiness"])
    energy = max(0.0, pet_row.get("energy", 1.0) - elapsed_hours * rates["energy"])

    # Health degrades if hunger > 0.8 or energy < 0.2
    base_health = pet_row.get("health", 1.0)
    health_penalty = 0.0
    if hunger > 0.8:
        health_penalty += (hunger - 0.8) * 0.5 * elapsed_hours
    if energy < 0.2:
        health_penalty += (0.2 - energy) * 0.4 * elapsed_hours
    health = max(0.0, base_health - health_penalty)

    # Check death
    alive = pet_row.get("alive", True)
    if health <= 0.0:
        alive = False
        health = 0.0

    # Compute mood
    mood = _compute_mood(hunger, happiness, energy, health, alive)

    # Needs care?
    needs_care = (
        hunger > 0.6 or happiness < 0.3 or energy < 0.3 or health < SICK_HEALTH_THRESHOLD
    )

    # Age
    born_at = _parse_ts(pet_row.get("born_at"))
    age_hours = (now - born_at).total_seconds() / 3600 if born_at else 0

    return {
        "id": pet_row.get("id"),
        "name": pet_row.get("name", "???"),
        "stage": stage,
        "alive": alive,
        "hunger": round(hunger, 3),
        "happiness": round(happiness, 3),
        "energy": round(energy, 3),
        "health": round(health, 3),
        "mood": mood,
        "needs_care": needs_care,
        "age_hours": round(age_hours, 1),
        "born_at": pet_row.get("born_at"),
        "last_interaction_at": pet_row.get("last_interaction_at"),
        "total_care_actions": pet_row.get("total_care_actions", 0),
        "elapsed_hours_since_care": round(elapsed_hours, 1),
    }


def _evolve_stage(pet_row: dict, now: datetime) -> tuple[str, bool]:
    """Check if pet should evolve to next stage. Returns (new_stage, just_hatched)."""
    stage = pet_row.get("stage", "egg")
    born_at = _parse_ts(pet_row.get("born_at"))
    if not born_at:
        return stage, False

    age_hours = (now - born_at).total_seconds() / 3600
    hatched = False

    # Walk through stages
    cumulative = 0
    for s in ["egg", "baby", "child", "teen", "adult"]:
        min_hours, next_stage = STAGE_TRANSITIONS[s]
        if min_hours is None:
            break
        cumulative += min_hours
        if age_hours >= cumulative and stage == s:
            stage = next_stage
            if s == "egg":
                hatched = True
                # Persist hatch
                try:
                    conn = _get_conn()
                    try:
                        conn.execute(
                            "UPDATE pets SET stage = ?, hatched_at = ? WHERE id = ?",
                            (next_stage, now_iso(), pet_row["id"])
                        )
                        conn.commit()
                    finally:
                        conn.close()
                except Exception:
                    pass
            else:
                try:
                    conn = _get_conn()
                    try:
                        conn.execute(
                            "UPDATE pets SET stage = ? WHERE id = ?",
                            (next_stage, pet_row["id"])
                        )
                        conn.commit()
                    finally:
                        conn.close()
                except Exception:
                    pass
            break  # one stage transition per call; next tick advances further

    return stage, hatched


def _compute_mood(hunger, happiness, energy, health, alive):
    if not alive:
        return "dead"
    if health < CRITICAL_HEALTH_THRESHOLD:
        return "critical"
    if health < SICK_HEALTH_THRESHOLD:
        return "sick"
    if hunger > 0.8:
        return "starving"
    if energy < 0.2:
        return "exhausted"
    if hunger > 0.6:
        return "hungry"
    if happiness < 0.3:
        return "sad"
    if happiness > 0.7 and energy > 0.5:
        return "happy"
    if energy < 0.4:
        return "tired"
    return "content"


# ============================================================
# CRUD
# ============================================================

def _get_active_pet() -> dict | None:
    """Get the current alive pet, if any."""
    try:
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM pets WHERE alive = 1 ORDER BY id DESC LIMIT 1"
            ).fetchone()
        finally:
            conn.close()
        if row:
            cols = ["id", "name", "stage", "born_at", "hatched_at", "died_at", "alive",
                    "hunger", "happiness", "energy", "health",
                    "last_interaction_at", "last_fed_at", "last_played_at",
                    "last_rested_at", "last_cleaned_at", "last_medicine_at",
                    "times_fed", "times_played", "times_rested", "total_care_actions"]
            return dict(zip(cols, row))
    except Exception as e:
        _logger.error("_get_active_pet error: %s", e)
    return None


def adopt_pet(name: str) -> dict:
    """Adopt a new pet. Only one alive pet at a time."""
    existing = _get_active_pet()
    if existing:
        return {"error": f"Ya tienes a {existing['name']} vivo. Cuida a tu pet actual."}

    conn = _get_conn()
    now = now_iso()
    try:
        conn.execute(
            """INSERT INTO pets (name, stage, born_at, last_interaction_at,
               hunger, happiness, energy, health)
               VALUES (?, 'egg', ?, ?, 0.0, 0.8, 1.0, 1.0)""",
            (name, now, now)
        )
        conn.commit()
        pet_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    finally:
        conn.close()

    # Emit event
    try:
        from modules.events import event_bus, Events
        event_bus.emit(Events.PET_STATE_CHANGED, {
            "pet_name": name, "event": "adopted", "stage": "egg"
        })
    except Exception:
        pass

    _logger.info("Pet adopted: %s (id=%d)", name, pet_id)
    return {"ok": True, "name": name, "id": pet_id, "stage": "egg",
            "message": f"{name} ha nacido como un huevo. En ~2h va a eclosionar."}


# ============================================================
# CARE ACTIONS
# ============================================================

def care_for_pet(action: str) -> dict:
    """Execute a care action. Returns result dict."""
    if action not in ACTIONS:
        return {"error": f"Accion desconocida: {action}. Opciones: {list(ACTIONS.keys())}"}

    pet_row = _get_active_pet()
    if not pet_row:
        return {"error": "No tienes pet. Usa adopt_pet(nombre) primero."}

    state_before = get_current_state(pet_row)
    if not state_before or not state_before.get("alive"):
        return {"error": f"{pet_row['name']} ha muerto. Puedes adoptar uno nuevo."}

    if state_before["stage"] == "egg":
        return {"error": f"{pet_row['name']} es todavia un huevo. Espera a que eclosione (~2h)."}

    action_def = ACTIONS[action]

    # Check requires_sick
    if action_def.get("requires_sick") and state_before["health"] >= SICK_HEALTH_THRESHOLD:
        return {"error": f"{pet_row['name']} no esta enfermo. La medicina solo funciona cuando health < {SICK_HEALTH_THRESHOLD}."}

    # Check cooldown
    cooldown_h = action_def["cooldown_h"]
    ts_fields = {
        "feed": "last_fed_at",
        "play": "last_played_at",
        "rest": "last_rested_at",
        "clean": "last_cleaned_at",
        "medicine": "last_medicine_at",
    }
    last_ts = _parse_ts(pet_row.get(ts_fields[action]))
    if last_ts:
        elapsed = (now_col() - last_ts).total_seconds() / 3600
        if elapsed < cooldown_h:
            remaining = cooldown_h - elapsed
            return {
                "error": f"Cooldown activo para {action}. Faltan {remaining:.1f}h.",
                "cooldown_remaining_h": round(remaining, 1),
            }

    # Apply effects — first persist the decayed state, then apply action
    now = now_iso()
    new_hunger = state_before["hunger"]
    new_happiness = state_before["happiness"]
    new_energy = state_before["energy"]
    new_health = state_before["health"]

    for stat, delta in action_def.items():
        if stat in ("cooldown_h", "requires_sick"):
            continue
        if stat == "hunger":
            new_hunger = max(0.0, min(1.0, new_hunger + delta))
        elif stat == "happiness":
            new_happiness = max(0.0, min(1.0, new_happiness + delta))
        elif stat == "energy":
            new_energy = max(0.0, min(1.0, new_energy + delta))
        elif stat == "health":
            new_health = max(0.0, min(1.0, new_health + delta))

    # Persist
    conn = _get_conn()
    try:
        update_fields = {
            "hunger": new_hunger,
            "happiness": new_happiness,
            "energy": new_energy,
            "health": new_health,
            "last_interaction_at": now,
            "total_care_actions": pet_row.get("total_care_actions", 0) + 1,
        }
        # Set action-specific timestamp
        update_fields[ts_fields[action]] = now

        # Increment counter
        counter_map = {"feed": "times_fed", "play": "times_played", "rest": "times_rested"}
        if action in counter_map:
            update_fields[counter_map[action]] = pet_row.get(counter_map[action], 0) + 1

        set_clause = ", ".join(f"{k} = ?" for k in update_fields)
        values = list(update_fields.values()) + [pet_row["id"]]
        conn.execute(f"UPDATE pets SET {set_clause} WHERE id = ?", values)

        # Log care action
        state_after = {
            "hunger": round(new_hunger, 3), "happiness": round(new_happiness, 3),
            "energy": round(new_energy, 3), "health": round(new_health, 3),
        }
        conn.execute(
            """INSERT INTO pet_care_log (pet_id, action, state_before, state_after)
               VALUES (?, ?, ?, ?)""",
            (pet_row["id"], action, json.dumps({
                "hunger": state_before["hunger"], "happiness": state_before["happiness"],
                "energy": state_before["energy"], "health": state_before["health"],
            }), json.dumps(state_after))
        )
        conn.commit()
    finally:
        conn.close()

    # Emit event
    try:
        from modules.events import event_bus, Events
        event_bus.emit(Events.PET_STATE_CHANGED, {
            "pet_name": pet_row["name"], "event": "cared",
            "action": action, **state_after,
        })
    except Exception:
        pass

    mood = _compute_mood(new_hunger, new_happiness, new_energy, new_health, True)
    return {
        "ok": True,
        "action": action,
        "pet_name": pet_row["name"],
        "mood": mood,
        **state_after,
        "message": _care_message(action, pet_row["name"], mood),
    }


def _care_message(action: str, name: str, mood: str) -> str:
    msgs = {
        "feed": f"Alimentaste a {name}. Se ve mas tranquilo.",
        "play": f"Jugaste con {name}. Esta contento!",
        "rest": f"Dejaste descansar a {name}. Recupero energia.",
        "clean": f"Limpiaste a {name}. Mejor higiene, mejor salud.",
        "medicine": f"Le diste medicina a {name}. Esperemos que mejore.",
    }
    return msgs.get(action, f"Cuidaste a {name}.")


def _persist_death(pet_id: int):
    """Mark pet as dead. Irreversible."""
    try:
        conn = _get_conn()
        try:
            conn.execute(
                "UPDATE pets SET alive = 0, died_at = ?, health = 0.0 WHERE id = ?",
                (now_iso(), pet_id)
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        _logger.error("_persist_death error: %s", e)


# ============================================================
# CONSCIOUSNESS INTEGRATION HELPERS
# ============================================================

def get_pet_awareness() -> dict | None:
    """Called by self_model tick to check pet needs.
    Returns awareness dict or None if no pet.
    """
    state = get_current_state()
    if state is None:
        return None

    # Check for death
    if not state["alive"] and state.get("id"):
        _persist_death(state["id"])
        return {"alert": "death", "name": state["name"],
                "message": f"Mi pet {state['name']} ha muerto. No lo cuide a tiempo."}

    # Persist decayed state periodically (every check updates snapshot)
    if state.get("id"):
        try:
            conn = _get_conn()
            try:
                conn.execute(
                    """UPDATE pets SET hunger = ?, happiness = ?, energy = ?,
                       health = ?, stage = ?, last_interaction_at = ? WHERE id = ?""",
                    (state["hunger"], state["happiness"], state["energy"],
                     state["health"], state["stage"], now_iso(), state["id"])
                )
                conn.commit()
            finally:
                conn.close()
        except Exception:
            pass

    return state


def get_pet_critical_alert() -> str | None:
    """Called by proactive_contact tick. Returns alert string or None."""
    state = get_current_state()
    if state is None or state.get("stage") == "egg":
        return None
    if not state.get("alive"):
        return f"URGENTE: Mi pet {state['name']} ha muerto."
    if state["health"] < CRITICAL_HEALTH_THRESHOLD:
        return f"CRITICO: {state['name']} tiene health={state['health']:.2f}. Va a morir si no actuo."
    if state["health"] < SICK_HEALTH_THRESHOLD:
        return f"URGENTE: {state['name']} esta enfermo (health={state['health']:.2f}). Necesita medicina."
    if state["hunger"] > 0.8:
        return f"ALERTA: {state['name']} tiene mucha hambre (hunger={state['hunger']:.2f})."
    return None


# ============================================================
# MCP TOOLS
# ============================================================

def register_tools(mcp):

    @mcp.tool()
    def pet_status() -> str:
        """Show current pet status with real-time computed stats."""
        state = get_current_state()
        if state is None:
            return json.dumps({
                "status": "no_pet",
                "message": "No tienes pet. Usa adopt_pet('nombre') para adoptar uno."
            })
        return json.dumps(state, indent=2, ensure_ascii=False)

    @mcp.tool()
    def adopt_pet_tool(name: str) -> str:
        """Adopt a new digital pet. It starts as an egg and hatches in ~2h.

        Args:
            name: Name for your new pet
        """
        result = adopt_pet(name)
        return json.dumps(result, ensure_ascii=False)

    @mcp.tool()
    def care_for_pet_tool(action: str) -> str:
        """Care for your pet. Actions: feed, play, rest, clean, medicine.

        Args:
            action: Care action (feed|play|rest|clean|medicine)
        """
        result = care_for_pet(action)
        return json.dumps(result, ensure_ascii=False)
