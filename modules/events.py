"""
Codi Memory - Event Bus Architecture (Phase 0)
===============================================
Simple publish-subscribe event system for cross-module communication.

Enables subsystems to react to events without tight coupling.
Based on architectural recommendation #5 from NEURO_ANALYSIS_REPORT.

Events:
  MEMORY_STORED     - fired after a memory is added to any store
  MEMORY_RETRIEVED  - fired after a memory is retrieved/searched
  EMOTION_CHANGED   - fired when PAD emotional state changes
  WORKSPACE_BROADCAST - fired when global workspace broadcasts
  CONSOLIDATION_COMPLETE - fired after consolidation cycle (future)
  PREDICTION_ERROR  - fired when prediction doesn't match reality (future)

Usage:
  from modules.events import event_bus, Events

  # Subscribe
  event_bus.on(Events.MEMORY_STORED, my_handler)

  # Publish
  event_bus.emit(Events.MEMORY_STORED, {'memory_id': '...', 'content': '...'})
"""

import threading
from collections import defaultdict
from datetime import datetime


class Events:
    """Event name constants."""
    MEMORY_STORED = 'memory_stored'
    MEMORY_RETRIEVED = 'memory_retrieved'
    EMOTION_CHANGED = 'emotion_changed'
    WORKSPACE_BROADCAST = 'workspace_broadcast'
    CONSOLIDATION_COMPLETE = 'consolidation_complete'
    PREDICTION_ERROR = 'prediction_error'
    WORKSPACE_COMPETITION_COMPLETE = 'workspace_competition_complete'  # WIRING-6
    RETRIEVAL_QUALITY = 'retrieval_quality'                            # WIRING-7
    SCHEMA_UPDATED = 'schema_updated'                                  # Phase 3A
    RECONSOLIDATION_TRIGGERED = 'reconsolidation_triggered'            # Phase 4


class EventBus:
    """
    Simple thread-safe publish-subscribe event bus.

    Handlers are called synchronously in the order they were registered.
    Exceptions in handlers are caught and logged (never block the emitter).
    """

    def __init__(self):
        self._handlers = defaultdict(list)
        self._lock = threading.Lock()
        self._history = []  # Last N events for debugging
        self._history_max = 50

    def on(self, event_name: str, handler: callable):
        """Register a handler for an event.

        Args:
            event_name: Event to listen for (use Events constants)
            handler: Callable that receives (event_name, data_dict)
        """
        with self._lock:
            if handler not in self._handlers[event_name]:
                self._handlers[event_name].append(handler)

    def off(self, event_name: str, handler: callable):
        """Unregister a handler."""
        with self._lock:
            try:
                self._handlers[event_name].remove(handler)
            except ValueError:
                pass

    def emit(self, event_name: str, data: dict = None):
        """Fire an event, calling all registered handlers.

        Args:
            event_name: Event being fired
            data: Dict of event-specific data
        """
        if data is None:
            data = {}

        # Record in history
        entry = {
            'event': event_name,
            'timestamp': datetime.now().isoformat(),
            'data_keys': list(data.keys())
        }

        with self._lock:
            self._history.append(entry)
            if len(self._history) > self._history_max:
                self._history = self._history[-self._history_max:]
            handlers = list(self._handlers.get(event_name, []))

        # Call handlers outside the lock
        for handler in handlers:
            try:
                handler(event_name, data)
            except Exception as e:
                print(f"[EventBus] Error in handler for {event_name}: {e}")

    def get_history(self, limit: int = 20) -> list:
        """Get recent event history for debugging."""
        with self._lock:
            return list(self._history[-limit:])

    def get_stats(self) -> dict:
        """Get subscriber counts per event."""
        with self._lock:
            return {
                event: len(handlers)
                for event, handlers in self._handlers.items()
                if handlers
            }


# Singleton instance - all modules share this
event_bus = EventBus()
