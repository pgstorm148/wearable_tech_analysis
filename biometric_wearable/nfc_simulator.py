import uuid
import numpy as np
from typing import List, Dict

class NFCEventSimulator:
    """Simulates NFC touch-to-action events."""
    def __init__(self, duration: int, num_taps: int = 5, seed: int = 42):
        self.duration = duration
        self.num_taps = num_taps
        self.rng = np.random.default_rng(seed)
        self.events = self._generate_events()
        
    def _generate_events(self) -> List[Dict]:
        events = []
        if self.num_taps == 0:
            return events
            
        tap_times = np.sort(self.rng.uniform(1, self.duration - 1, self.num_taps))
        
        if self.num_taps >= 3:
            tap_times[-1] = tap_times[-2] + 0.3
            
        state = "init"
        
        for i, t in enumerate(tap_times):
            if i > 0 and (t - tap_times[i-1]) <= 0.5:
                action = "end_session"
                state = "ended"
            elif state == "init":
                action = "start_session"
                state = "running"
            elif state == "running":
                action = "pause"
                state = "paused"
            elif state == "paused":
                action = "resume"
                state = "running"
            else:
                action = "sync"
                
            event = {
                "event_id": str(uuid.uuid4()),
                "tag_uid": "04:A3:2B:1C:8D:00:91",
                "ndef_payload": "athlete_id=PG001&session=training",
                "timestamp_ms": t * 1000.0,
                "action": action
            }
            events.append(event)
            
        return events
        
    def get_events_in_window(self, start_ms: float, end_ms: float) -> List[Dict]:
        return [e for e in self.events if start_ms <= e["timestamp_ms"] < end_ms]
