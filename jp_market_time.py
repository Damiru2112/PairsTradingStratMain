from __future__ import annotations
import time
import logging
from datetime import datetime, timezone

log = logging.getLogger("jp_market_time")

def sleep_until_next_15m_close(buffer_s: float = 5.0):
    """
    Sleeps until the next xx:00, xx:15, xx:30, or xx:45 mark.
    Adds 'buffer_s' to ensure the candle is fully closed at the broker side.
    """
    now = datetime.now()
    minutes = now.minute
    seconds = now.second
    micros = now.microsecond
    
    # Minutes to next quarter
    # e.g. 10:14 -> 10:15 => 1 min remaining
    # e.g. 10:15 -> 10:30 => 15 mins remaining
    remainder = minutes % 15
    # if remainder is 0 and we are just past the mark (e.g. 10:15:05), we want next one (10:30)
    # logic: (15 - remainder) gives 15 if remainder is 0.
    
    minutes_to_wait = 15 - remainder
    
    # If we are cleanly on the minute (unlikely), minutes_to_wait is correct.
    # If we are passed the minute (e.g. 10:15:05), minutes % 15 is 0, wait 15m.
    # Correct.
    
    total_seconds_wait = (minutes_to_wait * 60) - seconds - (micros / 1_000_000)
    
    # Add buffer
    final_wait = total_seconds_wait + buffer_s
    
    if final_wait < 0:
        # Should not happen unless buffer is negative or logic edge case
        final_wait = 0
        
    log.info(f"Sleeping {final_wait:.1f}s until next 15m mark (+{buffer_s}s buffer)")
    if final_wait > 0:
        time.sleep(final_wait)


def sleep_for_next_bar(interval_s: float = 60.0, buffer_s: float = 5.0):
    """
    Sleep for a fixed interval (default 60s) plus buffer.
    
    Use this when polling for delayed data instead of wall-clock alignment.
    The idea is: we poll every ~60s, but data is 15min delayed so wall-clock
    alignment doesn't help.
    
    Args:
        interval_s: Base polling interval in seconds
        buffer_s: Additional buffer to ensure bar is fully available
    """
    total = interval_s + buffer_s
    log.info(f"Sleeping {total:.1f}s (interval={interval_s}s + buffer={buffer_s}s)")
    time.sleep(total)


def sleep_short(seconds: float = 10.0):
    """
    Short sleep when no new bar is available yet.
    Avoids busy loop while waiting for new data.
    """
    log.debug(f"No new data, sleeping {seconds:.1f}s")
    time.sleep(seconds)


def get_current_utc_ts() -> datetime:
    """Get current timestamp in UTC (timezone-aware)."""
    return datetime.now(timezone.utc)

