#!/usr/bin/env python3
"""Verify the Phase 2b queue contains only Table VII relevant entries.

Prints topic distribution, display_identifier values, and any entries that look
off-topic. Doesn't touch any files.
"""
import json
from collections import Counter
from pathlib import Path

QUEUE = Path.home() / "census_toc" / "tableVII_newocr.jsonl"

entries = []
with QUEUE.open() as f:
    for line in f:
        entries.append(json.loads(line))

print(f"Total queue entries: {len(entries)}\n")

topics = Counter(e.get("canonical_topic", "?") for e in entries)
print("canonical_topic distribution:")
for topic, n in topics.most_common():
    print(f"  {topic:35s} {n}")
print()

identifiers = Counter(e.get("display_identifier", "?") for e in entries)
print("display_identifier distribution (top 15):")
for ident, n in identifiers.most_common(15):
    print(f"  {ident!r:30s} {n}")
print()

tiers = Counter(e.get("relevance_tier", "?") for e in entries)
print("relevance_tier distribution:")
for tier, n in tiers.most_common():
    print(f"  {tier:20s} {n}")
print()

# Flag anything that doesn't look age/civil-condition/Table VII-ish
OK_TOPICS = {"age_civil_condition", "age_distribution", "age_sex_population", "age_by_religion"}
off_topic = [e for e in entries if e.get("canonical_topic") not in OK_TOPICS]
print(f"Off-topic (not age/civil-condition family): {len(off_topic)}")
for e in off_topic[:10]:
    print(f"  topic={e.get('canonical_topic')!r:30s} ident={e.get('display_identifier')!r:15s} title={(e.get('title') or '')[:60]!r}")
