#!/usr/bin/env python3
"""Rename live registry entity_ids to match default_entity_id declared in config.

Usage: migrate_entity_registry.py <configuration.yaml> <core.entity_registry>

For every template entity whose config declares `unique_id` + `default_entity_id`,
if the live entity registry has that unique_id registered under a different
entity_id (e.g. an old pinyin slug auto-generated from the Chinese name),
rename it to the declared id. Run ONLY while Home Assistant is stopped.
"""
import json
import re
import shutil
import sys
import time

config_path, registry_path = sys.argv[1], sys.argv[2]

# command_line 传感器无法在 YAML 里声明 default_entity_id(HA 不接受该键),
# 因此其 entity_id 会按中文名生成拼音 slug。这里显式指定它们应有的 entity_id,
# 否则 automations/模板里写的 sensor.smart_dehumidifier_* 引用会指向不存在的实体。
EXTRA_MAP = {
    "smart_dehumidifier_ml_engine": "sensor.smart_dehumidifier_ml_engine",
    "smart_dehumidifier_learning_stats": "sensor.smart_dehumidifier_learning_stats",
}

text = open(config_path, encoding="utf-8").read()
pairs = re.findall(
    r"unique_id:\s*([\w]+)\s*\n\s*default_entity_id:\s*([\w.]+)", text
)
desired = {uid: eid for uid, eid in pairs}
desired.update(EXTRA_MAP)
print(f"{len(desired)} unique_id -> entity_id mappings ({len(EXTRA_MAP)} explicit command_line)")

backup = registry_path + ".bak." + time.strftime("%Y%m%d-%H%M%S")
shutil.copy2(registry_path, backup)
print("registry backup:", backup)

data = json.load(open(registry_path, encoding="utf-8"))
entries = data["data"]["entities"]
taken = {e["entity_id"] for e in entries}

renamed = skipped = 0
for e in entries:
    uid = e.get("unique_id")
    if e.get("platform") != "template" and uid not in EXTRA_MAP:
        continue
    target = desired.get(uid)
    if not target or e["entity_id"] == target:
        continue
    if target in taken:
        print(f"SKIP {e['entity_id']} -> {target} (target id already taken)")
        skipped += 1
        continue
    print(f"RENAME {e['entity_id']} -> {target}")
    taken.discard(e["entity_id"])
    taken.add(target)
    e["entity_id"] = target
    renamed += 1

json.dump(data, open(registry_path, "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(f"done: {renamed} renamed, {skipped} skipped")
