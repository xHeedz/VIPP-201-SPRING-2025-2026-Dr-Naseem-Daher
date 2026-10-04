# knowledge: rules used only to explain a flagged window

one file per region. each file starts with machine readable lines `key: value` (read by `model/rag.py` `load_knowledge`), followed by the sources. these values never enter the score; they only appear in the text of `explain()`, which names the file it used.

keys: `following_s` (recommended minimum time gap), `following_rule` (the rule in words), `limit_urban_kmh`, `limit_rural_kmh`, `limit_motorway_kmh`, `dataset` (which data in the repo comes from this region).
