.PHONY: compile eval-schema-linked eval-repair eval-controlled-agent manifest-full eval-expanded summarize-expanded

compile:
	PYTHONPYCACHEPREFIX=.pycache python3 -m compileall src scripts

eval-schema-linked:
	PYTHONPATH=src python3 scripts/run_schema_linked.py --db-id california_schools --limit 50 --promptfix

eval-repair:
	PYTHONPATH=src python3 scripts/run_strict_repair.py

eval-controlled-agent:
	PYTHONPATH=src python3 scripts/run_controlled_agent.py --db-id california_schools --limit 50

manifest-full:
	PYTHONPATH=src python3 scripts/build_eval_manifest.py --output results/expanded/full_manifest.jsonl

eval-expanded:
	bash scripts/run_expanded_core_eval.sh results/expanded/full_manifest.jsonl

summarize-expanded:
	PYTHONPATH=src python3 scripts/summarize_expanded_eval.py
