.PHONY: compile eval-schema-linked eval-repair eval-controlled-agent

compile:
	PYTHONPYCACHEPREFIX=.pycache python3 -m compileall src scripts

eval-schema-linked:
	PYTHONPATH=src python3 scripts/run_schema_linked.py --db-id california_schools --limit 50 --promptfix

eval-repair:
	PYTHONPATH=src python3 scripts/run_strict_repair.py

eval-controlled-agent:
	PYTHONPATH=src python3 scripts/run_controlled_agent.py --db-id california_schools --limit 50
