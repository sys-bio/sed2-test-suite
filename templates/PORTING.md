# Porting the sed/ test templates

The 39 documents in `sed/tests/unit test files/` (the old `versionStr`/`versionNum` format) were converted to the
current specification and checked with libsed2: every document in this folder validates (`tests/test_templates.py`).
They are templates for the cases that go into `cases/semantic/`, not test cases: they have no expected results
and no `settings.json`.  The old expected outputs in `sed/tests/expected test results/` were not used (not hand
checked); results here come from hand analysis or from simulators that agree.

Mechanical changes: `versionStr`/`versionNum` became `"version": "v1.0.0"`; `kisaoID` became
`workingAlgorithms: [{"algorithm": ...}]`; `outputModel` was dropped.  Models and data are next to the documents
(`three_species_chain.xml`, `three_species_chain_stochlevels.xml`, `experimental_data.csv`).

| Template(s) | Result | What changed |
|---|---|---|
| SEDDocument_empty | ported | header only |
| constant_value, constant_list, constant_matrix | ported | header only |
| constant_dict_num, constant_dict_list | ported with a change | reporting a dictionary constant directly is rejected (SEDBase-0017); the report now goes through a createDataBlock task (TODO.md: reporting a dictionary constant) |
| model_import | ported | header only |
| model_full_output | closest valid form | a whole model cannot be reported; the document lists the model's element ids instead (TODO.md) |
| explicit_ode_simulation, bounded_ode_simulation | ported | `kisaoID` became `workingAlgorithms`; `outputModel` dropped |
| steadyState | ported | same changes |
| jacobian | ported | `_type: KISAO:0000809` became `jacobianFull` |
| data_import | ported | validates; the translator skips DataImport (TODO.md: CsvImport) |
| cosimulation | ported | loop-variable models are referenced without `.model`; needs monod_CRM.xml and ecoli_GSM.xml, which are not in the repositories; the translator skips FluxBalanceAnalysis |
| explicit_stochastic_simulation, scatter_stochastic_simulation | ported | validates; stochastic tasks are not translated yet |
| scatter_stochastic_simulation_aggregate | closest valid form | aggregates indexed by position, not by key (TODO.md: labels of `[id].aggregates`); the aggregation functions cannot be selected (TODO.md: AggregationCalculation) |
| calculate_aggregate_* (22 files) | closest valid form | `kisaoID` is not allowed on AggregationCalculation and there is no function selector; the intended function stays in `name` and `notes` (TODO.md: AggregationCalculation).  Each is the same document with a different function |

Questions for the specification found on the way are in `SED2/TODO.md` (section "Reporting a dictionary
constant or a whole model; labels of `[id].aggregates`", and the AggregationCalculation and CsvImport sections).
