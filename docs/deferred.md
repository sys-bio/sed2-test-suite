# Deferred until the SED2 format is decided

The suite has no test cases for these, because what a correct interpreter must produce is not yet defined.  Add
tests when the specification settles each one.

* `ModelChange.addElements` and `ModelChange.replaceElements` (and the order in which the four change attributes
  of one ModelChange apply): see SED2/TODO.md.  `setValues` and `removeElements` are tested for the clear cases only
  (a parameter, a compartment, a species in a compartment of volume 1, a reaction); the points that SED2/TODO.md
  lists as open ("ModelChange for SBML, CsvImport and plots") are not.
* `AggregationCalculation` and a repeat's `aggregateOutputVariables`: the function cannot be named in the document.
* `taskParameters` on any task: their meaning is undefined.
* `DataImport`, and `CsvImport.organization`: no data formats / CsvImport attributes are defined yet.

Stochastic simulations and DrawFromDistribution are a later phase (cases/stochastic).

Flux balance analysis has cases (series `fba`, 00274 onwards), with these points left open because the specification does not
say: what an infeasible (or unbounded) problem gives, and what `[id].model` of an analysis is (see SED2/TODO.md).  Neither is tested.
