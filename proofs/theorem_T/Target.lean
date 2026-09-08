-- The target theorem T. By convention, set target_theorem_name in the config
-- to this declaration's exact name; its dependency closure within this
-- folder (here: aux_bound_one, aux_bound_two) is auto-detected as s1..sn.

theorem target_theorem_example (n : Nat) : n ≤ n + 1 := by
  exact aux_bound_one n
