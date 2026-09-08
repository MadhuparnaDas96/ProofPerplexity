-- Helper lemmas (project-local proof steps) for target_theorem_example.
-- Cites real Mathlib declarations (Order.le_succ, comp_mul_left) so this
-- demo is meaningful against the bundled Mathlib library, not just a smoke test.

theorem aux_bound_one (n : Nat) : n ≤ n + 1 := by
  exact Order.le_succ n

theorem aux_bound_two (a b : Nat) : (a * ·) ∘ (b * ·) = (a * b * ·) := by
  exact comp_mul_left a b
