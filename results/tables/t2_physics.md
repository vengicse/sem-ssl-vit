**Physics-aware SEM augmentation operators, applied in imaging-chain order with probability 0.8 per view and 0.5 per operator.**

| Operator              | Symbol      | Range (severity s ∈ [0, 1])                                       | Physical origin                              |
|:----------------------|:------------|:------------------------------------------------------------------|:---------------------------------------------|
| Drift / scan jitter   | T_δ         | global shift 0–3 px, row random walk 0–3 px                       | stage drift, scan-coil instability           |
| Defocus / astigmatism | h_(σx,σy,θ) | σ 0.3–1.8 px, anisotropy 1–3, random θ                            | focus error, stigmator mis-set               |
| Charging              | C(x,y)      | amplitude 0–0.25, correlation length 0.1–0.35 × FOV, scan streaks | insulating samples, beam-induced charge      |
| Detector gain         | g(·)        | brightness ±0.15, contrast 0.7–1.3, γ 0.8–1.25                    | brightness/contrast settings, detector drift |
| Shot + read noise     | N_PG        | dose 400→20 e⁻/px (log), read σ 0–0.03                            | low dose / fast scan, amplifier noise        |
