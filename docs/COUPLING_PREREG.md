# Coupling ablation: accumulated evidence re-alignment - pre-registration

Status: **proposed, awaiting approval, not started.** It follows
`E2_EVIDENCE_RESULTS.md`, which closed E2 as a non-execution: the evidence
pathway works in isolation (78.80 % on one expert) but the continual multi-expert
formulation collapses (13.04 % at 20 experts), and the `lambda = 0` diagnostic did
not isolate the cause because it also removes the only training signal for the
shared routing pathway.

## 1. The question, and the smallest useful test of it

> How does continual coupling across accumulated experts prevent a locally valid
> evidence representation from remaining usable globally?

The neutral question does not name a component, so the first study tests **one**
coupling variable, chosen because it is the heaviest and most novel mechanism E2
introduced: E2 re-aligned the evidence projections of **all** seen experts at
every optimizer step. The two arms:

```text
C0   the current W_t is trainable; W_1..W_{t-1} are frozen
C1   all W_1..W_t are trainable
```

Everything else is identical and unchanged from the E2 contract: the shared query
`P`, the shared readout `g`, the current adapter `E_t`, `L_task`, `L_evidence`
(averaged, never summed, with no `g` inside it), the winner-take-all decision, the
inference path, the seeds, the budget.

**`g` is deliberately not touched.** Changing the readout as well would collapse
two coupling factors into one comparison, which is what the previous three studies
established is the way to learn nothing.

## 2. Endpoints

**Primary mechanistic metric:** `Delta C@3` between C1 and C0 - the question is
whether evidence alignment breaks routing.

**Read together:** `Acc`, `conditional_oracle@3`, `Ceiling@3`. A `C@3` change with
an oracle change means the coupling affects usability as well as routing, and the
report must say which.

## 3. Outcome reading, fixed in advance

| result | reading |
| :-- | :-- |
| C1 > C0 | global evidence re-alignment helps; accumulated experts benefit from being re-aligned |
| C1 ~ C0 | joint alignment is not necessary at this scale |
| C1 < C0 | continually re-aligning accumulated experts produces destructive interference |

**None of the three says the shared readout is the cause.** That is a separate
coupling variable and a separate pre-registration.

## 4. Feasibility is a first-class constraint, not a footnote

E2 measured **3095 s per 20-expert cell (51.6 min)** under the pinned contract, so
a 24-cell grid is about **20.6 hours**. That is not a note; it is a hard design
constraint here.

```text
declared grid          2 arms x 2 regimes x 3 seeds = 12 cells
measured cell cost     3095 s (E2, seed 42, coherent)
projected total        12 x 3095 s ~ 10.3 h
hard ceiling           12 h of training wall-clock
development smoke      one arm pair, one regime, one seed (2 cells, ~1.7 h)
```

**The smoke decides execution, and the protocol is never sped up inside this
study.** If the measured smoke cost projects the grid above the ceiling, the study
is **not executed** and is reported as a feasibility failure; if the smoke's arms
are degenerate the study is not executed either. Making it cheaper - sparser
evidence loss, prototype subset sampling, fewer epochs, a smaller bank - is a
**separate feasibility study with its own pre-registration**, because each of those
changes the mechanism under test.

## 5. Vetoes

```text
C1 reproduces E2          the C1 arm must return E2's lambda = 1 numbers exactly
                          (13.04 % / 0.9041 at seed 42, coherent), since it is the
                          same protocol
C0 gradient guard         only W_t receives gradient; W_1..W_{t-1} have none -
                          a runtime check, recorded per task
either arm degenerate     both arms must classify above chance-level to give a
                          usable operating point
cost projection           the smoke's measured cost must project under the ceiling
```

## 6. Statistics

If executed, the S11 protocol with the sample size the ceiling allows (three seeds
declared; if only a reduced grid fits, the actual `n` is reported and the
paired comparisons are run on what exists rather than on what was intended).

## 7. Out of scope

```text
the shared readout g        a separate coupling variable, separate pre-registration
the shared query P          likewise
expert accumulation itself  a task-count ablation, separate
protocol speed-ups          a separate feasibility study, as above
L_task or L_evidence form   closed by the earlier studies
```
