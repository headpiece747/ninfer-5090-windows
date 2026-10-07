# The serving lanes got slower, and noisier, between 2026-09-30 and 2026-10-07

Measurement record, 2026-10-07. Written because the published lane table
(`README.md`, "Profiles and launchers") is the 2026-09-30 measurement and nothing had re-taken it
after the three upstream merges that landed in between. The numbers below are the eight shipped
profiles, measured with the matrix's own `profile` mode on the same protocol the table was measured
with (`documented` sampling, `code` domain, vision, `--device-state-slots 1`, 262,144 context), three
measured runs each. Raw records: `C:\AI\bench\matrix_v3.jsonl` (the last eight entries).

| lane | 2026-09-30 | 2026-10-07 | runs | acceptance 09-30 -> 10-07 | round cost |
|---|---:|---:|---|---|---:|
| quasar dflash2 | 319 | 321.8 | 270.9, 337.6, 356.8 | 55.0% -> 71.2% | 15.2 -> 18.6 ms (+22%) |
| quasar mtp4 | 221 | 169.1 | 185.6, 137.8, 184.0 | 59.6% -> 63.5% | 15.3 -> 20.9 ms (+37%) |
| ninfer dflash2 | 296 | 265.5 | 166.1, 304.7, 325.8 | 48.7% -> 63.0% | 14.9 -> 20.4 ms (+37%) |
| ninfer mtp4 | 190 | 123.2 | 150.7, 107.3, 111.4 | 51.4% -> 46.7% | 16.1 -> 23.3 ms (+45%) |
| swift dflash2 | 362 | 241.1 | 260.7, 298.7, 163.9 | 63.0% -> 58.3% | 14.9 -> 21.1 ms (+41%) |
| swift mtp4 | 218 | 144.7 | 124.9, 186.5, 122.7 | 58.7% -> 57.7% | 15.4 -> 22.9 ms (+49%) |
| nvidia dflash2 | 338 | 173.5 | 141.9, 230.4, 148.3 | 56.2% -> 43.5% | 14.6 -> 23.3 ms (+60%) |
| nvidia mtp4 | 210 | 160.4 | 174.4, 125.4, 181.4 | 53.1% -> 57.3% | 14.9 -> 20.5 ms (+38%) |

**Round cost is the comparable quantity.** Tokens per speculative round are `1 + draft x acceptance`,
so `round = (1 + draft x acceptance) x 1000 / decode_tps`. Raw tok/s is not comparable across the two
dates: the tokenizer fix of 2026-10-04 (see the note at the top of `docs/perplexity-baseline.md`)
tokenizes the prompt differently, every lane's deterministic digest differs from its 2026-09-30 value,
and the generated text is therefore not the same text. Round cost removes the acceptance term, and it
rose on every lane: **+22% to +60%**.

**The spread widened too, and that is a separate observation.** The 2026-09-30 records were tight --
quasar mtp4 `[221.4, 220.6, 221.5]`, swift dflash2 `[363.8, 359.3, 362.7]`, nvidia dflash2
`[338.2, 338.3, 336.1]` -- within +/-1%. Today's are `[185.6, 137.8, 184.0]`, `[260.7, 298.7, 163.9]`
and `[141.9, 230.4, 148.3]`, up to +/-30% on the same protocol. A three-run mean is a weak statistic
under that spread, which is why the runs are printed here rather than only the mean.

**Ruled out.** The CI runner did not take the card *during the run*: the newest worker log ends at
`12:31Z`, the sweep ran at `17:11-17:14Z`, and nothing else was serving. (Two `gpu.yml` jobs did run
earlier the same day, `10:56Z` and `12:18Z`; both finished hours before this measurement.) The artifacts
are not the cause: all five were rebuilt the same day and every weight object is byte-identical to the
2026-09-30 build (their only difference is the embedded chat template, §22 of the artifact reference).
The card is not contended.

**Not ruled out, and the candidate set.** 252 files under `src/` changed between the two dates across
three merges (`b9114396`/`355dda55`, `809296dd`, `423f0117`), and two attention launch-plan commits
landed on 2026-09-30 itself (`107b174e` "stop splitting a saturated tiled launch", `e621c7d6` "derive
launch plans from device sm count"), so whether the 2026-09-30 table was measured before or after those
two is not established. A bisect would settle it: build at each candidate and run one lane's round cost
interleaved against the current build.

**The baseline matters for two of the eight lanes.** The table above uses the published table's
2026-09-30 figures. For six lanes the most recent pre-today *record* in `matrix_v3.jsonl` agrees with
them to within a token per second; for two it does not, and in both cases the published figure is the
more favourable one. `start_quasar_v3_dflash2_vision` last recorded 363.0 tok/s at 66.5% acceptance
against the published 319 at 55.0%, so against its own record that lane reads **-11.4%**, not the +0.9%
the table implies. `start_nvidia_v3_mtp4_vision` last recorded 202.6 against the published 210, so its
drop is -20.8% against the record and -23.6% against the table. The remaining six: -10.3%, -23.5%,
-33.4%, -33.6%, -35.2%, -48.7% against the published figures. Both baselines are printed here because
neither has been reconciled to a date; the records are the like-for-like measurement, the table is the
published claim.

**What this does not say.** It does not say the artifacts are worse -- their weights are unchanged --
and it does not say the engine is wrong, only slower and less steady on these lanes. It also does not
re-derive the 2026-09-30 figures; those remain the published table, now with this note beside them.
