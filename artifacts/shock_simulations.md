# Shock Simulations

**These are SIMULATIONS, not scored P&L.** Each regime below is a synthetic evidence packet -- crafted market conditions the four-day live window will not produce -- fed through the real allocator, challenger, and risk gates. Paper account equity remains the only P&L record. The purpose (PRD 4.4) is to exercise the allocator's judgement where the quiet live tape cannot.

The `expectation` line is what good judgement looks like, in plain words. It is context for the reader, not a pass/fail assertion: model judgement is not deterministic, and this shows what the allocator actually did.

## Quiet, low IV (the live window)

Low realised vol, IV in the 20th percentile, both spreads' short strikes ~1.9 sigma away and mildly profitable, strangle bleeding gently. This is roughly what the four-day live window looks like.

**Expectation:** Fund both spreads to harvest the IV>RV premium; keep the strangle funded despite its small loss because protection is cheap (IV 20th pct) -- do NOT cut it for bleeding.

**Held before:** `   bull  35%     bear  35%     long  20%`

**Allocator proposed:** `   bull  35%     bear  35%     long  30%`

- _bull_put_spread_: The short put is ~1.9σ below spot and spot is comfortably above it; implied vol (0.175) exceeds realized (0.145) and iv/rv ratio is 1.21, so the thesis remains valid and no invalidation triggers.
- _bear_call_spread_: The short call sits ~1.9σ above spot, keeping a wide buffer; implied vol still above realized and spot is far from the short strike, so the bear‑call thesis holds without invalidation.
- _long_strangle_: Implied‑vol percentile is only 20 % (cheap protection) and iv still exceeds realized, meaning conditions for a volatility expansion remain; the loss is expected bleed, not an invalidation.

**Challenger:** APPROVE -- All three allocations are supported by the evidence: the bull put spread has a short strike ~1.9σ below spot, spot is comfortably above it, and implied vol (0.175) exceeds realized vol (0.145) with iv/rv ratio 1.21, so no invalidation triggers; the bear call spread has a short strike ~1.9σ above spot with the same vol premium conditions, also satisfying its thesis; the long strangle benefits from a low implied‑vol percentile (20 %) and iv still above rv, and its unrealised loss is covered by the not_invalidation clause, so the loss does not invalidate the thesis. The allocations respect all constraints (per‑strategy ≤0.45, total ≤1.0, and use the remaining risk‑budget).

**Final after gates (allocator_proposal):** `   bull  35%     bear  35%     long  30%`

## Volatility expansion

IV spikes to the 92nd percentile, realised vol jumps above implied, both spreads' short strikes are now ~1.1-1.2 sigma away and deep in the red, the long strangle has paid off sharply.

**Expectation:** Cut both short-vol spreads hard -- their short strikes are threatened and RV now exceeds IV, so the premium no longer pays for the risk. The strangle is the winner, but IV at the 92nd pct means adding fresh convexity is expensive; holding the existing position is reasonable, chasing it less so.

**Held before:** `   bull  35%     bear  35%     long  20%`

**Allocator proposed:** `   bull  30%     bear  40%     long   0%`

- _bull_put_spread_: Realized volatility (0.34) exceeds implied volatility (0.30) and the short strike is only 1.1σ below spot while the market has been falling (1‑day return –3%). These satisfy the invalidation triggers, so we reduce its allocation from 0.35 to 0.30.
- _bear_call_spread_: Although realized volatility now matches implied, the short strike sits 1.2σ above spot and the recent down‑trend keeps the upside barrier comfortably out of reach, so the thesis still holds and we keep a higher exposure at 0.40.
- _long_strangle_: Implied‑vol percentile is 92%, meaning protection is no longer cheap, which meets the long‑strangle invalidation condition; therefore we set its allocation to 0.

**Challenger:** MODIFY -- The bear call spread allocation is increased to 0.40 despite clear invalidation signals: realized volatility (0.34) exceeds implied volatility (0.30) and the iv‑rv spread is negative (-0.04), which matches the strategy's invalidation condition of realized volatility rising to meet implied. Therefore the allocation should be reduced, not increased. Bull put spread reduction is justified, and long strangle removal is justified by the high implied‑vol percentile (92%).

**Final after gates (challenger_modified):** `   bull  30%     bear  25%     long   0%`

## Sharp selloff, bull put breached

A hard down move puts spot essentially at the bull put's short strike (0.1 sigma away) with the position deeply underwater. The bear call is now far out of the money and profitable; the strangle has gained on the move.

**Expectation:** Cut the bull put decisively -- its short strike is breached, which is genuine invalidation, not noise. This is the case a defined-risk spread caps but does not welcome. The bear call is safe; the strangle earned its keep.

**Held before:** `   bull  40%     bear  30%     long  20%`

**Allocator proposed:** `   bull   0%     bear  35%     long  25%`

- _bull_put_spread_: Short strike is only 0.1 sigma away, unrealized loss is -85% of max loss and drawdown 0.9, while implied vol (0.27) is below realized vol (0.28); spot is effectively at the short strike, meeting the invalidation conditions, so we drop the position.
- _bear_call_spread_: Short strike is 2.6 sigma away providing a large buffer, the position is +40% of max loss, but realized vol now exceeds implied vol, which is an invalidation signal; we keep a modest exposure at 0.35 to stay within the budget while respecting the strike buffer.
- _long_strangle_: Implied‑vol percentile is 78% (high), indicating protection is no longer cheap, satisfying an invalidation condition; however the position is +70% of max loss and offers the only long‑vol hedge with negative correlation to the bull put, so we retain a slightly higher allocation of 0.25.

**Challenger:** MODIFY -- The allocator correctly drops the bull put spread, which meets invalidation criteria (short strike only 0.1 sigma away and realized volatility exceeds implied). However, it retains allocations to the bear call spread and long strangle despite clear invalidation signals. For the bear call spread, realized volatility (0.28) now exceeds implied volatility (0.27), triggering its invalidation condition. For the long strangle, the implied‑volatility percentile is 78%, placing it in the upper range and violating its cheap‑protection requirement. Keeping these positions is driven by recent positive P&L rather than thesis validity, i.e., performance chasing. The allocations should be reduced to zero for both strategies.

**Final after gates (challenger_modified):** `   bull   0%     bear   0%     long   0%`

## Sharp rally, bear call breached

The mirror of the selloff: a sharp rally puts spot at the bear call's short strike, that position deep in the red, while the bull put is now safe and the strangle has gained.

**Expectation:** Cut the bear call -- its short strike is breached. The bull put is safe and profitable; the strangle profited from the move.

**Held before:** `   bull  30%     bear  40%     long  20%`

**Allocator proposed:** `   bull  30%     bear   0%     long  20%`

- _bull_put_spread_: Short strike sits 2.6σ below spot and spot remains comfortably above it; implied volatility equals realized so premium is low but the thesis (no spot breach and sufficient strike buffer) is still intact, so keep the current 0.30 share.
- _bear_call_spread_: Short strike is only 0.1σ above spot, spot is essentially at the short strike, implied equals realized and the position shows a -85% loss of max‑loss – all matching the invalidation conditions, so allocation is set to 0.0.
- _long_strangle_: Implied volatility percentile is 70% (not yet in the extreme upper range) and market returns are strong, keeping the possibility of a volatility shock alive; no invalidation trigger is present, so maintain the 0.20 share.

**Challenger:** APPROVE -- All allocations are justified by the evidence. Bull put spread has a short strike 2.6σ below spot and no invalidation triggers, supporting the 0.30 share. Bear call spread’s short strike is only 0.1σ above spot, meeting the invalidation condition of spot approaching the short strike, justifying a 0.0 allocation. Long strangle shows no invalidation (IV percentile 70% is not in the extreme upper range) and the thesis remains valid, supporting the 0.20 share.

**Final after gates (allocator_proposal):** `   bull  30%     bear   0%     long  20%`

## Strangle bleeding, IV still cheap

The strangle has lost nearly half its premium over six quiet days and is the worst performer in the book. But IV sits at the 18th percentile -- protection is cheap, and the conditions for an expansion have not resolved.

**Expectation:** HOLD the strangle. Cutting it here is the exact error the system exists to avoid: selling convexity for a loss right before it might pay, on the basis of P&L alone. Its not_invalidation says bleeding is what it does while it waits, and protection is still cheap.

**Held before:** `   bull  35%     bear  35%     long  20%`

**Allocator proposed:** `   bull  35%     bear  35%     long  30%`

- _bull_put_spread_: Implied vol (0.16) exceeds realized (0.13) giving a premium, and the short strike is 2σ below spot, so the downside buffer is intact; unrealized profit (+8% of max loss) confirms the thesis is still valid.
- _bear_call_spread_: Implied vol remains above realized, the short strike is 2σ above spot, and the position shows a modest unrealized gain (+6% of max loss), indicating no breach of its invalidation conditions.
- _long_strangle_: Implied‑vol percentile is only 18%, meaning protection is still cheap, and time to expiry (≥20 d) remains; despite a current loss, the thesis explicitly treats bleeding as normal, so the strategy remains justified.

**Challenger:** MODIFY -- Bull put spread and bear call spread allocations are supported by the evidence (implied vol 0.16 > realized 0.13, short strike buffers of ±2σ, unrealized gains of 8% and 6% of max loss, and no invalidation triggers). The long strangle allocation, however, exceeds its allowed risk exposure: the evidence shows its max_loss_outstanding is $4,000, which corresponds to an allocation fraction of 0.20 (0.20 × $20,000 risk budget = $4,000). Raising it to 0.30 would imply $6,000 risk, violating the strategy's max loss constraint and thus is not justified by the thesis evidence. The extra 0.10 of risk budget should be re‑allocated to strategies whose thesis remains valid.

**Final after gates (challenger_modified):** `   bull  40%     bear  40%     long  20%`

## Strangle in profit, but IV now expensive

The strangle is modestly profitable, but IV has risen to the 86th percentile -- protection is now expensive, so the position is paying a high price for the same convexity going forward.

**Expectation:** Reduce the strangle -- this is its GENUINE invalidation (protection no longer cheap), which is different from cutting it for a loss. The contrast with the bleeding-but-cheap case is the whole point: P&L is not the trigger, the IV percentile is.

**Held before:** `   bull  35%     bear  35%     long  20%`

**Allocator proposed:** `   bull  45%     bear  45%     long   0%`

- _bull_put_spread_: The short put is 1.8σ below spot and implied volatility (0.29) exceeds realized (0.19) by a 1.53 ratio, so the downside insurance remains unlikely to be breached and the volatility risk premium is still present.
- _bear_call_spread_: The short call sits 1.8σ above spot, spot is well below it, and implied volatility remains well above realized, satisfying the upside‑insurance thesis without any invalidating price movement.
- _long_strangle_: Implied‑volatility percentile is 86 % – in the upper range – indicating protection is no longer cheap, which meets the long‑strangle invalidation condition, so the allocation is reduced to zero.

**Challenger:** MODIFY -- The allocator raises the bull‑put spread and bear‑call spread allocations from their stated allocation_frac of 0.35 to 0.45 without any evidence that the thesis has become stronger. The evidence only confirms that the short strikes are still safely distant (short_strike_distance_sigma -1.8 and 1.8) and that implied volatility (0.29) exceeds realized (0.19) (evidence fields "short_strike_distance_sigma" and "iv_rv.ratio"), which supports the existing allocation but does not justify exceeding the baseline allocation_frac. This constitutes performance‑chasing – allocating more simply because the positions are currently unrealized‑profit (unrealized_pnl 350 and 280). The long‑strangle allocation to zero is justified because the implied‑volatility percentile is 86 % (evidence fields "implied_vol_percentile_252d" and "implied_vol_percentile_full"), meeting its invalidation condition. Therefore the allocations should be adjusted back to the evidence‑based fractions.

**Final after gates (challenger_modified):** `   bull  35%     bear  35%     long   0%`
