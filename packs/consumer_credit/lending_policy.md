# Consumer Lending Policy

Version 11. Issued by the Credit Committee.

Clauses formalised into the logic database carry a machine-readable annotation.
Every extracted rule cites the clause it came from, so a reviewer can check the
rule against the policy rather than against their recollection of it.

Clauses without an annotation are deliberately left unformalised — either because
they require judgement the system should not be making, or because they are
about process rather than determination. Clause s8.1 is the clearest example.

## s4.1 Current account conduct

An applicant whose current account is overdrawn, and who holds little or no
savings, presents elevated risk.

<!-- rule: elevated_risk(A) <- overdrawn(A), thin_savings(A). | p=0.82 -->

## s4.2 Adverse history

An adverse repayment record is on its own sufficient to treat an applicant as
presenting elevated risk, irrespective of current account conduct.

<!-- rule: elevated_risk(A) <- adverse_history(A). | p=0.75 -->

## s5.1 Exposure and term

A large advance over a long term, where the applicant is not in established
employment, presents elevated risk. Neither the amount nor the term is
determinative on its own.

<!-- rule: elevated_risk(A) <- large_amount(A), long_term(A), not employed_long(A). | p=0.79 -->

## s5.2 Instalment burden

Where a high share of disposable income would be committed and the applicant
already holds instalment plans elsewhere, risk is elevated.

<!-- rule: elevated_risk(A) <- high_instalment_burden(A), other_plans(A). | p=0.72 -->

## s6.1 Security and guarantees

A guarantor or co-applicant is treated as mitigating security.

<!-- rule: mitigated(A) <- guarantor(A). | p=0.90 -->

## s6.2 Property and employment

Ownership of real estate, together with established employment of four years or
more, is treated as mitigating security.

<!-- rule: mitigated(A) <- owns_property(A), employed_long(A). | p=0.78 -->

## s7.1 Determination

An application presenting elevated risk and without mitigating security shall be
declined.

<!-- rule: decline(A) <- elevated_risk(A), not mitigated(A). | p=0.85 -->

## s7.2 Approval

An application with mitigating security and no elevated risk shall be approved.

<!-- rule: approve(A) <- mitigated(A), not elevated_risk(A). | p=0.84 -->

## s8.1 Protected characteristics

No determination shall rest on the applicant's sex, marital status, or on any
attribute serving as a proxy for either. This clause is not formalised as a rule
and could not be: a rule can state a condition, but not the absence of a hidden
correlation.

It is given effect structurally instead. Sex is not a declared predicate in this
domain, so no rule can be written over it and no rule can be mined over it. The
constraint is enforced by the vocabulary rather than by review, because review
cannot reliably detect a proxy and should not be relied on to.

## s8.2 Reasons

An applicant who is declined is entitled to the reasons for the determination.
The proof recorded against each decision is the record from which those reasons
are given.
