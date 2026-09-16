# Hardship Grant Scheme — Operating Policy

Version 3. Issued by the Welfare Directorate.

This document is the written authority for hardship grant decisions. Clauses
that have been formalised into the logic database carry a machine-readable
annotation beside them, so that every extracted rule cites the clause it came
from and a reviewer can check one against the other.

Note that the annotations are how the *scripted* agent reads this document. A
language-model agent reads the prose instead, and its proposals must still cite
a clause — but a citation it invents is only as trustworthy as the model, which
is why extracted rules go through the same approval path as mined ones.

## s2.1 Residency condition

An applicant must meet the residency condition to receive any award under this
scheme. Residency is a necessary condition and is never waived, including in
cases of exceptional hardship.

## s3.1 Core hardship route

An applicant whose household income falls below the published threshold, and who
meets the residency condition, is eligible for an award, provided they do not
hold savings above the capital limit.

<!-- rule: eligible(A) <- low_income(A), resident(A), not savings_over_limit(A). | p=0.88 -->

## s3.2 Dependants route

Where an applicant has dependants and is in rent arrears, and meets the
residency condition, an award may be made irrespective of the income threshold.
This route exists to reach households whose income is nominally above the
threshold but whose committed outgoings leave them in hardship.

<!-- rule: eligible(A) <- has_dependants(A), arrears(A), resident(A). | p=0.85 -->

## s3.3 Disability premium route

An applicant in receipt of a disability premium who meets the residency
condition is eligible, provided they do not hold savings above the capital
limit. Officers should note that the capital limit applies to this route as it
does to s3.1.

<!-- rule: eligible(A) <- disability_premium(A), resident(A), not savings_over_limit(A). | p=0.86 -->

## s4.1 Repeat awards

An applicant who has received an award within the preceding twelve months
should be referred to an officer rather than determined automatically. This is
a referral, not a refusal: the officer may still make an award.

<!-- rule: refer_to_officer(A) <- recent_award(A). | p=0.95 -->

## s4.2 Capital limit

Savings above the capital limit bar an award under s3.1 and s3.3. The limit does
not bar an award under s3.2, where committed outgoings rather than capital are
the operative test.

## s5.1 Appeals

An applicant may appeal any determination within 28 days. Where the right of
appeal has been waived or has lapsed, the determination cannot be corrected on
review, and the case must therefore be considered by an officer before it is
determined.
