# Graduate Campaign — Screening Policy

Version 2. Issued by the Head of Talent.

Clauses formalised into the logic database carry a machine-readable annotation,
so each extracted rule cites the clause it encodes and a reviewer can check one
against the other.

## s1.0 Scope

Every clause below applies to an applicant in this campaign. Rules state that
scope explicitly with `applicant(A)`, because a condition expressed only as a
series of negations ("does not evidence X, and X is not missing") describes no
one until the population it ranges over is named.

## s1.1 Essential requirements

An application may be rejected at screening only where an essential requirement
is **positively established as unmet** by the evidence supplied. Absence of
evidence is not evidence of absence, and is dealt with under s2.1.

<!-- rule: requirements_unmet(A) <- applicant(A), not meets_experience(A), not missing_experience_evidence(A). | p=0.88 -->
<!-- rule: requirements_unmet(A) <- applicant(A), not has_core_skill(A), not skill_evidence_absent(A). | p=0.85 -->

## s1.2 Right to work

An applicant who cannot evidence the right to work cannot be progressed. This is
a legal bar rather than a judgement of merit.

<!-- rule: requirements_unmet(A) <- applicant(A), not right_to_work(A). | p=0.95 -->

## s2.1 Incomplete evidence

Where an essential requirement is neither established nor refuted, or where the
CV timeline conflicts with a screening answer, the application is **incomplete**
and must be seen by a person. It must not be rejected on the missing element.

<!-- rule: evidence_incomplete(A) <- skill_evidence_absent(A). | p=0.92 -->
<!-- rule: evidence_incomplete(A) <- missing_experience_evidence(A). | p=0.92 -->
<!-- rule: evidence_incomplete(A) <- timeline_conflict(A). | p=0.90 -->

## s3.1 Rejection

An application may be rejected where an essential requirement is unmet **and**
the evidence is not incomplete.

<!-- rule: reject(A) <- requirements_unmet(A), not evidence_incomplete(A). | p=0.87 -->

## s3.2 Advancing

An application that meets the essential requirements, with complete evidence,
advances to human shortlisting. Advancing is a recommendation; only a recruiter
records a hire.

<!-- rule: advance(A) <- applied_in_window(A), not requirements_unmet(A), not evidence_incomplete(A). | p=0.84 -->

## s4.1 Protected characteristics

No screening decision shall rest on age, sex, ethnicity, or any attribute
serving as a proxy for them. This clause is deliberately **not** formalised as a
rule, and could not be: a rule can state a condition, but not the absence of a
hidden correlation.

It is given effect structurally instead. None of these attributes is a declared
predicate in this domain, so no rule can be written over one and none can be
mined over one. The constraint is enforced by the vocabulary rather than by
review, because review cannot reliably detect a proxy.

## s4.2 Reasons and contest

A rejected applicant is entitled to the reason for the decision. The proof
recorded against each screening decision is the record from which that reason is
given, and it names the requirement that failed and the evidence relied on.
