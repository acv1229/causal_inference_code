"""
Formal Statement: Anti-Conservatism of R_D Under Propensity Misspecification

This file contains the formal proposition and proof for the paper, plus
a definition of "anti-conservative diagnostic." The empirical verification
is in decomposition_analysis.py.

================================================================================

DEFINITION (Anti-Conservative Diagnostic).

A diagnostic T for a problem P is anti-conservative if T increases (signals
safety) as P worsens, under the convention that higher T corresponds to a
more favorable assessment.

In our setting: higher R_D is conventionally interpreted as "more residual
treatment variation → better overlap → safer inference." If R_D increases
with propensity misspecification while bias simultaneously increases, R_D
is anti-conservative — it reassures the practitioner precisely when
inference is least reliable.

================================================================================

PROPOSITION 1 (Variance Decomposition of R_D).

Let:
    m_0(X) = E[D | X]                    (true propensity)
    m_hat(X)                              (estimated propensity, possibly misspecified)
    V = D - m_0(X)                        (true treatment residual)
    delta(X) = m_0(X) - m_hat(X)         (misspecification error)
    D_tilde = D - m_hat(X) = V + delta(X) (observed treatment residual)

    R_D = Var(D_tilde) / Var(D)           (the diagnostic)
    R_D* = Var(V) / Var(D)                (the oracle diagnostic under correct specification)

Then:

    R_D = R_D* + Var(delta) / Var(D)

and R_D is strictly increasing in Var(delta), the misspecification magnitude.

PROOF.

Since V = D - m_0(X) with E[V | X] = 0, and delta(X) is sigma(X)-measurable:

    Cov(V, delta(X)) = E[V * delta(X)] - E[V] * E[delta(X)]
                      = E[E[V * delta(X) | X]] - 0
                      = E[delta(X) * E[V | X]]
                      = E[delta(X) * 0]
                      = 0

Therefore:

    Var(D_tilde) = Var(V + delta(X))
                 = Var(V) + Var(delta(X)) + 2 * Cov(V, delta(X))
                 = Var(V) + Var(delta(X))

Dividing by Var(D):

    R_D = Var(V)/Var(D) + Var(delta(X))/Var(D) = R_D* + Var(delta)/Var(D)

Since Var(delta)/Var(D) >= 0, we have R_D >= R_D* with equality iff
delta(X) is constant a.s. (i.e., m_hat differs from m_0 by at most a constant).
R_D is strictly increasing in Var(delta).                                   QED

================================================================================

PROPOSITION 2 (Anti-Conservatism).

Under the conditions of Proposition 1, suppose additionally that:

(i)  The outcome model g_hat(X) is imperfect: g_0(X) - g_hat(X) is not
     identically zero.

(ii) The misspecification error delta(X) is correlated with the outcome
     model error: E[delta(X) * (g_0(X) - g_hat(X))] != 0.

Then the DML bias is approximately:

    Bias(theta_hat) ≈ E[delta(X) * (g_0(X) - g_hat(X))] / E[D_tilde^2]

(This is the standard DML bias decomposition from Chernozhukov et al. 2018,
arising from the failure of Neyman orthogonality when nuisance functions
are misspecified.)

Under (ii), both |Bias| and R_D increase with Var(delta):

    - R_D increases by Proposition 1 (the inflation term Var(delta)/Var(D))
    - |Bias| increases because the numerator E[delta * (g_0 - g_hat)]
      scales with the magnitude of delta (by Cauchy-Schwarz:
      |E[delta * eta]| <= sqrt(Var(delta)) * sqrt(Var(eta)),
      and when delta and eta are structurally correlated, the bound tightens)

Therefore R_D is anti-conservative: it signals safety (higher R_D)
precisely when bias is increasing.

REMARK 1 (Double Robustness).

When the outcome model is correctly specified (g_hat = g_0), the bias
numerator is zero regardless of delta. This explains why XGBoost on
the linear outcome surface maintains 94% coverage even at R_D = 0.15:
the outcome model compensates for propensity misspecification. The
anti-conservative property requires BOTH nuisance functions to be
imperfect — propensity misspecification alone is not sufficient.

REMARK 2 (Why Flexible Learners Produce Conservative R_D).

When m_hat ≈ m_0 (as with XGBoost on the structural propensity),
Var(delta) ≈ 0 and R_D ≈ R_D*. The diagnostic correctly reflects
the true overlap. Low R_D from a flexible learner is a genuine
signal of weak overlap, not an artifact of misspecification.

This creates an asymmetry:
    - Low R_D from a flexible learner: genuine warning (conservative)
    - High R_D from a rigid learner: possibly false comfort (anti-conservative)

REMARK 3 (Practitioner Implication).

A practitioner who computes R_D from a single learner cannot distinguish
between "R_D is high because overlap is good" and "R_D is high because
my propensity model missed confounding structure." Comparing R_D across
a rigid and a flexible learner resolves this ambiguity: if they diverge,
the rigid learner's R_D is inflated by Var(delta)/Var(D).

================================================================================
"""

# No computational code — this file is the formal write-up.
# Empirical verification: see decomposition_analysis.py
# Reversal plots: see plot_reversal.py
