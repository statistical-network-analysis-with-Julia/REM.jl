# Golden fixture: REM.jl's relational-event coefficients against R's
# conditional logistic regression (survival::clogit).
#
# REM.jl fits a stratified conditional logit — one stratum per event, the case
# being the dyad that actually acted — by its own Newton-Raphson. That is
# exactly the likelihood `survival::clogit` maximizes (a Cox model with exact
# ties on a 1:k matched design), so R is a real, independent check on BOTH
# halves of the pipeline:
#
#   * the ESTIMATOR: same likelihood, same data, must give the same MLE; and
#   * the STATISTICS: the design matrix below is recomputed from the raw
#     edgelist in plain R, not exported from Julia. If REM's Repetition /
#     Reciprocity / SenderActivity / ReceiverPopularity / TransitiveClosure
#     mean something other than what their names say, the coefficients move
#     and this fixture goes red.
#
# The risk set is enumerated in FULL (all n*(n-1) ordered dyads per event) on
# both sides. That makes the design deterministic — no sampled controls to
# reconcile between two RNGs — so the comparison is exact rather than
# distributional. (REM's case-control SAMPLING is a variance/compute tradeoff
# on top of this likelihood; it is not what R would disagree with.)
#
# Regenerate from the package root:
#
#   Rscript test/fixtures/r/rem_clogit.R > test/fixtures/rem_clogit.toml

suppressMessages({
  .libPaths(c(path.expand("~/R/library"), .libPaths()))
  library(survival)
})

seed <- 20260714
set.seed(seed)

n <- 10L   # actors
M <- 80L   # events

# Simulate a sequence with repetition, reciprocation and sender-activity
# reinforcement, so each fitted effect is identified. n(n-1) = 90 dyads vs 80
# events keeps the "ever interacted" graph sparse — otherwise TransitiveClosure
# saturates at n-2 for every dyad and stops being identified at all.
ev <- matrix(0, M, 3)
tt <- 0
cnt <- matrix(0, n, n)
for (m in 1:M) {
  lam <- matrix(0, n, n)
  for (i in 1:n) for (j in 1:n) if (i != j)
    lam[i, j] <- exp(-1 + 0.7 * log1p(cnt[i, j]) + 0.5 * log1p(cnt[j, i]) +
                     0.25 * log1p(sum(cnt[i, ])))
  tot <- sum(lam)
  tt <- tt + rexp(1, tot)
  k <- sample(seq_len(n * n), 1, prob = as.vector(lam) / tot)
  i <- ((k - 1) %% n) + 1L
  j <- ((k - 1) %/% n) + 1L
  ev[m, ] <- c(tt, i, j)
  cnt[i, j] <- cnt[i, j] + 1
}

# --- Build the stratified full-risk-set design in plain R -------------------
# Statistics, as REM.jl defines them (decay = 0, event weights 1):
#   repetition          count of past events sender -> receiver
#   reciprocity         count of past events receiver -> sender
#   sender_activity     sender's out-degree (weighted count of events sent)
#   receiver_popularity receiver's in-degree (count of events received)
#   transitive_closure  |{k != i,j : i ever->k and k ever->j}|  (adjacency, not counts)
# History is strictly pre-event: the focal event updates the state only after
# its own stratum has been written.
cnt <- matrix(0, n, n)
adj <- matrix(FALSE, n, n)
rows <- vector("list", M * n * (n - 1))
k <- 0L
for (m in 1:M) {
  s0 <- ev[m, 2]
  r0 <- ev[m, 3]
  outdeg <- rowSums(cnt)
  indeg <- colSums(cnt)
  for (i in 1:n) for (j in 1:n) if (i != j) {
    others <- setdiff(1:n, c(i, j))
    tc <- sum(adj[i, others] & adj[others, j])
    k <- k + 1L
    rows[[k]] <- c(m, as.integer(i == s0 && j == r0),
                   cnt[i, j], cnt[j, i], outdeg[i], indeg[j], tc)
  }
  cnt[s0, r0] <- cnt[s0, r0] + 1
  adj[s0, r0] <- TRUE
}
df <- as.data.frame(do.call(rbind, rows))
names(df) <- c("stratum", "is_event", "repetition", "reciprocity",
               "sender_activity", "receiver_popularity", "transitive_closure")

fit <- clogit(is_event ~ repetition + reciprocity + sender_activity +
                receiver_popularity + transitive_closure + strata(stratum),
              data = df,
              control = coxph.control(eps = 1e-11, toler.chol = 1e-14,
                                      iter.max = 200))

se <- sqrt(diag(vcov(fit)))
num <- function(x) paste(sprintf("%.17g", x), collapse = ", ")

cat('name = "rem_clogit"\n\n')

cat("[provenance]\n")
cat(sprintf('r_version = "%s"\n', as.character(getRversion())))
cat(sprintf('survival_version = "%s"\n', as.character(packageVersion("survival"))))
cat(sprintf("seed = %d\n", seed))
cat('script = "test/fixtures/r/rem_clogit.R"\n')
cat(sprintf('date = "%s"\n', format(Sys.Date())))
cat('dataset = "simulated relational event sequence (10 actors, 80 events); frozen below under input_*"\n')
cat('design = "full risk set: all n(n-1) = 90 ordered dyads per event, one stratum per event"\n\n')

cat("[tolerance]\n")
cat("# Both sides maximize the SAME exact conditional-logit likelihood on the\n")
cat("# SAME design by Newton-Raphson: there is no Monte Carlo and no sampling,\n")
cat("# so nothing here is allowed to differ except floating-point summation\n")
cat("# order. Observed disagreement at the frozen values is < 1e-13 on every\n")
cat("# coefficient and standard error. 1e-8 is that with five orders of\n")
cat("# margin, and is still tight enough that any real change in a statistic's\n")
cat("# definition, or in the estimator, fails the test.\n")
cat("default = 1e-8\n")
cat("# The inputs are echoed exactly; they are read, not compared.\n")
cat("input_time = 1e-12\n\n")

cat("[values]\n")
cat("# --- inputs (echoed so the Julia test fits the identical data) ---\n")
cat(sprintf("n_actors = %d\n", n))
cat(sprintf("input_time = [%s]\n", num(ev[, 1])))
cat(sprintf("input_sender = [%s]\n", paste(as.integer(ev[, 2]), collapse = ", ")))
cat(sprintf("input_receiver = [%s]\n", paste(as.integer(ev[, 3]), collapse = ", ")))
cat("\n# --- survival::clogit on the full-risk-set stratified design ---\n")
cat(sprintf("statistic_names = [%s]\n",
            paste(sprintf('"%s"', names(coef(fit))), collapse = ", ")))
cat(sprintf("coefficients = [%s]\n", num(coef(fit))))
cat(sprintf("std_errors = [%s]\n", num(se)))
cat(sprintf("loglik = %.17g\n", fit$loglik[2]))
cat(sprintf("loglik_null = %.17g\n", fit$loglik[1]))
cat(sprintf("n_strata = %d\n", M))
cat(sprintf("risk_set_size = %d\n", n * (n - 1L)))
