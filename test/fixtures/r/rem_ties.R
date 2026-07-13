# Golden fixture: REM.jl's TIE CORRECTIONS against R's Cox partial likelihood
# (survival::coxph with ties = "breslow" and ties = "efron").
#
# REM.jl's conditional-logit partial likelihood IS a Cox partial likelihood with
# one stratum per event, so tied event times are the classical Cox tie problem
# and Breslow/Efron are the classical answers to it. This fixture is what turns
# "we implemented Breslow and Efron" into "we implemented Breslow and Efron AND
# they are the same numbers R computes".
#
# The design is the counting-process form of exactly the model REM fits:
#
#   * one interval (m-1, m] per DISTINCT event time m (a "tie block");
#   * every one of the n(n-1) ordered dyads is at risk in every interval — the
#     full risk set, so there are no sampled controls to reconcile across two
#     RNGs and the comparison is exact rather than distributional;
#   * the covariates of interval m are the network statistics as they stand
#     BEFORE any event of block m — frozen across the block, which is precisely
#     what the tie corrections mean: simultaneous events cannot have influenced
#     one another;
#   * the d events of block m are d deaths sharing that one risk set.
#
# coxph() with ties="breslow"/"efron" then maximizes the Breslow/Efron partial
# likelihood of that risk set, which is what `fit_rem(...; ties=:breslow/:efron)`
# maximizes. Statistics (repetition, reciprocity, sender_activity) are rebuilt
# from the raw edgelist in plain R — nothing is imported from Julia — so this
# checks the STATISTICS and the TIE CORRECTION and the ESTIMATOR at once.
#
# Ties are made by observing a continuous-time process on a COARSE CLOCK (times
# rounded to a grid), which is what ties in real event data almost always are: a
# measurement artifact of the recording resolution. Simultaneous repeats of the
# SAME dyad are resampled away, so that every dyad appears at most once per
# block and the risk set of a block contains each dyad exactly once (REM's
# risk set is a set of dyads; a duplicated case dyad would enter R's
# counting-process risk set twice and the two designs would stop being the same
# design — that is a property of the fixture, not a limitation of either side).
#
# Regenerate from the package root:
#
#   Rscript test/fixtures/r/rem_ties.R > test/fixtures/rem_ties.toml

suppressMessages({
  .libPaths(c(path.expand("~/R/library"), .libPaths()))
  library(survival)
})

seed <- 20260713
set.seed(seed)

n <- 8L     # actors
M <- 90L    # events
grid <- 0.03 # clock resolution: event times are rounded to multiples of this

# --- simulate a relational event sequence -----------------------------------
# Rates reward repetition, reciprocation and sender activity, so each effect is
# identified. Times are drawn in continuous time and then coarsened onto the
# grid, which is what creates the ties.
ev <- matrix(0, M, 3)
tt <- 0
cnt <- matrix(0, n, n)
m <- 1L
while (m <= M) {
  lam <- matrix(0, n, n)
  for (i in 1:n) for (j in 1:n) if (i != j)
    lam[i, j] <- exp(-1 + 0.6 * log1p(cnt[i, j]) + 0.4 * log1p(cnt[j, i]) +
                     0.2 * log1p(sum(cnt[i, ])))
  tot <- sum(lam)
  tt <- tt + rexp(1, tot)
  k <- sample(seq_len(n * n), 1, prob = as.vector(lam) / tot)
  i <- ((k - 1) %% n) + 1L
  j <- ((k - 1) %/% n) + 1L

  t_coarse <- round(tt / grid) * grid

  # A dyad may not act twice within one tie block (see the header).
  if (m > 1L && any(ev[1:(m - 1L), 1] == t_coarse &
                    ev[1:(m - 1L), 2] == i & ev[1:(m - 1L), 3] == j)) next

  ev[m, ] <- c(t_coarse, i, j)
  cnt[i, j] <- cnt[i, j] + 1
  m <- m + 1L
}
ord <- order(ev[, 1])   # stable: keeps the simulation order within a tie block
ev <- ev[ord, , drop = FALSE]

times <- ev[, 1]
blocks <- unique(times)
B <- length(blocks)
d_of_block <- as.integer(table(factor(times, levels = blocks)))
stopifnot(any(d_of_block > 1))   # the fixture is pointless without ties

# --- build the counting-process full-risk-set design in plain R -------------
# Statistics, as REM.jl defines them (decay = 0, event weights 1), computed from
# the state BEFORE the block (frozen across the tied events):
#   repetition      count of past events sender -> receiver
#   reciprocity     count of past events receiver -> sender
#   sender_activity sender's out-degree (count of events sent)
cnt <- matrix(0, n, n)
rows <- vector("list", B * n * (n - 1))
k <- 0L
for (b in seq_len(B)) {
  tb <- blocks[b]
  in_block <- which(times == tb)
  outdeg <- rowSums(cnt)
  for (i in 1:n) for (j in 1:n) if (i != j) {
    is_ev <- any(ev[in_block, 2] == i & ev[in_block, 3] == j)
    k <- k + 1L
    rows[[k]] <- c(b - 1L, b, as.integer(is_ev), cnt[i, j], cnt[j, i], outdeg[i])
  }
  # the whole block is absorbed at once — no event of a block may enter the
  # statistics of another event of the same block
  for (e in in_block) cnt[ev[e, 2], ev[e, 3]] <- cnt[ev[e, 2], ev[e, 3]] + 1
}
df <- as.data.frame(do.call(rbind, rows))
names(df) <- c("start", "stop", "is_event", "repetition", "reciprocity",
               "sender_activity")

ctrl <- coxph.control(eps = 1e-11, toler.chol = 1e-14, iter.max = 200)
f <- function(method)
  coxph(Surv(start, stop, is_event) ~ repetition + reciprocity + sender_activity,
        data = df, ties = method, control = ctrl)

fit_b <- f("breslow")
fit_e <- f("efron")

num <- function(x) paste(sprintf("%.17g", x), collapse = ", ")

cat('name = "rem_ties"\n\n')

cat("[provenance]\n")
cat(sprintf('r_version = "%s"\n', as.character(getRversion())))
cat(sprintf('survival_version = "%s"\n', as.character(packageVersion("survival"))))
cat(sprintf("seed = %d\n", seed))
cat('script = "test/fixtures/r/rem_ties.R"\n')
cat(sprintf('date = "%s"\n', format(Sys.Date())))
cat(sprintf('dataset = "simulated relational event sequence (%d actors, %d events) observed on a coarse clock (resolution %g), which is what makes the ties; frozen below under input_*"\n',
            n, M, grid))
cat('design = "counting-process Cox: one interval per distinct event time, all n(n-1) ordered dyads at risk in every interval, covariates frozen across each tie block"\n')
cat('model = "coxph(Surv(start, stop, is_event) ~ repetition + reciprocity + sender_activity, ties = breslow | efron)"\n\n')

cat("[tolerance]\n")
cat("# Both sides maximize the SAME Breslow (resp. Efron) partial likelihood on\n")
cat("# the SAME full risk set by Newton-Raphson: no Monte Carlo, no sampling,\n")
cat("# nothing may differ but floating-point summation order. Observed\n")
cat("# disagreement at the frozen values is < 1e-11 on every coefficient,\n")
cat("# standard error and log-likelihood.\n")
cat("default = 1e-8\n")
cat("# The inputs are echoed exactly; they are read, not compared.\n")
cat("input_time = 1e-12\n\n")

cat("[values]\n")
cat("# --- inputs (echoed so the Julia test fits the identical data) ---\n")
cat(sprintf("n_actors = %d\n", n))
cat(sprintf("input_time = [%s]\n", num(times)))
cat(sprintf("input_sender = [%s]\n", paste(as.integer(ev[, 2]), collapse = ", ")))
cat(sprintf("input_receiver = [%s]\n", paste(as.integer(ev[, 3]), collapse = ", ")))
cat(sprintf("n_events = %d\n", M))
cat(sprintf("n_blocks = %d\n", B))
cat(sprintf("n_tied_blocks = %d\n", sum(d_of_block > 1)))
cat(sprintf("max_block_size = %d\n", max(d_of_block)))
cat(sprintf("risk_set_size = %d\n", n * (n - 1L)))
cat(sprintf("statistic_names = [%s]\n",
            paste(sprintf('"%s"', names(coef(fit_b))), collapse = ", ")))

cat("\n# --- survival::coxph, ties = \"breslow\" ---\n")
cat(sprintf("breslow_coefficients = [%s]\n", num(coef(fit_b))))
cat(sprintf("breslow_std_errors = [%s]\n", num(sqrt(diag(vcov(fit_b))))))
cat(sprintf("breslow_loglik = %.17g\n", fit_b$loglik[2]))

cat("\n# --- survival::coxph, ties = \"efron\" (R's own default) ---\n")
cat(sprintf("efron_coefficients = [%s]\n", num(coef(fit_e))))
cat(sprintf("efron_std_errors = [%s]\n", num(sqrt(diag(vcov(fit_e))))))
cat(sprintf("efron_loglik = %.17g\n", fit_e$loglik[2]))
