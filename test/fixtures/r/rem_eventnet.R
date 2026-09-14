# Golden fixture: REM.jl's eventnet-style statistics against R's conditional
# logistic regression (survival::clogit).
#
# rem_clogit.R pins the COUNT statistics (Repetition, Reciprocity, degrees and
# the unweighted TransitiveClosure). This fixture pins what makes REM.jl a port
# of eventnet (Lerner & Lomi): the WEIGHTED triadic statistics with eventnet's
# aggregation of the two dyad weights of a two-path (min by default; max, sum,
# product) summed over the parallel two-paths, the undirected repetition, the
# halflife-DECAYED counts, and the sliding-WINDOW counts. Every column below is
# rebuilt from the raw edgelist in plain R — nothing is imported from Julia —
# so a change in what a statistic means moves the coefficients and fails the
# test.
#
# Definitions (all "history strictly before the focal event"; w(i,j) is the
# count of past i -> j events, decayed or windowed where the model says so):
#   undirected_repetition   w(i,j) + w(j,i)
#   transitive_closure      sum_k agg(w(i,k), w(k,j))  over k != i,j with BOTH > 0
#   cyclic_closure          sum_k agg(w(j,k), w(k,i))
#   shared_sender           sum_k agg(w(k,i), w(k,j))
#   shared_receiver         sum_k agg(w(i,k), w(j,k))
#   decayed w(i,j)          sum_{e: i->j} exp(-lambda (t_m - t_e)),  lambda = log(2)/halflife
#   windowed w(i,j)         #{e: i->j, t_m - t_e <= window}
# A two-path exists only when both of its dyads have a positive weight: for
# `sum` and `max` this matters (a missing leg contributes nothing, not the
# other leg's weight); for `min` and `product` it is automatic.
#
# The same simulated sequence as rem_clogit.R (same seed, same generator), so
# the two fixtures pin two sets of statistics on one dataset. The risk set is
# enumerated in FULL (all n(n-1) ordered dyads per event) on both sides.
#
# Regenerate from the package root:
#
#   Rscript test/fixtures/r/rem_eventnet.R > test/fixtures/rem_eventnet.toml

suppressMessages({
  .libPaths(c(path.expand("~/R/library"), .libPaths()))
  library(survival)
})

seed <- 20260714
set.seed(seed)

n <- 10L   # actors
M <- 80L   # events

# --- the sequence: identical to rem_clogit.R ---------------------------------
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

halflife <- 0.5
lambda <- log(2) / halflife
window <- 0.5

# eventnet's combining functions
agg_fun <- list(min = function(a, b) pmin(a, b), max = function(a, b) pmax(a, b),
                sum = function(a, b) a + b, product = function(a, b) a * b)

# sum over third parties k != i, j of agg(w1[k], w2[k]) restricted to k where
# both legs of the two-path exist
two_path <- function(w1, w2, i, j, agg) {
  others <- setdiff(1:n, c(i, j))
  a <- w1[others]; b <- w2[others]
  ok <- a > 0 & b > 0
  sum(ok * agg_fun[[agg]](a, b))
}

# Build the full-risk-set stratified design from a per-event weight matrix
# provider `weights_at(m)` returning the n x n matrix W of "past weight" for
# the state before event m. Returns a data.frame with the requested columns.
build_design <- function(weights_at, columns) {
  rows <- vector("list", M * n * (n - 1))
  k <- 0L
  for (m in 1:M) {
    W <- weights_at(m)
    s0 <- ev[m, 2]; r0 <- ev[m, 3]
    for (i in 1:n) for (j in 1:n) if (i != j) {
      k <- k + 1L
      rows[[k]] <- c(m, as.integer(i == s0 && j == r0), columns(W, i, j))
    }
  }
  df <- as.data.frame(do.call(rbind, rows))
  df
}

# Weight providers ------------------------------------------------------------
# (a) plain counts
count_before <- function(m) {
  W <- matrix(0, n, n)
  if (m > 1) for (e in 1:(m - 1)) W[ev[e, 2], ev[e, 3]] <- W[ev[e, 2], ev[e, 3]] + 1
  W
}
# (b) halflife-decayed counts, read at the focal event's time
decayed_before <- function(m) {
  W <- matrix(0, n, n)
  if (m > 1) for (e in 1:(m - 1))
    W[ev[e, 2], ev[e, 3]] <- W[ev[e, 2], ev[e, 3]] + exp(-lambda * (ev[m, 1] - ev[e, 1]))
  W
}
# (c) windowed counts: events at most `window` old still count
windowed_before <- function(m) {
  W <- matrix(0, n, n)
  if (m > 1) for (e in 1:(m - 1)) if (ev[m, 1] - ev[e, 1] <= window)
    W[ev[e, 2], ev[e, 3]] <- W[ev[e, 2], ev[e, 3]] + 1
  W
}

fit_model <- function(df, names) {
  names(df) <- c("stratum", "is_event", names)
  f <- as.formula(paste("is_event ~", paste(names, collapse = " + "), "+ strata(stratum)"))
  clogit(f, data = df,
         control = coxph.control(eps = 1e-11, toler.chol = 1e-14, iter.max = 200))
}

# --- Model A: eventnet's min-weighted triadic family + undirected repetition ---
nm_A <- c("undirected_repetition", "transitive_closure", "cyclic_closure",
          "shared_sender", "shared_receiver")
df_A <- build_design(count_before, function(W, i, j) c(
  W[i, j] + W[j, i],
  two_path(W[i, ], W[, j], i, j, "min"),
  two_path(W[j, ], W[, i], i, j, "min"),
  two_path(W[, i], W[, j], i, j, "min"),
  two_path(W[i, ], W[j, ], i, j, "min")))
fit_A <- fit_model(df_A, nm_A)

# --- Models agg_*: repetition + transitive closure under each aggregation ------
fits_agg <- list()
for (agg in names(agg_fun)) {
  df <- build_design(count_before, function(W, i, j) c(
    W[i, j], two_path(W[i, ], W[, j], i, j, agg)))
  fits_agg[[agg]] <- fit_model(df, c("repetition", "transitive_closure"))
}

# --- Model B: halflife-decayed repetition, activity, min-weighted closure ------
nm_B <- c("repetition", "sender_activity", "transitive_closure")
df_B <- build_design(decayed_before, function(W, i, j) c(
  W[i, j], sum(W[i, ]), two_path(W[i, ], W[, j], i, j, "min")))
fit_B <- fit_model(df_B, nm_B)

# --- Model C: windowed repetition, reciprocity, min-weighted closure -----------
nm_C <- c("repetition", "reciprocity", "transitive_closure")
df_C <- build_design(windowed_before, function(W, i, j) c(
  W[i, j], W[j, i], two_path(W[i, ], W[, j], i, j, "min")))
fit_C <- fit_model(df_C, nm_C)

num <- function(x) paste(sprintf("%.17g", x), collapse = ", ")
emit <- function(prefix, fit) {
  cat(sprintf("%s_names = [%s]\n", prefix,
              paste(sprintf('"%s"', names(coef(fit))), collapse = ", ")))
  cat(sprintf("%s_coefficients = [%s]\n", prefix, num(coef(fit))))
  cat(sprintf("%s_std_errors = [%s]\n", prefix, num(sqrt(diag(vcov(fit))))))
  cat(sprintf("%s_loglik = %.17g\n", prefix, fit$loglik[2]))
  cat(sprintf("%s_loglik_null = %.17g\n", prefix, fit$loglik[1]))
}

cat('name = "rem_eventnet"\n\n')

cat("[provenance]\n")
cat(sprintf('r_version = "%s"\n', as.character(getRversion())))
cat(sprintf('survival_version = "%s"\n', as.character(packageVersion("survival"))))
cat(sprintf("seed = %d\n", seed))
cat('script = "test/fixtures/r/rem_eventnet.R"\n')
cat(sprintf('date = "%s"\n', format(Sys.Date())))
cat('dataset = "the rem_clogit.R sequence (10 actors, 80 events; same seed and generator); frozen below under input_*"\n')
cat('design = "full risk set: all n(n-1) = 90 ordered dyads per event, one stratum per event; statistics rebuilt in plain R"\n')
cat('definitions = "eventnet: sum over third parties of agg(two-path weights), both legs > 0; decay exp(-log(2)/halflife * elapsed); window = events at most `window` old"\n\n')

cat("[tolerance]\n")
cat("# As in rem_clogit.toml: both sides maximize the SAME exact conditional-\n")
cat("# logit likelihood on the SAME design by Newton-Raphson — no Monte Carlo,\n")
cat("# no sampling — so only floating-point summation order may differ. The\n")
cat("# decayed columns are sums of exponentials, which adds rounding at the\n")
cat("# 1e-15 level, nothing more. 1e-8 keeps five orders of margin over that\n")
cat("# and still fails on any change in a statistic's definition.\n")
cat("default = 1e-8\n")
cat("input_time = 1e-12\n\n")

cat("[values]\n")
cat("# --- inputs (echoed so the Julia test fits the identical data) ---\n")
cat(sprintf("n_actors = %d\n", n))
cat(sprintf("input_time = [%s]\n", num(ev[, 1])))
cat(sprintf("input_sender = [%s]\n", paste(as.integer(ev[, 2]), collapse = ", ")))
cat(sprintf("input_receiver = [%s]\n", paste(as.integer(ev[, 3]), collapse = ", ")))
cat(sprintf("n_strata = %d\n", M))
cat(sprintf("risk_set_size = %d\n", n * (n - 1L)))
cat("\n# --- Model A: min-weighted triadic family + undirected repetition ---\n")
emit("eventnet", fit_A)
for (agg in names(agg_fun)) {
  cat(sprintf("\n# --- repetition + transitive closure with aggregation = :%s ---\n", agg))
  emit(paste0("agg_", agg), fits_agg[[agg]])
}
cat("\n# --- Model B: halflife-decayed counts ---\n")
cat(sprintf("decay_halflife = %.17g\n", halflife))
emit("decay", fit_B)
cat("\n# --- Model C: sliding-window counts ---\n")
cat(sprintf("window_length = %.17g\n", window))
emit("window", fit_C)
