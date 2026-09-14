# Golden fixture: REM.jl against R's relevent::rem.dyad on the BUNDLED dataset.
#
# The rem_clogit / rem_eventnet fixtures pin REM.jl against survival::clogit on
# a simulated sequence. This one pins it against the R package whose model it
# shares the estimand with — relevent's ORDINAL dyadic relational event model
# (Butts 2008) — on the dataset relevent's own tutorial uses: the World Trade
# Center police radio calls (481 events among 37 actors, Butts, Petrescu-Prahova
# & Cross 2007), which Networks.jl bundles as `load_dataset(:wtc_police_calls)`.
#
# R reads the SAME two TSV files Networks.jl reads (../Networks.jl/data/), so
# the 481 events and the 37-actor universe are provably the same on both
# sides; nothing is regenerated or re-keyed. (relevent 1.2.x no longer ships
# the WTCPoliceCalls object; when it is present it is checked for equality.)
#
# Two models, both from the relevent tutorial's ICR ("institutionalised
# coordinator role") covariate:
#
#   1. effects = "CovInt"  — the tutorial's first model (wtcfit1). relevent's
#      CovInt is "covariate effect for both outgoing and incoming actions":
#      x_i + x_j, which is REM.jl's NodeSum(icr). (It is NOT x_i * x_j, and it
#      is collinear with CovSnd + CovRec — fitting the three together is a
#      singular Hessian in R.)
#   2. effects = CovSnd + CovRec — the sender effect x_i and the receiver
#      effect x_j of ICR: REM.jl's SenderAttribute and ReceiverAttribute.
#      (The ICR-to-ICR product, CovEvent(outer(x, x)) / NodeProduct, is NOT
#      fitted: no ICR actor ever calls another in these data, so that
#      coefficient separates to -Inf and no optimizer can pin it.)
#   3. effects = CovSnd + CovRec + CovEvent(D), with D a fixed dyadic
#      covariate D[i, j] = ((7 i + 3 j) mod 5) / 4 — arbitrary, deterministic,
#      non-degenerate — which pins REM.jl's DyadCovariate against relevent's
#      dyad-level covariate effect.
#
# With ordinal = TRUE, rem.dyad's likelihood is exactly the conditional-logit
# partial likelihood over the FULL risk set (all n(n-1) ordered dyads), which
# REM.jl computes when n_controls = n(n-1) - 1 (every non-case dyad is
# enumerated, nothing is sampled). Both sides maximize the same function.
#
# Regenerate from the package root:
#
#   Rscript test/fixtures/r/rem_relevent_wtc.R > test/fixtures/rem_relevent_wtc.toml

suppressMessages({
  .libPaths(c(path.expand("~/R/library"), .libPaths()))
  library(relevent)
})

seed <- 20260909   # no randomness in the fit; recorded for the provenance contract
set.seed(seed)

# Locate the bundled data relative to this script (test/fixtures/r/ -> the
# ecosystem root holds Networks.jl beside REM.jl)
args <- commandArgs(trailingOnly = FALSE)
script <- sub("^--file=", "", grep("^--file=", args, value = TRUE))
script_dir <- if (length(script) == 1) dirname(normalizePath(script)) else getwd()
data_dir <- normalizePath(file.path(script_dir, "..", "..", "..", "..", "Networks.jl", "data"),
                          mustWork = TRUE)

events <- read.delim(file.path(data_dir, "wtc_police_calls_events.tsv"))
actors <- read.delim(file.path(data_dir, "wtc_police_calls_actors.tsv"))
stopifnot(identical(names(events), c("number", "source", "recipient")),
          identical(names(actors), c("id", "is_icr")),
          nrow(events) == 481L, nrow(actors) == 37L,
          all(actors$id == seq_len(37)))
n <- nrow(actors)
M <- nrow(events)
is_icr <- as.numeric(actors$is_icr)
el <- as.matrix(events)   # columns: time (event number), sender, receiver

# The tutorial object, when relevent still ships it, must be the same data
if (exists("WTCPoliceCalls")) {
  stopifnot(all(as.matrix(WTCPoliceCalls) == el), all(WTCPoliceIsICR == is_icr))
}

# optim's defaults stop short of the MLE; tighten so the comparison measures
# agreement, not R's termination slack
ctl <- list(reltol = 1e-15, maxit = 10000)

fit1 <- rem.dyad(el, n = n, effects = "CovInt", covar = list(CovInt = is_icr),
                 ordinal = TRUE, hessian = TRUE, fit.method = "MLE",
                 verbose = FALSE, gof = FALSE, optim.control = ctl)

fit2 <- rem.dyad(el, n = n, effects = c("CovSnd", "CovRec"),
                 covar = list(CovSnd = is_icr, CovRec = is_icr),
                 ordinal = TRUE, hessian = TRUE, fit.method = "MLE",
                 verbose = FALSE, gof = FALSE, optim.control = ctl)

D <- outer(seq_len(n), seq_len(n), function(i, j) ((7 * i + 3 * j) %% 5) / 4)
fit3 <- rem.dyad(el, n = n, effects = c("CovSnd", "CovRec", "CovEvent"),
                 covar = list(CovSnd = is_icr, CovRec = is_icr, CovEvent = D),
                 ordinal = TRUE, hessian = TRUE, fit.method = "MLE",
                 verbose = FALSE, gof = FALSE, optim.control = ctl)

num <- function(x) paste(sprintf("%.17g", x), collapse = ", ")
emit <- function(prefix, fit) {
  cat(sprintf("%s_coefficients = [%s]\n", prefix, num(fit$coef)))
  cat(sprintf("%s_std_errors = [%s]\n", prefix, num(sqrt(diag(fit$cov)))))
  cat(sprintf("%s_loglik = %.17g\n", prefix, -fit$residual.deviance / 2))
  cat(sprintf("%s_loglik_null = %.17g\n", prefix, -fit$null.deviance / 2))
}

cat('name = "rem_relevent_wtc"\n\n')

cat("[provenance]\n")
cat(sprintf('r_version = "%s"\n', as.character(getRversion())))
cat(sprintf('relevent_version = "%s"\n', as.character(packageVersion("relevent"))))
cat(sprintf("seed = %d\n", seed))
cat('script = "test/fixtures/r/rem_relevent_wtc.R"\n')
cat(sprintf('date = "%s"\n', format(Sys.Date())))
cat('dataset = "WTC police radio calls (Butts, Petrescu-Prahova & Cross 2007): Networks.jl/data/wtc_police_calls_{events,actors}.tsv, read by R and by Networks.load_dataset(:wtc_police_calls)"\n')
cat('design = "relevent::rem.dyad(ordinal = TRUE, hessian = TRUE): full risk set of all 37*36 ordered dyads per event"\n')
cat('fit_method = "MLE (optim BFGS, reltol = 1e-15), observed-information SEs from the numerical Hessian"\n\n')

cat("[tolerance]\n")
cat("# The ordinal likelihood is EXACT (a multinomial partial likelihood over\n")
cat("# the full risk set): no Monte Carlo, no sampling, so both implementations\n")
cat("# maximize the same function and the only admissible discrepancy is\n")
cat("# optimizer termination slack — R uses optim/BFGS (reltol = 1e-15) with a\n")
cat("# finite-difference Hessian, Julia Newton-Raphson with the analytic one.\n")
cat("# Observed disagreement at the frozen values is < 2e-7 on coefficients and\n")
cat("# standard errors, and the log-likelihoods agree to < 1e-9 (both sit on\n")
cat("# the same maximum). 1e-6 is that with an order of magnitude of margin,\n")
cat("# as Relevent.jl's rem.dyad fixture justifies it.\n")
cat("default = 1e-6\n")
cat("# The log-likelihood is compared more tightly: it is the value of the\n")
cat("# function both optimizers agree they maximized.\n")
cat("model1_loglik = 1e-8\n")
cat("model2_loglik = 1e-8\n")
cat("model3_loglik = 1e-8\n\n")

cat("[values]\n")
cat(sprintf("n_actors = %d\n", n))
cat(sprintf("n_events = %d\n", M))
cat(sprintf("icr_actors = [%s]\n", paste(which(is_icr == 1), collapse = ", ")))
cat(sprintf("risk_set_size = %d\n", n * (n - 1L)))
cat("\n# --- model 1: CovInt (x_i + x_j) — relevent tutorial wtcfit1; REM.jl NodeSum(icr) ---\n")
cat('model1_names = ["sum_icr"]\n')
emit("model1", fit1)
cat("\n# --- model 2: CovSnd + CovRec (x_i, x_j); REM.jl SenderAttribute(icr), ReceiverAttribute(icr) ---\n")
cat('model2_names = ["sender_icr", "receiver_icr"]\n')
emit("model2", fit2)
cat("\n# --- model 3: CovSnd + CovRec + CovEvent(D), D[i,j] = ((7i + 3j) mod 5)/4; REM.jl + DyadCovariate ---\n")
cat('model3_names = ["sender_icr", "receiver_icr", "dyad_covariate"]\n')
emit("model3", fit3)
