"""Frozen P6 person-cluster inference. No window-level independent resampling."""
import numpy as np
from scipy.stats import binomtest, beta, norm
from sklearn.metrics import roc_auc_score

SEED = 20260921
REPLICATES = 10000


def cluster_draws(n, labels=None, replicates=REPLICATES, seed=SEED):
    if n < 1:
        raise ValueError('Independent people required')
    rng = np.random.default_rng(seed)
    if labels is None:
        return rng.integers(n, size=(replicates, n)), None
    y = np.asarray(labels, dtype=int)
    if len(y) != n or set(y) != {0, 1}:
        raise ValueError('Both independent-person strata required')
    parts = [rng.choice(np.flatnonzero(y == k), size=(replicates, int(sum(y == k)))) for k in (0, 1)]
    weights = np.concatenate([np.full(part.shape[1], .5 / part.shape[1]) for part in parts])
    return np.concatenate(parts, axis=1), weights


def mean_defined(values, labels=None):
    a = np.asarray(values, dtype=float)
    good = np.isfinite(a)
    if labels is None:
        return float(a[good].mean()) if good.any() else None
    y = np.asarray(labels)
    pieces = [a[good & (y == k)] for k in (0, 1)]
    return float(.5 * (pieces[0].mean() + pieces[1].mean())) if all(len(p) for p in pieces) else None


def bootstrap_mean(values, labels=None, replicates=REPLICATES, seed=SEED):
    a = np.asarray(values, dtype=float)
    draws, weights = cluster_draws(len(a), labels, replicates, seed)
    sample = a[draws]
    valid = np.isfinite(sample)
    if weights is None:
        den = valid.sum(axis=1)
        vals = np.divide(np.where(valid, sample, 0).sum(axis=1), den,
                         out=np.full(replicates, np.nan), where=den > 0)
    else:
        y = np.asarray(labels)[draws]
        pieces = []
        for k in (0, 1):
            use = valid & (y == k)
            den = use.sum(axis=1)
            pieces.append(np.divide(np.where(use, sample, 0).sum(axis=1), den,
                                     out=np.full(replicates, np.nan), where=den > 0))
        vals = .5 * (pieces[0] + pieces[1])
    finite = vals[np.isfinite(vals)]
    enough = len(finite) >= .95 * replicates
    return dict(estimate=mean_defined(a, labels),
                CI95_low=float(np.quantile(finite, .025, method='linear')) if enough else None,
                CI95_high=float(np.quantile(finite, .975, method='linear')) if enough else None,
                independent_people=len(a), contributing_people=int(np.isfinite(a).sum()),
                valid_replicates=len(finite), requested_replicates=replicates, seed=seed,
                undefined_reason=None if enough else 'less_than_95_percent_valid_cluster_replicates')


def paired_contrast(left, right, labels=None, replicates=REPLICATES):
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Paired primary endpoints need identical complete person support')
    d = a - b
    result = bootstrap_mean(d, labels, replicates)
    nonzero = d[d != 0]
    result.update(improved_people=int(sum(d < 0)), worsened_people=int(sum(d > 0)), ties=int(sum(d == 0)),
                  sign_effective_people=len(nonzero),
                  sign_p_raw=float(binomtest(int(sum(nonzero < 0)), len(nonzero), .5).pvalue) if len(nonzero) else 1.,
                  sign_test_null='probability of negative nonzero person difference equals 0.5; not a mean-difference null')
    return result, d


def holm_family(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid p values')
    order = np.argsort(p, kind='stable')
    corrected = np.minimum(1., np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    result = np.empty(len(p)); result[order] = corrected
    return result


def person_event_interval(events):
    a = np.asarray(events, dtype=int)
    if not len(a) or not set(a).issubset({0, 1}):
        raise ValueError('Independent-person binary indicators required')
    n, k = len(a), int(a.sum()); p = k / n; z = norm.ppf(.975)
    centre = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return dict(independent_people=n, people_with_at_least_one_event=k, proportion=p,
                Wilson95_low=float(centre-half), Wilson95_high=float(centre+half),
                exact95_low=float(beta.ppf(.025, k, n-k+1)) if k else 0.,
                exact95_high=float(beta.ppf(.975, k+1, n-k)) if k < n else 1.,
                exact_one_sided95_upper=float(beta.ppf(.95, k+1, n-k)) if k < n else 1.,
                zero_observed_events_not_zero_risk_proof=True)


def bootstrap_patient_auc(labels, scores, replicates=REPLICATES):
    y, p = np.asarray(labels), np.asarray(scores, dtype=float)
    good = np.isfinite(p)
    if set(y[good]) != {0, 1}:
        return dict(estimate=None, CI95_low=None, CI95_high=None, independent_people=len(y),
                    contributing_people=int(good.sum()), valid_replicates=0, requested_replicates=replicates,
                    undefined_reason='insufficient_person_probability_support')
    draws, _ = cluster_draws(len(y), y, replicates)
    negative, positive = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    counts = np.zeros((replicates, len(y)), dtype=float)
    for j in range(draws.shape[1]):
        np.add.at(counts, (np.arange(replicates), draws[:, j]), 1)
    counts[:, ~good] = 0
    comparison = ((p[positive, None] > p[None, negative]).astype(float)
                  + .5*(p[positive, None] == p[None, negative]))
    comparison[~np.isfinite(comparison)] = 0
    den = counts[:, positive].sum(axis=1)*counts[:, negative].sum(axis=1)
    numerator = np.einsum('bi,ij,bj->b', counts[:, positive], comparison, counts[:, negative], optimize=True)
    vals = (numerator[den > 0]/den[den > 0]).tolist()
    enough = len(vals) >= .95 * replicates
    return dict(estimate=float(roc_auc_score(y[good], p[good])),
                CI95_low=float(np.quantile(vals, .025)) if enough else None,
                CI95_high=float(np.quantile(vals, .975)) if enough else None,
                independent_people=len(y), contributing_people=int(good.sum()), valid_replicates=len(vals),
                requested_replicates=replicates, undefined_reason=None if enough else 'less_than_95_percent_valid_cluster_replicates')
