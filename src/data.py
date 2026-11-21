"""Read packed L2 values and compare only aligned, quality-screened pixels."""

import numpy as np


EXCLUDE_FLAGS = ("ATMFAIL", "LAND", "HIGLINT", "HILT", "STRAYLIGHT", "CLDICE",
                 "CHLFAIL", "NAVFAIL", "PRODFAIL", "HISATZEN", "HISOLZEN",
                 "LOWLW", "MAXAERITER", "ATMWARN", "NAVWARN", "CHLWARN",
                 "SEAICE", "BOWTIEDEL")


def read_valid(variable, selection=...):
    raw = np.asarray(variable[selection])
    valid = np.isfinite(raw)
    for key, compare in (("_FillValue", np.not_equal), ("valid_min", np.greater_equal),
                         ("valid_max", np.less_equal)):
        if key in variable.attrs:
            valid &= compare(raw, variable.attrs[key])
    value = raw.astype(np.float64) * variable.attrs.get("scale_factor", 1)
    value += variable.attrs.get("add_offset", 0)
    return np.where(valid, value, np.nan)


def quality_mask(variable, selection=..., exclude_flags=EXCLUDE_FLAGS):
    meanings = variable.attrs["flag_meanings"].split()
    masks = np.asarray(variable.attrs["flag_masks"]).astype(np.uint32)
    missing = set(exclude_flags) - set(meanings)
    if missing:
        raise ValueError(f"Missing quality flag definitions: {sorted(missing)}")
    excluded = np.bitwise_or.reduce([m for n, m in zip(meanings, masks) if n in exclude_flags],
                                    initial=np.uint32(0))
    return (np.asarray(variable[selection]).astype(np.uint32) & excluded) == 0


def correlation(x, y):
    valid = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x, y = x[valid], np.log10(y[valid])
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"n": len(x), "pearson_r_log10_target": None}
    return {"n": len(x), "pearson_r_log10_target": float(np.corrcoef(x, y)[0, 1])}


def check_pair(aop, bgc, selection):
    for key in ("time_coverage_start", "time_coverage_end", "processing_version", "platform", "instrument"):
        if key not in aop.attrs or aop.attrs[key] != bgc.attrs.get(key):
            raise ValueError(f"AOP/BGC mismatch: {key}")
    for name in ("latitude", "longitude"):
        a = aop.groups["navigation_data"].variables[name]
        b = bgc.groups["navigation_data"].variables[name]
        if a.shape != b.shape or not np.allclose(read_valid(a, selection), read_valid(b, selection),
                                                rtol=0, atol=1e-5, equal_nan=True):
            raise ValueError(f"AOP/BGC mismatch: {name}")
