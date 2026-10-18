"""Policy-only CPU profiling with actual preprocessing/calls inside callbacks."""
import hashlib
import platform
from time import perf_counter_ns
import numpy as np

def profile_tool(windows,callback,design,warmup=20,repeats=200):
    windows=list(windows);design.require([w.subject for w in windows],'policy')
    if warmup<0 or repeats<1:raise ValueError('Invalid profiling repetitions')
    groups={}
    for w in windows:
        key=(w.subject,w.record,w.start,w.end,w.wrist)
        h=hashlib.sha256(('20260921|profile|'+repr(key)).encode()).hexdigest()
        groups.setdefault(w.subject,[]).append((h,w))
    selected=[w for s in sorted(groups) for _,w in sorted(groups[s],key=lambda x:x[0])[:20]]
    timings=[];ids=[]
    for k in range(warmup+repeats):
        w=selected[k%len(selected)];t0=perf_counter_ns();callback(w.runtime());dt=perf_counter_ns()-t0
        if k>=warmup:
            timings.append(dt/1e6);ids.append([w.subject,w.record,w.start,w.end,w.wrist])
    return {'clock':'perf_counter_ns','warmup':warmup,'timed_repeats':repeats,'selected_ids':ids,'CPU_milliseconds':timings,'p95_CPU_milliseconds':float(np.quantile(timings,.95,method='linear')),'mean_CPU_milliseconds':float(np.mean(timings)),'platform':platform.platform(),'processor':platform.processor(),'physical_acquisition_latency':None,'disk_IO_included':False}

def latency_ratio(primary_profile,secondary_plus_fusion_profile):
    a=primary_profile['p95_CPU_milliseconds'];b=secondary_plus_fusion_profile['p95_CPU_milliseconds']
    if not np.isfinite(a) or not np.isfinite(b) or a<=0 or b<=0:raise ValueError('Positive policy latencies required')
    return float(b/a)
