"""Read only locally acquired, frozen P2-P5 windows; no ZIP/downloader execution.

One person/record is loaded at a time. SHDB is an int16 memory-mapped stream.
Pickles are executable formats: only the already acquired trusted PKLs may be read.
Independent references are offline Window fields, never RuntimeBlock fields.
"""
from pathlib import Path
import csv
import json
import pickle
import numpy as np
import pandas as pd
import wfdb
from .protocol import FrozenDesign, PACKAGE, sha256
from .schema import Signal, Window, activity_mode, pure_wesad_label, dalia_reference, check_dalia_count

def truth(value): return str(value).lower()=='true'

def empatica_start(value):
    try: return float(value)
    except ValueError:
        stamp=pd.Timestamp(value)
        return float((stamp.tz_localize('UTC') if stamp.tzinfo is None else stamp.tz_convert('UTC')).timestamp())

def read_empatica(path, expected=None):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.reader(f); first=next(reader); second=next(reader)
    start=[empatica_start(x) for x in first]; rates=[float(x) for x in second]
    if len(set(start))!=1 or len(set(rates))!=1: raise ValueError('Inconsistent channel headers')
    a=pd.read_csv(path,skiprows=2,header=None).to_numpy(dtype=float)
    if expected:
        if rates[0]!=expected['fs'] or start[0]!=expected['start'] or len(a)!=expected['original_n']: raise ValueError('Frozen native CSV schema changed')
        # P2 official S02 duplicate suffix only; no real-time gap stitching.
        a=a[:expected['n']]
    return a,start[0],rates[0]

class DatasetAdapters:
    def __init__(self,data_root,indices=None,design=None):
        self.root=Path(data_root).resolve()
        self.indices=Path(indices or PACKAGE.parents[1]/'03_Data'/'qualified_indices').resolve()
        self.design=design or FrozenDesign()
        self._tables={}
        manifest=json.loads((self.indices/'manifest.json').read_text(encoding='utf-8'))
        for row in manifest['files']:
            if row['stage']=='P6': continue
            p=self.indices/row['stage']/Path(row['file']).name
            if sha256(p)!=row['sha256']: raise ValueError('Qualified index changed: '+str(p))

    def table(self,stage,name):
        key=(stage,name)
        if key not in self._tables:
            p=self.indices/stage/name
            self._tables[key]=json.loads(p.read_text(encoding='utf-8-sig')) if p.suffix=='.json' else pd.read_csv(p,dtype={'record':str,'caseid':str,'subjectid':str,'source_subject':str})
        return self._tables[key]

    def people(self,source,role=None): return self.design.select(source,role)

    def iter_person(self,person):
        p=self.design.people[person]; source=p['source']
        dispatch={'VITALDB':self._vital,'SHDB-AF':self._shdb,'PPG-DaLiA':self._pickle,'WESAD':self._pickle,'PPGSPORT':self._sport,'GALAXYPPG':self._galaxy,'Stress-Predict':self._stress,'WearableDevice_V1':self._stress,'WearableDevice_V2':self._stress}
        yield from dispatch[source](p)

    def _window(self,p,record,start,end,signals,**kw):
        return Window(p['subject'],p['source'],str(record),p['role'],float(start),float(end),signals,**kw).validate()

    def _vital(self,p):
        table=self.table('P3','window_eligibility.csv')
        rows=table[(table.subjectid==p['source_subject']) & table.technical_qualified.astype(str).str.lower().eq('true')]
        for case, group in rows.groupby('caseid',sort=True):
            if self.design.record_subject('VITALDB',case)!=p['subject']: raise ValueError('VitalDB role conflict')
            with np.load(self.root/'vitaldb_af_afl_p3_v1'/'prepared_native_windows'/(case+'.npz'),allow_pickle=False) as z:
                if str(z['subjectid'].item())!=p['source_subject'] or int(z['label'].item())!=p['label']: raise ValueError('VitalDB identity/label changed')
                starts=z['starts']; lookup={float(v):i for i,v in enumerate(starts)}
                if len(lookup)!=len(group) or set(lookup)!=set(group.start_seconds): raise ValueError('Prepared window index mismatch')
                for r in group.itertuples():
                    j=lookup[r.start_seconds]
                    signals={'ECG':Signal(z['ecg'][j],1/float(z['ecg_dt'].item()),'source_unit'),'BVP':Signal(z['ppg'][j],1/float(z['ppg_dt'].item()),'arbitrary_unit')}
                    yield self._window(p,case,r.start_seconds,r.end_seconds,signals,label=int(r.label))

    def _pickle(self,p):
        sid=p['source_subject']; dalia=p['source']=='PPG-DaLiA'
        path=self.root/('ppg+dalia/data/PPG_FieldStudy' if dalia else 'WESAD/WESAD')/sid/(sid+'.pkl')
        with path.open('rb') as f: item=pickle.load(f,encoding='latin1')
        if str(item['subject'])!=sid: raise ValueError('PKL identity mismatch')
        wrist=item['signal']['wrist']; bvp=np.asarray(wrist['BVP']); acc=np.asarray(wrist['ACC'])
        labels=np.asarray(item['label']).reshape(-1)
        duration=min(len(bvp)/64,len(acc)/32)
        if dalia:
            activity=np.asarray(item['activity']); duration=min(duration,len(activity)/4)
            check_dalia_count(len(labels),duration)
        else: duration=min(duration,len(wrist['EDA'])/4,len(labels)/700)
        for k in range(int(duration//10)):
            s=k*10; e=s+10
            label=None if dalia else pure_wesad_label(labels[s*700:e*700])
            if not dalia and label is None: continue
            signals={'BVP':Signal(bvp[s*64:e*64],64,'arbitrary_unit'),'ACC':Signal(acc[s*32:e*32],32,'1/64g')}
            if not dalia: signals['EDA']=Signal(np.asarray(wrist['EDA'])[s*4:e*4],4,'microSiemens')
            valid=all(np.isfinite(x.values).all() for x in signals.values()) and np.ptp(signals['BVP'].values)>0
            yield self._window(p,sid,s,e,signals,label=label,reference_hr=dalia_reference(labels,s,e) if dalia else None,activity=activity_mode(activity[s*4:e*4]) if dalia else int(labels[s*700]),technical_valid=bool(valid),metadata={'reference_wholly_inside':dalia_reference(labels,s,e,True)} if dalia else {})

    def _stress(self,p):
        sp=p['source']=='Stress-Predict'
        meta=self.table('P2','stress_predict_time_alignment.json' if sp else 'backup_segments_and_modalities.json')
        table=self.table('P2','stress_predict_eligible_windows.csv' if sp else 'backup_eligible_windows.csv')
        sid=p['source_subject']; old_id=('stress_predict:' if sp else 'wearable_device:')+sid
        rows=table[table.subject==old_id]
        if rows.empty: raise ValueError('No frozen stress windows')
        for record,group in rows.groupby('record',sort=True):
            entry=next(x for x in meta if x.get('record',x['subject'])==record)
            directory=self.root/('additional_cohorts/stress_predict/extracted/Stress-Predict-Dataset-main/Raw_data' if sp else 'additional_cohorts/wearable_device_stress_exercise_1_0_1/extracted/wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.1/Wearable_Dataset/STRESS')/record
            arrays={key:read_empatica(directory/(key+'.csv'),entry['modalities'][key]) for key in ['BVP','ACC','EDA']}
            for r in group.to_dict('records'):
                signals={}
                for key,(a,start,fs) in arrays.items():
                    lo=int(r[key+'_sample_start']);hi=int(r[key+'_sample_stop'])
                    # Frozen P2 timestamp arithmetic, not new alignment optimization.
                    if abs((start+lo/fs)-r['start_utc_s'])>1/fs+1e-6 or hi-lo!=int(10*fs): raise ValueError('Stress time support mismatch')
                    signals[key]=Signal(a[lo:hi],fs,'microSiemens' if key=='EDA' else '1/64g' if key=='ACC' else 'arbitrary_unit')
                yield self._window(p,record,r['start_utc_s'],r['end_utc_s'],signals,label=int(r['label']),activity=r['task'])

    def _sport(self,p):
        table=self.table('P4','qualified_windows.csv');rows=table[(table.subject==p['subject']) & table.included_in_primary_cohort.astype(str).str.lower().eq('true') & table.eligible.astype(str).str.lower().eq('true')]
        for activity, group in rows.groupby('activity',sort=True):
            cache={}
            for r in group.itertuples():
                signals={}
                for key,rel in [('BVP',r.ppg_path),('ACC',r.acc_path)]:
                    path=self.root/'ppg_sport_p4_candidate'/'source_repository'/Path(rel)
                    if str(path) not in cache: cache[str(path)]=np.load(path,mmap_mode='r',allow_pickle=False)
                    signals[key]=Signal(cache[str(path)][int(r.start_sample):int(r.stop_sample)],64,'source_unit' if key=='ACC' else 'arbitrary_unit')
                yield self._window(p,activity,r.start_s,r.end_s,signals,reference_hr=float(r.reference_hr_bpm),activity=activity,wrist=str(r.hand))

    def _galaxy(self,p):
        table=self.table('P4','galaxy_qualified_windows.csv');rows=table[(table.subject==p['subject']) & table.included_in_primary_cohort.astype(str).str.lower().eq('true') & table.eligible.astype(str).str.lower().eq('true')]
        directory=self.root/'galaxyppg_p4_v1'/'Dataset'/p['source_subject']
        frames={key:pd.read_csv(directory/'E4'/(key+'.csv')) for key in ['BVP','ACC']}
        for key,f in frames.items():
            if np.any(np.diff(f.timestamp.to_numpy())<=0): raise ValueError('Native wrist clock not increasing')
        for r in rows.itertuples():
            signals={}
            for key,f in frames.items():
                t=f.timestamp.to_numpy()/1e6
                lo=np.searchsorted(t,r.start_utc,'left');hi=np.searchsorted(t,r.end_utc,'left')
                columns=['value'] if key=='BVP' else ['x','y','z']
                signals[key]=Signal(f.iloc[lo:hi][columns].to_numpy(dtype=float),64 if key=='BVP' else 32,'arbitrary_unit' if key=='BVP' else '1/64g')
            yield self._window(p,p['source_subject'],r.start_utc,r.end_utc,signals,reference_hr=float(r.reference_hr_bpm),activity=r.activity,metadata={'reference_kind':'PolarH10 device HR; P4 fixed clock correction'})

    def _shdb(self,p):
        table=self.table('P5','candidate_windows.csv')
        for record in sorted(p['records']):
            self.design.record_subject('SHDB-AF',record)
            directory=self.root/'shdb_af_p5_v1_0_1'; header=wfdb.rdheader(str(directory/record))
            if header.fs!=200 or header.sig_name[0]!='ECG1' or header.fmt!=['16','16'] or header.units[0]!='mV': raise ValueError('SHDB native schema changed')
            digital=np.memmap(directory/(record+'.dat'),mode='r',dtype='<i2',shape=(header.sig_len,header.n_sig))
            rows=table[(table.subject==p['subject']) & (table.record==record)]
            for r in rows.itertuples():
                raw=digital[int(r.start_sample):int(r.end_sample),0]
                physical=(raw.astype(float)-header.baseline[0])/header.adc_gain[0]
                missing=raw==-32768;physical[missing]=np.nan
                valid=not missing.any() and np.isfinite(physical).all() and np.ptp(physical)>0
                if bool(valid)!=truth(r.signal_valid): raise ValueError('P5 signal qualification disagrees')
                yield self._window(p,record,r.start_seconds,r.end_seconds,{'ECG':Signal(physical,200,'mV')},label=int(r.target_label),technical_valid=bool(valid),metadata={'eligible_pure_window':truth(r.eligible_pure_window)})

    def monitor_reference(self,person):
        p=self.design.people[person]
        if p['source']!='SHDB-AF': raise ValueError('Not monitor source')
        intervals=self.table('P5','rhythm_intervals.csv');events=self.table('P5','target_episodes.csv')
        return intervals[intervals.subject==person].copy(),events[(events.subject==person)&events.eligible.astype(str).str.lower().eq('true')].copy()
