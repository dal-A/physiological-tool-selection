"""Frozen identities and role enforcement. Never reconstruct a split from outcomes."""
from pathlib import Path
import hashlib
import json

PACKAGE = Path(__file__).resolve().parents[1]
PROTOCOL_SHA = '9b333fcfb75748291d90730b3fd55b166ecff54c0521e7270478200244987f58'
ALLOCATION_SHA = '3c289513aee99340abdb3c960cb1bd99382b7b3cd1f4981183dc97d7039c64e4'
ROLES = ('train', 'policy', 'conformal', 'evaluation', 'supplementary_evaluation')

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): h.update(block)
    return h.hexdigest()

class FrozenDesign:
    def __init__(self, config=None):
        self.config = Path(config or PACKAGE/'config')
        for name, expected in [('protocol.json', PROTOCOL_SHA), ('person_allocations.json', ALLOCATION_SHA)]:
            if sha256(self.config/name) != expected: raise ValueError('Frozen input changed: '+name)
        self.protocol = json.loads((self.config/'protocol.json').read_text(encoding='utf-8-sig'))
        self.tasks = json.loads((self.config/'person_allocations.json').read_text(encoding='utf-8-sig'))['tasks']
        self.people = {}
        for task, rows in self.tasks.items():
            for row in rows:
                if row['subject'] in self.people: raise ValueError('Duplicate namespaced identity')
                if row['role'] not in ROLES: raise ValueError('Unknown role')
                self.people[row['subject']] = dict(row, task=task)
        self.records = {}
        for person, row in self.people.items():
            for record in row.get('records', []):
                key = (row['source'], str(record))
                if key in self.records and self.records[key] != person: raise ValueError('Record identity conflict')
                self.records[key] = person

    def require(self, subjects, role):
        subjects = set(subjects)
        if not subjects: raise ValueError('Empty fit population')
        wrong = [s for s in subjects if s not in self.people or self.people[s]['role'] != role]
        if wrong: raise ValueError('Role leakage: '+repr(wrong[:5]))

    def select(self, source, role=None):
        return sorted((x for x in self.people.values() if x['source'] == source and (role is None or x['role']==role)), key=lambda x: x.get('allocation_rank', hashlib.sha256(('20260921|'+x['subject']).encode()).hexdigest()))

    def record_subject(self, source, record):
        if (source,str(record)) not in self.records: raise ValueError('Excluded/unqualified record')
        return self.records[(source,str(record))]

    def fold(self, subject):
        self.require([subject], 'train')
        return self.people[subject]['training_cv_fold']
