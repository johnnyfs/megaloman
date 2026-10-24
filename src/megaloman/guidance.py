"""Guidance units: small, general rules with provenance, assembled into step prompts by scope and role.

workspace/guidance/<scope>/units/<id>.md  (front matter: id, scope, applies_to [roles], status, provenance, evidence)
A step opts in with `guidance: {scopes: [...], role: <role>}`; its prompt gets the units at a {guidance}
placeholder (or appended). The unit texts are part of the prompt, so they are part of every cache key.
"""
import os, re

import yaml

from . import layout

ROOT = os.path.join(layout.W, 'guidance')


def load_units(scopes, role):
    units = []
    for scope in scopes:
        d = os.path.join(ROOT, scope, 'units')
        for fn in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if not fn.endswith('.md'):
                continue
            text = open(os.path.join(d, fn)).read()
            m = re.match(r'---\n(.*?)\n---\n(.*)', text, re.S)
            meta, body = yaml.safe_load(m.group(1)), m.group(2).strip()
            if role in meta.get('applies_to', []) and meta.get('status') != 'rejected':
                units.append({**meta, 'text': body, 'file': os.path.relpath(os.path.join(d, fn), layout.W)})
    return units


def render(units):
    if not units:
        return ''
    return 'General guidance:\n' + '\n'.join(f'- {u["text"]}' for u in units)


def apply(prompt, spec):
    """-> (prompt with guidance, [unit ids])"""
    if not spec:
        return prompt.replace('{guidance}', ''), []
    units = load_units(spec.get('scopes', []), spec['role'])
    block = render(units)
    out = prompt.replace('{guidance}', block) if '{guidance}' in prompt else (prompt.rstrip() + '\n\n' + block if block else prompt)
    return out, [u['id'] for u in units]


SMALL_TARGET_PX = 32


def for_concept(prompt, target_px, role='concept_art'):
    """Concept prompts for a sprite of target_px (the smallest rung) get the pixel-art/concept units when the
    target is tiny (<= SMALL_TARGET_PX): fill the square, no thin protrusions, folded limbs, ... -> (prompt, ids)"""
    if target_px > SMALL_TARGET_PX:
        return prompt.replace('{guidance}', ''), []
    return apply(prompt, {'scopes': ['pixel-art/concept'], 'role': role})
