"""
hippocampal_index.py — FHRR Hippocampal Index for codi-memory.

Computational analog of hippocampal indexing (Teyler & DiScenna 1986).
Binary memory that acts as a fast spatial index: "where did we talk about X?"
Complements the existing vector+BM25+semantic pipeline (neocortex).

Architecture:
  Query → [Bloom: exists?] → [Cascade: where?] → session context
  FHRR is ADDITIVE. Never blocks existing pipeline. Graceful degradation.

Consolidates Sprint 1-3 prototype:
  - FHRR core: bundle, bind, similarity, concept_to_fhrr
  - Extraction WITHOUT YAKE: regex + tokenization + ALIAS_MAP (0 new deps)
  - Session index: build, save, load
  - Cascade query + binary_recall with weighted_sum aggregation
  - D=2000, chunk_size=8, weights=(0.40, 0.10, 0.40, 0.10)

Papers:
  - Schlegel et al. 2022 — FHRR wins ALL metrics vs MAP, BSC, HRR
  - Teyler & DiScenna 1986 — Hippocampal Indexing Theory
  - McClelland 1995 — Complementary Learning Systems
  - HippoRAG NeurIPS 2024 — hippocampal memory indexing for RAG
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

from .config import DATA_DIR, FTS_DB_PATH, now_iso, connect_fts

_logger = logging.getLogger("codi-memory.hippocampal")

# ============================================================
# CONSTANTS
# ============================================================

D_DEFAULT = 2000           # FHRR vector dimensionality
CHUNK_SIZE_DEFAULT = 8     # turns per chunk
BLOOM_BITS = 2048          # Bloom filter size
BLOOM_HASHES = 4           # number of hash functions
MAX_VOCAB_DISCOVER = 500   # Guard: avoid OOM in discover_synonym_edges() with large vocabularies
THRESHOLD_DEFAULT = 0.05   # minimum score for a hit
MAX_RESULTS_DEFAULT = 10

# Scoring weights (validated Sprint 2: 213 configs, weighted_sum breakthrough)
W_ROLE_FHRR = 0.40
W_TURN_FHRR = 0.10
W_OVERLAP = 0.40
W_MATCH_BONUS = 0.10
DECAY = 0.7  # weighted_sum session aggregation decay

# Session boost cap (same range as temporal contiguity boost)
SESSION_BOOST_CAP = 0.15

# Storage
FHRR_SESSIONS_DIR = os.path.join(DATA_DIR, "fhrr_sessions")

# Hot index paths (pre-compiled vectorized query)
HOT_INDEX_PATH = os.path.join(DATA_DIR, "fhrr_hot_index.npz")
HOT_META_PATH = os.path.join(DATA_DIR, "fhrr_hot_meta.json")

# Role fields for FHRR encoding
ROLE_FIELDS = [
    ('TOPIC', 'topics'),
    ('ACTION', 'actions'),
    ('FILE', 'files'),
    ('RESULT', 'results'),
    ('ENTITY', 'entities'),
    ('DECISION', 'decisions'),
]

ROLE_NAMES = ['SPEAKER', 'TOPIC', 'ACTION', 'FILE', 'RESULT',
              'ENTITY', 'DECISION', 'SPRINT', 'COMMIT', 'METRIC']


# ============================================================
# FHRR CORE (from fhrr_core.py)
# ============================================================

def _fhrr_bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Associate two concepts. Element-wise multiply in complex space."""
    return a * b


def _fhrr_bundle(*vecs: np.ndarray) -> np.ndarray:
    """Superpose multiple concepts. Sum + normalize to unit circle."""
    s = sum(vecs)
    return (s / np.abs(s)).astype(np.complex64)


def _fhrr_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for FHRR complex vectors. Uses actual D."""
    d = len(a)
    return float(np.abs(np.dot(np.conj(a), b)) / d)


_concept_cache: Dict[tuple, np.ndarray] = {}


def concept_to_fhrr(concept: str, D: int = D_DEFAULT) -> np.ndarray:
    """Deterministic FHRR vector from concept string via hash-seed. Cached."""
    key = _canonicalize(concept)
    cache_key = (key, D)
    cached = _concept_cache.get(cache_key)
    if cached is not None:
        return cached
    seed = int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    angles = rng.uniform(-np.pi, np.pi, D)
    vec = np.exp(1j * angles).astype(np.complex64)
    _concept_cache[cache_key] = vec
    return vec


_role_vectors: Dict[tuple, np.ndarray] = {}

def _get_role_vector(role_name: str, D: int = D_DEFAULT) -> np.ndarray:
    """Deterministic role vectors via hash-seed. Cached."""
    key = (role_name, D)
    if key not in _role_vectors:
        seed = int(hashlib.sha256(f"ROLE:{role_name}".encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        angles = rng.uniform(-np.pi, np.pi, D)
        _role_vectors[key] = np.exp(1j * angles).astype(np.complex64)
    return _role_vectors[key]


# ============================================================
# BLOOM FILTER (from cascade.py)
# ============================================================

class BloomFilter:
    """Probabilistic membership filter. False positives OK, false negatives never."""

    def __init__(self, size_bits: int = BLOOM_BITS, num_hashes: int = BLOOM_HASHES):
        self.size_bits = int(size_bits)
        self.num_hashes = int(num_hashes)
        self.bits = np.zeros(self.size_bits, dtype=np.uint8)

    def _indexes(self, item: str):
        key = item.strip().lower()
        for i in range(self.num_hashes):
            h = int(hashlib.sha256(f"{i}:{key}".encode()).hexdigest(), 16)
            yield h % self.size_bits

    def add(self, item: str) -> None:
        for idx in self._indexes(item):
            self.bits[idx] = 1

    def add_many(self, items: list) -> None:
        for item in items:
            self.add(item)

    def contains(self, item: str) -> bool:
        return all(self.bits[idx] == 1 for idx in self._indexes(item))

    def contains_any(self, items: list) -> bool:
        return any(self.contains(item) for item in items)

    def to_numpy(self) -> np.ndarray:
        return self.bits.copy()

    @classmethod
    def from_numpy(cls, arr: np.ndarray, num_hashes: int = BLOOM_HASHES) -> "BloomFilter":
        bf = cls(size_bits=len(arr), num_hashes=num_hashes)
        bf.bits = arr.astype(np.uint8, copy=True)
        return bf


# ============================================================
# TEXT EXTRACTION — NO YAKE (regex + tokenization + domain terms)
# ============================================================

# --- Alias map (bilingual synonyms) ---
ALIAS_MAP = {
    # Tech
    'container': 'docker', 'containers': 'docker',
    'contenedor': 'docker', 'contenedores': 'docker',
    'k8s': 'kubernetes', 'js': 'javascript', 'ts': 'typescript',
    'py': 'python', 'repo': 'repository', 'repos': 'repository',
    'db': 'database', 'postgres': 'postgresql', 'pg': 'postgresql',
    # Actions (espanol -> ingles)
    'implementar': 'implement', 'implementa': 'implement', 'implementado': 'implement',
    'corregir': 'fix', 'corrige': 'fix', 'arreglar': 'fix', 'arregla': 'fix',
    'probar': 'test', 'prueba': 'test', 'testear': 'test', 'testeo': 'test',
    'debuggear': 'debug', 'depurar': 'debug',
    'buscar': 'search', 'busqueda': 'search',
    'crear': 'create', 'crea': 'create',
    'verificar': 'verify', 'verifica': 'verify',
    'commitear': 'commit', 'commiteo': 'commit',
    'pushear': 'push', 'pusheo': 'push',
    'verifico': 'verify', 'integro': 'implement',
    'reescribo': 'write', 'registro': 'register',
    'agrego': 'add', 'agregar': 'add', 'agregando': 'add', 'agregue': 'add',
    'eliminar': 'delete', 'elimino': 'delete',
    'actualizar': 'update', 'actualizo': 'update',
    'configuro': 'configure', 'configurar': 'configure',
    'desplegar': 'deploy', 'desplegando': 'deploy',
    'mergear': 'merge', 'mergeo': 'merge',
    'parsear': 'parse', 'parseo': 'parse',
    'refactorizar': 'refactor', 'refactorizo': 'refactor',
    # Results
    'funciona': 'success', 'exitoso': 'success', 'exito': 'success',
    'fallo': 'failure', 'falla': 'failure', 'error': 'failure',
    'paso': 'pass', 'pasan': 'pass', 'passed': 'pass',
}

ACTION_WORDS = {
    'implement', 'test', 'commit', 'push', 'debug', 'fix',
    'read', 'verify', 'run', 'integrate', 'create', 'register',
    'update', 'benchmark', 'search', 'review', 'report', 'deploy',
    'install', 'configure', 'refactor', 'merge', 'revert', 'write',
    'delete', 'add', 'remove', 'check', 'validate', 'parse',
    'implementar', 'probar', 'corregir', 'arreglar', 'crear',
    'verificar', 'buscar', 'revisar', 'debuggear', 'commitear',
    'agregar', 'eliminar', 'actualizar', 'configurar', 'desplegar',
    'mergear', 'parsear', 'refactorizar',
}

RESULT_WORDS = {
    'success', 'fail', 'failure', 'error', 'pass', 'passed',
    'failed', 'works', 'broken', 'fixed', 'improved', 'regression',
    'funciona', 'fallo', 'falla', 'roto', 'arreglado', 'mejorado',
}

DECISION_INDICATORS = {
    'decided', 'decision', 'chose', 'choosing', 'porque',
    'elegimos', 'optamos', 'usaremos', 'instead', 'rather',
    'should', 'better', 'prefer', 'descartado', 'descartamos',
}

# Domain-specific terms that should always be recognized as topics
DOMAIN_TERMS = {
    'docker', 'kubernetes', 'deploy', 'deployment', 'repo', 'repository',
    'branch', 'commit', 'push', 'pull', 'merge', 'build', 'release',
    'prod', 'production', 'staging', 'pipeline', 'worker', 'service',
    'api', 'backend', 'frontend', 'database', 'redis', 'postgres',
    'postgresql', 'fasttext', 'glove', 'embedding', 'fhrr', 'yake',
    'cascade', 'bloom', 'chunk', 'session', 'encoding', 'scoring',
    'fts', 'bm25', 'vector', 'query', 'expansion', 'harness', 'eval',
    'mcp', 'consciousness', 'consciencia', 'consolidation', 'prediction',
    'reconsolidation', 'working_memory', 'sleep_loop',
    'fullempaques', 'inventario', 'produccion', 'empaque',
    'recall', 'precision', 'quality', 'latency', 'accuracy', 'coverage',
    'ranking', 'threshold', 'score', 'similarity',
    'telegram', 'n8n', 'kraken', 'trading', 'supabase', 'qdrant',
}

STOP_WORDS = {
    # espanol
    'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'de', 'del',
    'en', 'con', 'por', 'para', 'al', 'es', 'que', 'no', 'si', 'ya',
    'se', 'lo', 'le', 'les', 'me', 'te', 'nos', 'su', 'sus', 'mi',
    'tu', 'a', 'y', 'o', 'e', 'pero', 'como', 'mas', 'muy', 'este',
    'esta', 'estos', 'estas', 'ese', 'esa', 'esos', 'esas', 'ahora',
    'aqui', 'ahi', 'donde', 'cuando', 'hay', 'ser', 'estar', 'tiene',
    'van', 'vamos', 'hacer', 'hoy', 'bien', 'solo', 'todo',
    'otra', 'otro', 'cada', 'entre', 'sobre', 'hasta', 'desde',
    'fue', 'era', 'son', 'estaba', 'tenia', 'tengo', 'hizo', 'hace',
    'dale', 'listo', 'bueno', 'pues', 'entonces', 'claro', 'eso',
    'ahi', 'aca', 'alla', 'asi', 'mucho', 'poco', 'algo', 'nada',
    'siempre', 'nunca', 'tambien', 'quiero', 'puedo', 'creo',
    'mira', 'oye', 'parce', 'hermano', 'parcero', 'llave',
    'corriendo', 'usando', 'haciendo', 'viendo', 'pasando',
    'va', 'iba', 'ir', 'dio', 'dar', 'ver', 'vi', 'vio',
    # ingles
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'shall', 'should', 'may', 'might', 'can', 'could',
    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
    'it', 'its', 'this', 'that', 'these', 'those', 'not', 'or',
    'and', 'but', 'if', 'so', 'we', 'you', 'he', 'she', 'they',
}

_ALL_ACTION_STEMS = ACTION_WORDS | {k for k, v in ALIAS_MAP.items() if v in ACTION_WORDS}
_ALL_RESULT_STEMS = RESULT_WORDS | {k for k, v in ALIAS_MAP.items() if v in RESULT_WORDS}

FILE_PATTERN = re.compile(
    r'[\w/.-]+\.(py|js|ts|tsx|jsx|md|json|yaml|yml|toml|sql|sh|css|html|ipynb)',
    re.IGNORECASE
)
COMMIT_PATTERN = re.compile(r'\b[0-9a-f]{7,40}\b')
METRIC_PATTERN = re.compile(
    r'(\d+\.?\d*)\s*(%|ms|us|MB|KB|GB|seconds?|tests?|passed|failed)',
    re.IGNORECASE
)
SPRINT_PATTERN = re.compile(r'sprint\s*[a-zA-Z0-9]+', re.IGNORECASE)


def _canonicalize(concept: str) -> str:
    """Normalize concept to canonical form."""
    key = concept.lower().strip()
    key = re.sub(r'[-_\s]+', '_', key)
    key = re.sub(r'[^\w.]', '', key)
    key = ALIAS_MAP.get(key, key)
    return key


def _join_spaced_letters(text: str) -> str:
    """Fix spaced-out text: 'b u e n o' -> 'bueno'."""
    if len(text) < 4:
        return text
    # Newline-separated characters
    lines = text.split('\n')
    if len(lines) >= 4:
        single_char_lines = sum(1 for line in lines if len(line.strip()) <= 1)
        if single_char_lines / len(lines) > 0.5:
            chars = []
            for line in lines:
                stripped = line.strip()
                if len(stripped) == 0 or stripped == ' ':
                    chars.append(' ')
                elif len(stripped) == 1:
                    chars.append(stripped)
                else:
                    chars.append(' ' + stripped + ' ')
            return re.sub(r' +', ' ', ''.join(chars)).strip()
    # Space-separated characters
    tokens = text.split(' ')
    if len(tokens) >= 4:
        single_chars = sum(1 for t in tokens if len(t) <= 1 and t.strip())
        if single_chars / len(tokens) > 0.5:
            placeholder = "\x00WORD\x00"
            result = re.sub(r' {2,}', placeholder, text)
            result = result.replace(' ', '')
            result = result.replace(placeholder, ' ')
            return result.strip()
    return text


def _tokenize(text: str) -> list:
    """Extract word tokens from text."""
    return re.findall(r'[a-z\u00e1\u00e9\u00ed\u00f3\u00fa\u00f1\u00fc0-9_.]+', text.lower())


def _scan_actions(text: str) -> list:
    """Scan text for action words directly."""
    found, seen = [], set()
    for tok in _tokenize(text):
        canonical = _canonicalize(tok)
        if canonical in seen:
            continue
        if tok in _ALL_ACTION_STEMS or canonical in ACTION_WORDS:
            mapped = ALIAS_MAP.get(tok, ALIAS_MAP.get(canonical, canonical))
            if mapped not in seen:
                seen.add(mapped)
                found.append(mapped)
    return found


def _scan_results(text: str) -> list:
    """Scan text for result words directly."""
    found, seen = [], set()
    for tok in _tokenize(text):
        canonical = _canonicalize(tok)
        if canonical in seen:
            continue
        if tok in _ALL_RESULT_STEMS or canonical in RESULT_WORDS:
            mapped = ALIAS_MAP.get(tok, ALIAS_MAP.get(canonical, canonical))
            if mapped not in seen:
                seen.add(mapped)
                found.append(mapped)
    return found


def _extract_topics_no_yake(text: str, top_k: int = 8) -> list:
    """Extract topic candidates WITHOUT YAKE — pure tokenization + domain terms + bigrams."""
    tokens = _tokenize(text)
    tokens = [t for t in tokens if len(t) >= 2]

    # Count unigrams and bigrams
    unigram_counts = Counter(tokens)
    bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
    bigram_counts = Counter(bigrams)

    scored = []
    seen_canonical = set()

    # Score unigrams
    for term, count in unigram_counts.items():
        if term in STOP_WORDS:
            continue
        canonical = _canonicalize(term)
        if canonical in seen_canonical:
            continue
        # Skip if it's an action/result (those go to their own roles)
        if canonical in ACTION_WORDS or canonical in RESULT_WORDS:
            continue
        seen_canonical.add(canonical)
        # Score: frequency + domain boost
        score = count + (3.0 if canonical in DOMAIN_TERMS or term in DOMAIN_TERMS else 0.0)
        scored.append((canonical, score))

    # Score bigrams — skip if either part is a stopword
    for bigram, count in bigram_counts.items():
        parts = bigram.split('_')
        if any(p in STOP_WORDS for p in parts):
            continue
        canonical = _canonicalize(bigram)
        if canonical in seen_canonical:
            continue
        # Check if any part is a domain term
        has_domain = any(p in DOMAIN_TERMS or _canonicalize(p) in DOMAIN_TERMS for p in parts)
        score = count + (3.0 if has_domain else 0.0)
        if score >= 1.0:
            seen_canonical.add(canonical)
            scored.append((canonical, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [term for term, _ in scored[:top_k]]


def auto_extract(text: str, speaker: str = "assistant") -> dict:
    """Full extraction pipeline: raw text -> structured roles.

    Replaces YAKE with regex + tokenization + domain terms.
    """
    result = {
        'speaker': speaker,
        'topics': [],
        'actions': [],
        'files': [],
        'results': [],
        'entities': [],
        'decisions': [],
    }

    text = _join_spaced_letters(text)

    # Phase 1: Direct scan for actions/results
    result['actions'] = _scan_actions(text)
    result['results'] = _scan_results(text)

    # Phase 2: Regex extractors
    result['files'] = [m.group() for m in FILE_PATTERN.finditer(text)]

    # Phase 3: Topic extraction (no YAKE)
    topic_candidates = _extract_topics_no_yake(text)

    already_captured = set()
    already_captured.update(_canonicalize(a) for a in result['actions'])
    already_captured.update(_canonicalize(r) for r in result['results'])
    already_captured.update(_canonicalize(f) for f in result['files'])

    entities = []
    for cand in topic_candidates:
        if cand in already_captured:
            continue
        already_captured.add(cand)
        entities.append(cand)

    # Phase 4: Promote top entities to topics
    if entities:
        result['topics'] = entities[:3]
        result['entities'] = entities[3:]

    # Phase 5: Decision tagging
    text_lower = text.lower()
    has_decision = any(ind in text_lower for ind in DECISION_INDICATORS)
    if has_decision and result['topics']:
        result['decisions'] = [result['topics'][0]]

    return result


def encode_turn(text: str, speaker: str = "assistant", D: int = D_DEFAULT) -> tuple:
    """Full pipeline: raw text -> FHRR vector + extraction result."""
    ext = auto_extract(text, speaker)
    components = []

    # Speaker
    speaker_vec = concept_to_fhrr(f"speaker_{speaker}", D)
    components.append(_fhrr_bind(_get_role_vector('SPEAKER', D), speaker_vec))

    for role_name, field_name in [
        ('TOPIC', 'topics'), ('ACTION', 'actions'), ('FILE', 'files'),
        ('RESULT', 'results'), ('ENTITY', 'entities'), ('DECISION', 'decisions'),
    ]:
        for term in ext.get(field_name, []):
            components.append(_fhrr_bind(
                _get_role_vector(role_name, D),
                concept_to_fhrr(term, D)
            ))

    if not components:
        return concept_to_fhrr(text[:50], D), ext

    turn_vec = _fhrr_bundle(*components) if len(components) > 1 else components[0]
    return turn_vec, ext


# ============================================================
# CASCADE HIT
# ============================================================

@dataclass
class CascadeHit:
    """Result from cascade query."""
    session_id: str
    chunk_id: int
    turn_ids: List[int]
    score: float
    fhrr_score: float
    overlap_score: float
    matched_roles: Dict[str, List[str]]
    texts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# CASCADE QUERY (3-gate filter)
# ============================================================

def _role_overlap(query_terms: list, candidate_terms: list) -> float:
    """Fraction of query terms found in candidate."""
    if not query_terms:
        return 0.0
    q = set(t.lower().strip() for t in query_terms if t)
    c = set(t.lower().strip() for t in candidate_terms if t)
    if not q:
        return 0.0
    return len(q & c) / len(q)


def cascade_query(
    query_roles: Dict[str, list],
    query_vector: np.ndarray,
    session_record: Dict[str, Any],
    *,
    D: int = D_DEFAULT,
    threshold: float = THRESHOLD_DEFAULT,
    max_chunks: int = 5,
) -> List[CascadeHit]:
    """3-gate cascade filter: Session Bloom -> Chunk Bloom -> FHRR + overlap scoring."""

    # Build bloom query keys
    bloom_keys = []
    for _, field_name in ROLE_FIELDS:
        for term in query_roles.get(field_name, []):
            key = term.lower().strip()
            if key:
                bloom_keys.append(key)

    if not bloom_keys:
        return []

    # Gate 1: Session Bloom
    session_bloom = session_record.get('bloom')
    if session_bloom is None or not session_bloom.contains_any(bloom_keys):
        return []

    hits: List[CascadeHit] = []

    for chunk in session_record.get('chunks', []):
        # Gate 2: Chunk Bloom
        chunk_bloom = chunk.get('bloom')
        if chunk_bloom is None or not chunk_bloom.contains_any(bloom_keys):
            continue

        # Gate 3: FHRR + Role Scoring
        role_fhrr_scores = []
        matched_roles: Dict[str, List[str]] = {}

        for role_name, field_name in ROLE_FIELDS:
            query_terms = query_roles.get(field_name, [])
            chunk_head = chunk.get('role_heads', {}).get(role_name)

            if not query_terms or chunk_head is None:
                continue

            best_score = 0.0
            matched = []
            for term in query_terms:
                term_vec = concept_to_fhrr(term, D)
                s = _fhrr_similarity(term_vec, chunk_head)
                if s > 0.02:
                    matched.append(term)
                best_score = max(best_score, s)

            if best_score > 0:
                role_fhrr_scores.append(best_score)
            if matched:
                matched_roles[field_name] = matched

        avg_role_fhrr = (
            sum(role_fhrr_scores) / len(role_fhrr_scores)
            if role_fhrr_scores else 0.0
        )

        # Turn-level FHRR similarity
        turn_fhrr = 0.0
        if chunk.get('turn_head') is not None:
            turn_fhrr = _fhrr_similarity(query_vector, chunk['turn_head'])

        # Symbolic role overlap
        all_query_terms = []
        all_chunk_terms = []
        for _, field_name in ROLE_FIELDS:
            all_query_terms.extend(query_roles.get(field_name, []))
            all_chunk_terms.extend(chunk.get('roles', {}).get(field_name, []))
        overlap = _role_overlap(all_query_terms, all_chunk_terms)

        # Combined score
        combined = (
            W_ROLE_FHRR * avg_role_fhrr
            + W_TURN_FHRR * turn_fhrr
            + W_OVERLAP * overlap
            + W_MATCH_BONUS * (1.0 if any(matched_roles.values()) else 0.0)
        )

        if combined < threshold:
            continue

        hits.append(CascadeHit(
            session_id=session_record['session_id'],
            chunk_id=chunk['chunk_id'],
            turn_ids=list(chunk.get('turn_ids', [])),
            score=combined,
            fhrr_score=avg_role_fhrr,
            overlap_score=overlap,
            matched_roles=matched_roles,
            texts=list(chunk.get('texts', [])),
            metadata=chunk.get('metadata', {}),
        ))

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:max_chunks]


# ============================================================
# SESSION INDEX — BUILD, SAVE, LOAD
# ============================================================

def build_session_record(
    session_id: str,
    turns: List[Dict[str, Any]],
    *,
    D: int = D_DEFAULT,
    chunk_size: int = CHUNK_SIZE_DEFAULT,
) -> Dict[str, Any]:
    """Build session record from raw conversation turns.

    Each turn: {"text": str, "role": str, "turn_id": int (optional)}
    """
    turn_records = []
    for i, turn in enumerate(turns):
        text = str(turn.get('text', ''))
        speaker = str(turn.get('role', 'unknown'))
        turn_id = int(turn.get('turn_id', i))

        ext = auto_extract(text, speaker)
        vec, _ = encode_turn(text, speaker, D)

        turn_records.append({
            'turn_id': turn_id,
            'text': text,
            'roles': {
                'topics': list(ext.get('topics', [])),
                'actions': list(ext.get('actions', [])),
                'files': list(ext.get('files', [])),
                'results': list(ext.get('results', [])),
                'entities': list(ext.get('entities', [])),
                'decisions': list(ext.get('decisions', [])),
            },
            'vector': vec,
        })

    if not turn_records:
        return _empty_session(session_id)

    # Session-level structures
    session_bloom = BloomFilter(size_bits=BLOOM_BITS, num_hashes=BLOOM_HASHES)
    session_roles: Dict[str, list] = {fn: [] for _, fn in ROLE_FIELDS}
    role_concept_sets: Dict[str, set] = {rn: set() for rn, _ in ROLE_FIELDS}

    for tr in turn_records:
        for role_name, field_name in ROLE_FIELDS:
            for term in tr['roles'].get(field_name, []):
                canon = _canonicalize(term)
                session_bloom.add(canon)
                role_concept_sets[role_name].add(canon)
                if canon not in session_roles[field_name]:
                    session_roles[field_name].append(canon)

    # Per-role FHRR heads: bundle of unique concept vectors
    session_role_heads = {}
    for role_name, _ in ROLE_FIELDS:
        concepts = list(role_concept_sets[role_name])
        if concepts:
            vecs = [concept_to_fhrr(c, D) for c in concepts]
            session_role_heads[role_name] = _fhrr_bundle(*vecs) if len(vecs) > 1 else vecs[0]
        else:
            session_role_heads[role_name] = None

    # Session turn head: bundle of all turn vectors
    all_turn_vecs = [tr['vector'] for tr in turn_records]
    session_turn_head = _fhrr_bundle(*all_turn_vecs) if len(all_turn_vecs) > 1 else all_turn_vecs[0]

    # Build chunks
    chunks = _build_chunks(turn_records, D, chunk_size)

    return {
        'session_id': session_id,
        'bloom': session_bloom,
        'role_heads': session_role_heads,
        'turn_head': session_turn_head,
        'roles': session_roles,
        'chunks': chunks,
        'num_turns': len(turn_records),
    }


def _build_chunks(turn_records: list, D: int, chunk_size: int) -> list:
    """Build chunks with Bloom + FHRR heads from turn records."""
    chunks = []
    for chunk_id, start in enumerate(range(0, len(turn_records), chunk_size)):
        batch = turn_records[start:start + chunk_size]

        chunk_bloom = BloomFilter(size_bits=BLOOM_BITS, num_hashes=BLOOM_HASHES)
        chunk_roles: Dict[str, list] = {fn: [] for _, fn in ROLE_FIELDS}
        chunk_concept_sets: Dict[str, set] = {rn: set() for rn, _ in ROLE_FIELDS}

        for tr in batch:
            for role_name, field_name in ROLE_FIELDS:
                for term in tr['roles'].get(field_name, []):
                    canon = _canonicalize(term)
                    chunk_bloom.add(canon)
                    chunk_concept_sets[role_name].add(canon)
                    if canon not in chunk_roles[field_name]:
                        chunk_roles[field_name].append(canon)

        chunk_role_heads = {}
        for role_name, _ in ROLE_FIELDS:
            concepts = list(chunk_concept_sets[role_name])
            if concepts:
                vecs = [concept_to_fhrr(c, D) for c in concepts]
                chunk_role_heads[role_name] = _fhrr_bundle(*vecs) if len(vecs) > 1 else vecs[0]
            else:
                chunk_role_heads[role_name] = None

        chunk_turn_vecs = [tr['vector'] for tr in batch]
        chunk_turn_head = _fhrr_bundle(*chunk_turn_vecs) if len(chunk_turn_vecs) > 1 else chunk_turn_vecs[0]

        chunks.append({
            'chunk_id': chunk_id,
            'turn_ids': [tr['turn_id'] for tr in batch],
            'texts': [tr['text'] for tr in batch],
            'roles': chunk_roles,
            'bloom': chunk_bloom,
            'role_heads': chunk_role_heads,
            'turn_head': chunk_turn_head,
            'metadata': {
                'num_turns': len(batch),
                'start_turn_id': batch[0]['turn_id'],
                'end_turn_id': batch[-1]['turn_id'],
            },
        })

    return chunks


def _empty_session(session_id: str) -> Dict[str, Any]:
    """Empty session record for edge case of no turns."""
    return {
        'session_id': session_id,
        'bloom': BloomFilter(),
        'role_heads': {rn: None for rn, _ in ROLE_FIELDS},
        'turn_head': None,
        'roles': {fn: [] for _, fn in ROLE_FIELDS},
        'chunks': [],
        'num_turns': 0,
    }


# ============================================================
# PERSISTENCE — SAVE / LOAD
# ============================================================

def save_session_record(record: Dict[str, Any], directory: str = None) -> str:
    """Save session record to disk. Returns path of saved files."""
    directory = directory or FHRR_SESSIONS_DIR
    os.makedirs(directory, exist_ok=True)
    sid = record['session_id']
    t0 = time.monotonic()

    # Metadata (JSON)
    meta = {
        'session_id': sid,
        'num_turns': record['num_turns'],
        'roles': record['roles'],
        'chunks': [
            {
                'chunk_id': c['chunk_id'],
                'turn_ids': c['turn_ids'],
                'texts': c['texts'],
                'roles': c['roles'],
                'metadata': c['metadata'],
            }
            for c in record['chunks']
        ],
    }

    json_path = os.path.join(directory, f"{sid}.json")
    with open(json_path, 'w') as f:
        json.dump(meta, f, ensure_ascii=False)

    # Vectors (.npz)
    arrays = {}
    arrays['s_bloom'] = record['bloom'].to_numpy()
    if record['turn_head'] is not None:
        arrays['s_turn'] = record['turn_head']
    for role_name, _ in ROLE_FIELDS:
        head = record['role_heads'].get(role_name)
        if head is not None:
            arrays[f's_{role_name}'] = head
    for chunk in record['chunks']:
        cid = chunk['chunk_id']
        arrays[f'c{cid}_bloom'] = chunk['bloom'].to_numpy()
        if chunk['turn_head'] is not None:
            arrays[f'c{cid}_turn'] = chunk['turn_head']
        for role_name, _ in ROLE_FIELDS:
            head = chunk['role_heads'].get(role_name)
            if head is not None:
                arrays[f'c{cid}_{role_name}'] = head

    npz_path = os.path.join(directory, f"{sid}.npz")
    np.savez_compressed(npz_path, **arrays)

    elapsed_ms = (time.monotonic() - t0) * 1000

    # Register in SQLite index
    try:
        file_size = os.path.getsize(json_path) + os.path.getsize(npz_path)
        topics = record['roles'].get('topics', [])
        db = connect_fts(FTS_DB_PATH)
        try:
            db.execute("""
                INSERT OR REPLACE INTO fhrr_session_index
                (session_id, encoded_at, num_turns, num_chunks,
                 file_size_bytes, encoding_time_ms, topics_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                sid, now_iso(), record['num_turns'], len(record['chunks']),
                file_size, round(elapsed_ms, 1), json.dumps(topics)
            ))
            db.commit()
        finally:
            db.close()
    except Exception as e:
        _logger.warning("Failed to register session %s in SQLite: %s", sid, e)

    # Invalidate hot index so next query rebuilds with new session
    _invalidate_hot_index()

    return directory


def load_session_record(session_id: str, directory: str = None) -> Optional[Dict[str, Any]]:
    """Load a single session record from disk."""
    directory = directory or FHRR_SESSIONS_DIR
    json_path = os.path.join(directory, f"{session_id}.json")
    npz_path = os.path.join(directory, f"{session_id}.npz")

    if not os.path.exists(json_path) or not os.path.exists(npz_path):
        return None

    with open(json_path) as f:
        meta = json.load(f)

    npz = np.load(npz_path)

    session_bloom = BloomFilter.from_numpy(npz['s_bloom'])
    session_turn_head = npz['s_turn'] if 's_turn' in npz else None

    session_role_heads = {}
    for role_name, _ in ROLE_FIELDS:
        key = f's_{role_name}'
        session_role_heads[role_name] = npz[key] if key in npz else None

    chunks = []
    for chunk_meta in meta['chunks']:
        cid = chunk_meta['chunk_id']
        chunk_bloom = BloomFilter.from_numpy(npz[f'c{cid}_bloom'])
        chunk_turn_head = npz[f'c{cid}_turn'] if f'c{cid}_turn' in npz else None

        chunk_role_heads = {}
        for role_name, _ in ROLE_FIELDS:
            key = f'c{cid}_{role_name}'
            chunk_role_heads[role_name] = npz[key] if key in npz else None

        chunks.append({
            'chunk_id': cid,
            'turn_ids': chunk_meta['turn_ids'],
            'texts': chunk_meta['texts'],
            'roles': chunk_meta['roles'],
            'bloom': chunk_bloom,
            'role_heads': chunk_role_heads,
            'turn_head': chunk_turn_head,
            'metadata': chunk_meta.get('metadata', {}),
        })

    npz.close()

    return {
        'session_id': meta['session_id'],
        'bloom': session_bloom,
        'role_heads': session_role_heads,
        'turn_head': session_turn_head,
        'roles': meta['roles'],
        'chunks': chunks,
        'num_turns': meta['num_turns'],
    }


def load_all_sessions(directory: str = None) -> List[Dict[str, Any]]:
    """Load all session records from disk."""
    directory = directory or FHRR_SESSIONS_DIR
    if not os.path.exists(directory):
        return []

    sessions = []
    for fname in sorted(os.listdir(directory)):
        if fname.endswith('.json'):
            sid = fname.replace('.json', '')
            npz_path = os.path.join(directory, f"{sid}.npz")
            if os.path.exists(npz_path):
                try:
                    sessions.append(load_session_record(sid, directory))
                except Exception as e:
                    _logger.warning("Failed to load session %s: %s", sid, e)
    return sessions


# ============================================================
# HOT INDEX — Pre-compiled vectorized query (22s → <50ms)
# ============================================================

@dataclass
class _HotIndexData:
    """Pre-compiled session-level data for vectorized querying."""
    session_ids: List[str]
    bloom_matrix: np.ndarray      # (N, BLOOM_BITS) uint8
    topic_heads: np.ndarray       # (N, D) complex64
    turn_heads: np.ndarray        # (N, D) complex64
    roles: List[Dict[str, list]]  # session-level roles metadata
    compiled_at: str
    num_sessions: int


# Module-level singleton
_HOT_INDEX: Optional[_HotIndexData] = None
_hot_index_stale: bool = True


def _invalidate_hot_index():
    """Mark hot index as stale (e.g., after new session encoded)."""
    global _hot_index_stale
    _hot_index_stale = True


def _bloom_hash_indices(term: str) -> List[int]:
    """Compute Bloom hash indices for a term (matches BloomFilter._indexes)."""
    key = term.strip().lower()
    return [
        int(hashlib.sha256(f"{i}:{key}".encode()).hexdigest(), 16) % BLOOM_BITS
        for i in range(BLOOM_HASHES)
    ]


def _vectorized_bloom_check(bloom_matrix: np.ndarray, terms: list) -> np.ndarray:
    """Check which sessions MAY contain ANY of the terms. Returns (N,) bool mask."""
    N = bloom_matrix.shape[0]
    any_match = np.zeros(N, dtype=bool)

    for term in terms:
        indices = _bloom_hash_indices(term)
        # All hash bits must be set for this term (contains semantics)
        term_match = np.ones(N, dtype=bool)
        for idx in indices:
            term_match &= (bloom_matrix[:, idx] == 1)
        any_match |= term_match  # contains_any semantics

    return any_match


def compile_hot_index(directory: str = None) -> Optional[_HotIndexData]:
    """Compile session-level arrays for vectorized querying.

    Reads only s_bloom, s_TOPIC, s_turn from each NPZ (skips chunk arrays).
    Saves to fhrr_hot_index.npz + fhrr_hot_meta.json.
    """
    global _HOT_INDEX, _hot_index_stale

    directory = directory or FHRR_SESSIONS_DIR
    if not os.path.exists(directory):
        return None

    session_ids = sorted([
        f[:-5] for f in os.listdir(directory) if f.endswith('.json')
    ])
    N = len(session_ids)
    if N == 0:
        return None

    D = D_DEFAULT
    t0 = time.monotonic()

    bloom_matrix = np.zeros((N, BLOOM_BITS), dtype=np.uint8)
    topic_heads = np.zeros((N, D), dtype=np.complex64)
    turn_heads = np.zeros((N, D), dtype=np.complex64)
    roles_list = []
    loaded = 0

    for i, sid in enumerate(session_ids):
        npz_path = os.path.join(directory, f"{sid}.npz")
        json_path = os.path.join(directory, f"{sid}.json")

        try:
            # Selective NPZ load: only session-level arrays (no chunk decompression)
            npz = np.load(npz_path)
            bloom_matrix[i] = npz['s_bloom']
            npz_files = npz.files
            if 's_TOPIC' in npz_files:
                topic_heads[i] = npz['s_TOPIC']
            if 's_turn' in npz_files:
                turn_heads[i] = npz['s_turn']
            npz.close()

            with open(json_path) as f:
                meta = json.load(f)
            # Pre-canonicalize roles for fast overlap at query time
            raw_roles = meta.get('roles', {})
            canon_roles = {}
            all_terms_set = set()
            for _, fn in ROLE_FIELDS:
                terms = [_canonicalize(t) for t in raw_roles.get(fn, []) if t]
                canon_roles[fn] = terms
                all_terms_set.update(terms)
            canon_roles['_all'] = sorted(all_terms_set)
            roles_list.append(canon_roles)
            loaded += 1
        except Exception as e:
            _logger.warning("compile_hot_index: skip %s: %s", sid, e)
            roles_list.append({})

    compile_ms = (time.monotonic() - t0) * 1000

    # Save to disk (uncompressed for fast reload)
    np.savez(HOT_INDEX_PATH,
             bloom_matrix=bloom_matrix,
             topic_heads=topic_heads,
             turn_heads=turn_heads)

    meta_out = {
        'compiled_at': now_iso(),
        'num_sessions': N,
        'loaded': loaded,
        'compile_ms': round(compile_ms, 1),
        'D': D,
        'session_ids': session_ids,
        'roles': roles_list,
    }
    with open(HOT_META_PATH, 'w') as f:
        json.dump(meta_out, f, ensure_ascii=False)

    _logger.info("Hot index compiled: %d/%d sessions in %.0fms", loaded, N, compile_ms)

    hot = _HotIndexData(
        session_ids=session_ids,
        bloom_matrix=bloom_matrix,
        topic_heads=topic_heads,
        turn_heads=turn_heads,
        roles=roles_list,
        compiled_at=meta_out['compiled_at'],
        num_sessions=N,
    )
    _HOT_INDEX = hot
    _hot_index_stale = False
    return hot


def _load_hot_index() -> Optional[_HotIndexData]:
    """Load or compile hot index singleton."""
    global _HOT_INDEX, _hot_index_stale

    if _HOT_INDEX is not None and not _hot_index_stale:
        return _HOT_INDEX

    # Try loading from disk
    if os.path.exists(HOT_INDEX_PATH) and os.path.exists(HOT_META_PATH):
        try:
            t0 = time.monotonic()

            with open(HOT_META_PATH) as f:
                meta = json.load(f)

            # Check if index covers all sessions on disk
            if os.path.exists(FHRR_SESSIONS_DIR):
                num_on_disk = sum(1 for f in os.listdir(FHRR_SESSIONS_DIR)
                                  if f.endswith('.json'))
            else:
                num_on_disk = 0

            if meta.get('num_sessions', 0) >= num_on_disk:
                data = np.load(HOT_INDEX_PATH)

                _HOT_INDEX = _HotIndexData(
                    session_ids=meta['session_ids'],
                    bloom_matrix=data['bloom_matrix'],
                    topic_heads=data['topic_heads'],
                    turn_heads=data['turn_heads'],
                    roles=meta['roles'],
                    compiled_at=meta.get('compiled_at', ''),
                    num_sessions=meta['num_sessions'],
                )
                _hot_index_stale = False

                load_ms = (time.monotonic() - t0) * 1000
                _logger.info("Hot index loaded: %d sessions in %.0fms",
                              meta['num_sessions'], load_ms)
                return _HOT_INDEX
        except Exception as e:
            _logger.warning("Failed to load hot index: %s", e)

    # Compile fresh
    return compile_hot_index()


# ============================================================
# BINARY RECALL — main query interface
# ============================================================

def binary_recall(
    query_text: str,
    *,
    max_results: int = MAX_RESULTS_DEFAULT,
    threshold: float = THRESHOLD_DEFAULT,
    D: int = D_DEFAULT,
    sessions: Optional[List[Dict[str, Any]]] = None,
    max_chunks_per_session: int = 20,
) -> List[Dict[str, Any]]:
    """Query FHRR binary memory with natural text.

    Uses pre-compiled hot index for vectorized querying (sub-50ms).
    Falls back to sequential scan if hot index unavailable or sessions provided.

    Returns list of result dicts sorted by score descending.
    """
    # Legacy path: explicit sessions provided
    if sessions is not None:
        return _binary_recall_scan(
            query_text, sessions=sessions, max_results=max_results,
            threshold=threshold, D=D,
            max_chunks_per_session=max_chunks_per_session,
        )

    # Try hot index
    hot = _load_hot_index()
    if hot is None or hot.num_sessions == 0:
        return _binary_recall_scan(
            query_text, sessions=load_all_sessions(),
            max_results=max_results, threshold=threshold, D=D,
            max_chunks_per_session=max_chunks_per_session,
        )

    t0 = time.monotonic()

    # Step 1: Extract & encode query
    query_ext = auto_extract(query_text, speaker="user")
    query_roles = {fn: query_ext.get(fn, []) for _, fn in ROLE_FIELDS}
    query_vector, _ = encode_turn(query_text, speaker="user", D=D)

    # Build bloom keys (canonical, matching build_session_record)
    bloom_keys = []
    for _, fn in ROLE_FIELDS:
        for term in query_roles.get(fn, []):
            key = _canonicalize(term)
            if key:
                bloom_keys.append(key)

    if not bloom_keys:
        return []

    # Step 2: Vectorized Bloom filter (sub-ms)
    survivors = _vectorized_bloom_check(hot.bloom_matrix, bloom_keys)
    survivor_idx = np.where(survivors)[0]

    if len(survivor_idx) == 0:
        return []

    # Step 3: Vectorized FHRR similarity on survivors
    # Topic similarity: best match across query topic terms
    topic_sims = np.zeros(len(survivor_idx), dtype=np.float64)
    for term in query_roles.get('topics', []):
        term_vec = concept_to_fhrr(_canonicalize(term), D)
        sims = np.abs(np.conj(hot.topic_heads[survivor_idx]) @ term_vec) / D
        topic_sims = np.maximum(topic_sims, sims)

    # Turn similarity
    turn_sims = np.abs(np.conj(hot.turn_heads[survivor_idx]) @ query_vector) / D

    # Step 4: Symbolic overlap on survivors (uses pre-canonicalized roles)
    all_query_terms = set()
    for _, fn in ROLE_FIELDS:
        all_query_terms.update(_canonicalize(t) for t in query_roles.get(fn, []))
    all_query_terms.discard('')
    nq = len(all_query_terms)

    overlap_scores = np.zeros(len(survivor_idx), dtype=np.float64)
    if nq > 0:
        for i, idx in enumerate(survivor_idx):
            if idx < len(hot.roles):
                # _all is pre-canonicalized set of all session terms
                session_terms = hot.roles[idx].get('_all', [])
                overlap_scores[i] = len(all_query_terms & set(session_terms)) / nq

    # Step 5: Combined score (same weights as cascade_query)
    combined = (
        W_ROLE_FHRR * topic_sims
        + W_TURN_FHRR * turn_sims
        + W_OVERLAP * overlap_scores
        + W_MATCH_BONUS * (overlap_scores > 0).astype(np.float64)
    )

    # Filter & rank
    valid_local = np.where(combined >= threshold)[0]
    if len(valid_local) == 0:
        return []

    ranked_local = valid_local[np.argsort(combined[valid_local])[::-1]]
    top_k = ranked_local[:max(max_results * 3, 30)]

    # Step 6: Load JSON for top-K sessions (lightweight, ~3ms each)
    results = []
    for local_i in top_k:
        orig_idx = int(survivor_idx[local_i])
        sid = hot.session_ids[orig_idx]
        session_score = float(combined[local_i])

        json_path = os.path.join(FHRR_SESSIONS_DIR, f"{sid}.json")
        try:
            with open(json_path) as f:
                meta = json.load(f)
        except Exception:
            continue

        # Find best chunk by overlap
        best_chunk = None
        best_overlap = -1.0
        for chunk_meta in meta.get('chunks', []):
            chunk_terms = set()
            for _, fn in ROLE_FIELDS:
                for t in chunk_meta.get('roles', {}).get(fn, []):
                    chunk_terms.add(t.lower().strip())
            co = len(all_query_terms & chunk_terms) / nq if nq else 0.0
            if co > best_overlap:
                best_overlap = co
                best_chunk = chunk_meta

        if best_chunk is None and meta.get('chunks'):
            best_chunk = meta['chunks'][0]

        if best_chunk is None:
            continue

        # Build matched_roles
        matched = {}
        for _, fn in ROLE_FIELDS:
            q_terms = set(_canonicalize(t) for t in query_roles.get(fn, []))
            c_terms = set(
                _canonicalize(t) for t in best_chunk.get('roles', {}).get(fn, [])
            )
            common = q_terms & c_terms
            if common:
                matched[fn] = list(common)

        results.append({
            'session_id': sid,
            'chunk_id': best_chunk['chunk_id'],
            'turn_ids': best_chunk.get('turn_ids', []),
            'score': round(session_score, 4),
            'chunk_score': round(best_overlap, 4),
            'fhrr_score': round(float(topic_sims[local_i]), 4),
            'overlap_score': round(float(overlap_scores[local_i]), 4),
            'matched_roles': matched,
            'context': {
                'texts': best_chunk.get('texts', []),
                'metadata': best_chunk.get('metadata', {}),
            },
        })

        if len(results) >= max_results:
            break

    elapsed_ms = (time.monotonic() - t0) * 1000
    _logger.info("binary_recall hot: %d results in %.1fms (bloom: %d/%d survivors)",
                  len(results), elapsed_ms, len(survivor_idx), hot.num_sessions)

    return results


def _binary_recall_scan(
    query_text: str,
    *,
    sessions: List[Dict[str, Any]],
    max_results: int = MAX_RESULTS_DEFAULT,
    threshold: float = THRESHOLD_DEFAULT,
    D: int = D_DEFAULT,
    max_chunks_per_session: int = 20,
) -> List[Dict[str, Any]]:
    """Legacy binary_recall: sequential scan of all sessions."""
    if not sessions:
        return []

    query_ext = auto_extract(query_text, speaker="user")
    query_roles = {fn: query_ext.get(fn, []) for _, fn in ROLE_FIELDS}
    query_vector, _ = encode_turn(query_text, speaker="user", D=D)

    session_hits: Dict[str, List[CascadeHit]] = {}

    for session in sessions:
        hits = cascade_query(
            query_roles=query_roles,
            query_vector=query_vector,
            session_record=session,
            D=D,
            threshold=threshold,
            max_chunks=max_chunks_per_session,
        )
        if hits:
            session_hits[session['session_id']] = hits

    session_scores: list = []
    for sid, hits in session_hits.items():
        hits.sort(key=lambda h: h.score, reverse=True)
        session_score = sum(h.score * (DECAY ** i) for i, h in enumerate(hits))
        session_scores.append((session_score, sid, hits))

    session_scores.sort(key=lambda x: x[0], reverse=True)

    results = []
    for session_score, sid, hits in session_scores:
        for hit in hits:
            if len(results) >= max_results:
                break
            results.append({
                'session_id': hit.session_id,
                'chunk_id': hit.chunk_id,
                'turn_ids': hit.turn_ids,
                'score': round(session_score, 4),
                'chunk_score': round(hit.score, 4),
                'fhrr_score': round(hit.fhrr_score, 4),
                'overlap_score': round(hit.overlap_score, 4),
                'matched_roles': hit.matched_roles,
                'context': {
                    'texts': hit.texts,
                    'metadata': hit.metadata,
                },
            })
        if len(results) >= max_results:
            break

    return results


# ============================================================
# SESSION BOOST — for memory_core.py integration
# ============================================================

# Per-query cache: binary_recall runs ONCE (~10ms), then each memory checks cache (<1ms)
_session_boost_cache: Dict[str, Dict[str, float]] = {}


def get_session_boost(query: str, memory_session_id: str) -> float:
    """Get FHRR session boost for a memory. 0.0-0.15.

    Caches binary_recall results per query to avoid repeated computation.
    """
    if not memory_session_id:
        return 0.0

    if query not in _session_boost_cache:
        try:
            hits = binary_recall(query, max_results=10)
            boost_map = {}
            for hit in hits:
                sid = hit['session_id']
                if sid not in boost_map:
                    boost_map[sid] = min(hit['score'] * SESSION_BOOST_CAP, SESSION_BOOST_CAP)
            _session_boost_cache[query] = boost_map
        except Exception as e:
            _logger.debug("Session boost failed for query: %s", e)
            _session_boost_cache[query] = {}

    return _session_boost_cache.get(query, {}).get(memory_session_id, 0.0)


def clear_session_boost_cache():
    """Clear the per-query session boost cache."""
    _session_boost_cache.clear()


# ============================================================
# ENCODE CURRENT SESSION — from pgvector memories
# ============================================================

def encode_current_session(session_id: str = None) -> Optional[Dict[str, Any]]:
    """Encode the current session from pgvector memories.

    Fetches turns for the given session_id from memory storage,
    builds a session record, and saves to disk.

    Returns the session record or None on failure.
    """
    try:
        from modules.pg_store import pg

        # Get memories for this session
        all_points = pg.get_all(limit=500, is_semantic=False)
        if not all_points:
            return None

        # Filter by session_id if provided
        turns = []
        for pt in all_points:
            payload = pt.payload or {}
            mem_session = payload.get('temporal_session_id', '')
            if session_id and mem_session != session_id:
                continue
            turns.append({
                'text': payload.get('data', ''),
                'role': payload.get('speaker', 'unknown'),
                'turn_id': len(turns),
            })

        if not turns:
            _logger.debug("No turns found for session %s", session_id)
            return None

        sid = session_id or f"session_{now_iso()[:10]}"
        record = build_session_record(sid, turns)
        save_session_record(record)
        _logger.info("Encoded session %s: %d turns, %d chunks",
                      sid, record['num_turns'], len(record['chunks']))
        return record

    except Exception as e:
        _logger.error("Failed to encode session %s: %s", session_id, e)
        return None


# ============================================================
# STATS
# ============================================================

def get_index_stats() -> Dict[str, Any]:
    """Get statistics about the FHRR session index."""
    stats = {
        'sessions_dir': FHRR_SESSIONS_DIR,
        'dir_exists': os.path.exists(FHRR_SESSIONS_DIR),
        'num_sessions': 0,
        'total_size_bytes': 0,
        'total_turns': 0,
        'total_chunks': 0,
    }

    if not os.path.exists(FHRR_SESSIONS_DIR):
        return stats

    try:
        db = connect_fts(FTS_DB_PATH)
        try:
            row = db.execute("""
                SELECT COUNT(*), COALESCE(SUM(num_turns), 0),
                       COALESCE(SUM(num_chunks), 0), COALESCE(SUM(file_size_bytes), 0)
                FROM fhrr_session_index
            """).fetchone()
            if row:
                stats['num_sessions'] = row[0]
                stats['total_turns'] = row[1]
                stats['total_chunks'] = row[2]
                stats['total_size_bytes'] = row[3]
        finally:
            db.close()
    except Exception:
        # Fallback to filesystem count
        json_files = [f for f in os.listdir(FHRR_SESSIONS_DIR) if f.endswith('.json')]
        stats['num_sessions'] = len(json_files)
        for f in os.listdir(FHRR_SESSIONS_DIR):
            stats['total_size_bytes'] += os.path.getsize(
                os.path.join(FHRR_SESSIONS_DIR, f)
            )

    return stats


# ============================================================
# SCHEMA PROTOTYPES — novelty detection (van Kesteren 2012)
# ============================================================

FHRR_PROTOTYPES_DIR = os.path.join(DATA_DIR, "fhrr_prototypes")


def compute_schema_prototype(topic: str, sessions: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    """Average role_heads across sessions mentioning topic -> schema prototype.

    Paper: van Kesteren 2012 (mPFC schema congruency).
    The prototype captures "what a typical session about X looks like."
    Used ONLY for novelty detection, not retrieval.
    """
    topic_canon = _canonicalize(topic)
    if sessions is None:
        sessions = load_all_sessions()

    matching = []
    for s in sessions:
        s_topics = s.get('roles', {}).get('topics', [])
        if any(_canonicalize(t) == topic_canon for t in s_topics):
            matching.append(s)

    if len(matching) < 3:
        return None  # Need >= 3 sessions for meaningful prototype

    # Average each role head across matching sessions
    prototype_heads: Dict[str, Optional[np.ndarray]] = {}
    for role_name, _ in ROLE_FIELDS:
        vecs = []
        for s in matching:
            head = s.get('role_heads', {}).get(role_name)
            if head is not None:
                vecs.append(head)
        if vecs:
            avg = sum(vecs) / len(vecs)
            prototype_heads[role_name] = (avg / np.abs(avg)).astype(np.complex64)
        else:
            prototype_heads[role_name] = None

    # Collect all topics seen across matching sessions
    all_topics = set()
    for s in matching:
        for t in s.get('roles', {}).get('topics', []):
            all_topics.add(_canonicalize(t))

    return {
        'topic': topic_canon,
        'prototype_heads': prototype_heads,
        'num_sessions': len(matching),
        'topics_seen': sorted(all_topics),
    }


def load_prototype(topic: str) -> Optional[Dict[str, Any]]:
    """Load pre-saved schema prototype from FHRR_PROTOTYPES_DIR.

    Returns None if no prototype exists for the topic.
    Use compute_schema_prototype() only when sessions are available and prototype not yet saved.
    """
    topic_canon = _canonicalize(topic)
    if not topic_canon:
        return None
    npz_path = os.path.join(FHRR_PROTOTYPES_DIR, f"{topic_canon}.npz")
    if not os.path.exists(npz_path):
        return None
    try:
        data = np.load(npz_path)
        prototype_heads: Dict[str, Optional[np.ndarray]] = {}
        for role_name, _ in ROLE_FIELDS:
            key = f'proto_{role_name}'
            prototype_heads[role_name] = data[key] if key in data else None
        data.close()
        return {'topic': topic_canon, 'prototype_heads': prototype_heads}
    except Exception:
        return None


def compute_novelty_score(session_record: Dict[str, Any], sessions: Optional[List[Dict[str, Any]]] = None) -> float:
    """Compare session against schema prototypes. Returns 0.0-1.0.

    Loads pre-saved prototypes from FHRR_PROTOTYPES_DIR (no load_all_sessions needed).
    Fallback to compute_schema_prototype() only if sessions explicitly provided.
    Paper: Kumaran & Maguire 2007 (hippocampal match-mismatch).
    """
    s_topics = session_record.get('roles', {}).get('topics', [])
    if not s_topics:
        return 1.0  # No topics -> everything is novel

    similarities = []
    for topic in s_topics:
        # Load pre-saved prototype (fast — no session reload)
        proto = load_prototype(topic)
        # Fallback: compute on-the-fly only when sessions explicitly provided
        if proto is None and sessions is not None:
            proto = compute_schema_prototype(topic, sessions=sessions)
        if proto is None:
            continue
        # Compare session's TOPIC role head against prototype's TOPIC head
        s_head = session_record.get('role_heads', {}).get('TOPIC')
        p_head = proto['prototype_heads'].get('TOPIC')
        if s_head is not None and p_head is not None:
            sim = _fhrr_similarity(s_head, p_head)
            similarities.append(sim)

    if not similarities:
        return 1.0  # No prototypes exist -> everything is novel

    return round(1.0 - max(similarities), 4)


def get_all_prototypes(sessions: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Return prototypes for all known topics with >= 3 sessions.

    Uses pre-built inverted index for O(N) complexity vs O(T*N) brute force.
    N=sessions, T=unique topics. Speedup: ~500x with 1544 sessions/21K topics.
    """
    if sessions is None:
        sessions = load_all_sessions()

    # ONE PASS: build topic->sessions inverted index (O(N*T_avg))
    topic_to_sessions: Dict[str, list] = {}
    for s in sessions:
        for t in s.get('roles', {}).get('topics', []):
            canon = _canonicalize(t)
            if canon:
                topic_to_sessions.setdefault(canon, []).append(s)

    # Compute prototype for each topic with >= 3 sessions
    prototypes = []
    for topic_canon, matching in sorted(topic_to_sessions.items()):
        if len(matching) < 3:
            continue
        prototype_heads: Dict[str, Optional[np.ndarray]] = {}
        for role_name, _ in ROLE_FIELDS:
            vecs = [s.get('role_heads', {}).get(role_name) for s in matching
                    if s.get('role_heads', {}).get(role_name) is not None]
            if vecs:
                avg = sum(vecs) / len(vecs)
                prototype_heads[role_name] = (avg / np.abs(avg)).astype(np.complex64)
            else:
                prototype_heads[role_name] = None
        all_t: set = set()
        for s in matching:
            for t in s.get('roles', {}).get('topics', []):
                all_t.add(_canonicalize(t))
        prototypes.append({
            'topic': topic_canon,
            'prototype_heads': prototype_heads,
            'num_sessions': len(matching),
            'topics_seen': sorted(all_t),
        })
    return prototypes


def save_prototypes(prototypes: List[Dict[str, Any]]) -> int:
    """Save prototypes to disk and register in SQLite. Returns count saved."""
    os.makedirs(FHRR_PROTOTYPES_DIR, exist_ok=True)
    saved = 0
    for proto in prototypes:
        topic = proto['topic']
        npz_path = os.path.join(FHRR_PROTOTYPES_DIR, f"{topic}.npz")
        arrays = {}
        for role_name, _ in ROLE_FIELDS:
            head = proto['prototype_heads'].get(role_name)
            if head is not None:
                arrays[f'proto_{role_name}'] = head
        if arrays:
            np.savez_compressed(npz_path, **arrays)
            try:
                db = connect_fts(FTS_DB_PATH)
                try:
                    db.execute("""
                        INSERT OR REPLACE INTO fhrr_schema_prototypes
                        (topic, num_sessions, updated_at, prototype_path)
                        VALUES (?, ?, ?, ?)
                    """, (topic, proto['num_sessions'], now_iso(), npz_path))
                    db.commit()
                finally:
                    db.close()
            except Exception as e:
                _logger.warning("Failed to register prototype %s: %s", topic, e)
            saved += 1
    return saved


# ============================================================
# SYNONYM DISCOVERY — FHRR concept similarity (HippoRAG 2024)
# ============================================================

def discover_synonym_edges(threshold: float = 0.80, max_pairs: int = 100,
                           sessions: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Find concept pairs with FHRR cosine > threshold.

    Paper: HippoRAG NeurIPS 2024 (parahippocampal synonym detection).
    Resolves low coverage of spreading_edges by discovering similarity links.
    """
    if sessions is None:
        sessions = load_all_sessions()

    # Collect ALL unique concepts across all sessions with frequency
    concept_freq: Dict[str, int] = {}
    for s in sessions:
        roles = s.get('roles', {})
        for _, field_name in ROLE_FIELDS:
            for term in roles.get(field_name, []):
                canon = _canonicalize(term)
                if canon and len(canon) >= 2 and canon not in STOP_WORDS:
                    concept_freq[canon] = concept_freq.get(canon, 0) + 1

    # Cap vocabulary to prevent OOM: take top-N most frequent concepts
    if len(concept_freq) > MAX_VOCAB_DISCOVER:
        concepts = sorted(concept_freq, key=lambda c: -concept_freq[c])[:MAX_VOCAB_DISCOVER]
    else:
        concepts = sorted(concept_freq.keys())
    n = len(concepts)
    if n < 2:
        return []

    # Pre-compute all FHRR vectors as matrix for batch cosine
    vectors = np.array([concept_to_fhrr(c) for c in concepts])  # (n, D) complex64

    # Batch cosine similarity: |conj(a) . b| / D
    D = vectors.shape[1]
    sim_matrix = np.abs(np.conj(vectors) @ vectors.T) / D

    # Vectorized extraction of pairs above threshold (numpy, no Python loop)
    i_idx, j_idx = np.where(np.triu(sim_matrix > threshold, k=1))
    if len(i_idx) == 0:
        return []

    # Sort by similarity descending and take top max_pairs
    sims = sim_matrix[i_idx, j_idx]
    order = np.argsort(sims)[::-1][:max_pairs]
    return [
        {'from': concepts[int(i_idx[k])], 'to': concepts[int(j_idx[k])],
         'similarity': round(float(sims[k]), 4)}
        for k in order
    ]


# ============================================================
# MCP TOOL REGISTRATION
# ============================================================

def register_tools(mcp):
    """Register hippocampal index MCP tools."""

    @mcp.tool()
    def binary_recall_tool(query: str, max_results: int = 5) -> str:
        """FHRR hippocampal index query. Fast session localization (~10ms, 0 tokens).

        Returns sessions where the topic was discussed, ranked by relevance.
        Complements recall() which returns processed knowledge.

        Args:
            query: Natural language query
            max_results: Max results to return (default 5)
        """
        t0 = time.monotonic()
        results = binary_recall(query, max_results=max_results)
        elapsed_ms = (time.monotonic() - t0) * 1000

        if not results:
            return json.dumps({
                'status': 'no_hits',
                'query': query,
                'elapsed_ms': round(elapsed_ms, 1),
                'message': 'No matching sessions found in FHRR index.',
            })

        formatted = []
        for r in results:
            formatted.append({
                'session_id': r['session_id'],
                'score': r['score'],
                'chunk_score': r['chunk_score'],
                'matched_roles': r['matched_roles'],
                'preview': r['context']['texts'][0][:200] if r['context']['texts'] else '',
            })

        return json.dumps({
            'status': 'ok',
            'query': query,
            'num_results': len(formatted),
            'elapsed_ms': round(elapsed_ms, 1),
            'results': formatted,
        }, ensure_ascii=False)

    @mcp.tool()
    def compile_fhrr_index() -> str:
        """Compile FHRR hot index for fast binary_recall (<50ms vs 22s).

        Reads session-level data from all NPZ/JSON files and builds
        pre-stacked numpy arrays for vectorized querying.
        Run after encoding new sessions or when binary_recall is slow.
        """
        t0 = time.monotonic()
        hot = compile_hot_index()
        elapsed = (time.monotonic() - t0) * 1000
        if hot:
            index_size = os.path.getsize(HOT_INDEX_PATH) / (1024 * 1024)
            return json.dumps({
                'status': 'compiled',
                'sessions': hot.num_sessions,
                'compile_ms': round(elapsed, 1),
                'index_size_mb': round(index_size, 2),
                'path': HOT_INDEX_PATH,
            })
        return json.dumps({'status': 'no_sessions'})

    @mcp.tool()
    def fhrr_index_stats() -> str:
        """Show FHRR hippocampal index statistics."""
        stats = get_index_stats()
        size_mb = stats['total_size_bytes'] / (1024 * 1024) if stats['total_size_bytes'] else 0
        return json.dumps({
            **stats,
            'total_size_mb': round(size_mb, 2),
            'avg_size_per_session_kb': round(
                stats['total_size_bytes'] / max(1, stats['num_sessions']) / 1024, 1
            ),
        })
