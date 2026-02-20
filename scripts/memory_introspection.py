"""
MEMORY INTROSPECTION ANALYSIS
==============================
Codi's self-analysis: What patterns exist in active vs dormant memories?
Run: python scripts/memory_introspection.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from collections import Counter, defaultdict
from modules.config import COLLECTION_NAME, qdrant
from modules.sharpe import compute_sharpe_for_payload
from modules.sharpe_insights import domain_of

def scroll_all_memories():
    """Scroll all episodic memories from Qdrant."""
    all_mems = []
    offset = None
    while len(all_mems) < 600:
        pts, next_offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            with_payload=True,
            offset=offset,
        )
        if not pts:
            break
        for p in pts:
            payload = p.payload or {}
            data = payload.get("data", "")
            if not data or len(data) < 10:
                continue
            result = compute_sharpe_for_payload(payload)
            themes = payload.get("narrative_themes") or []
            domains = domain_of(themes)
            all_mems.append({
                "id": str(p.id),
                "sharpe": result["sharpe"],
                "utility": result["utility"],
                "uncertainty": result["uncertainty"],
                "acc": result["components"]["access_count"],
                "importance": result["components"]["importance"],
                "evidence": result["components"]["evidence_count"],
                "themes": themes,
                "domains": domains,
                "preview": data[:100],
                "created_at": payload.get("created_at", ""),
            })
        if not next_offset:
            break
        offset = next_offset
    return all_mems


def analyze(mems):
    active = [m for m in mems if m["acc"] > 0]
    dormant = [m for m in mems if m["acc"] == 0]

    lines = []
    lines.append("=" * 70)
    lines.append("MEMORY INTROSPECTION: Active vs Dormant")
    lines.append("=" * 70)
    lines.append(f"\nTotal: {len(mems)} | Active (acc>0): {len(active)} ({len(active)/len(mems)*100:.1f}%) | Dormant: {len(dormant)} ({len(dormant)/len(mems)*100:.1f}%)")

    # --- ACTIVE MEMORIES ANALYSIS ---
    lines.append("\n" + "=" * 50)
    lines.append("ACTIVE MEMORIES (acc > 0)")
    lines.append("=" * 50)

    # Access distribution
    acc_values = [m["acc"] for m in active]
    lines.append(f"\nAccess count distribution:")
    lines.append(f"  Min: {min(acc_values)}, Max: {max(acc_values)}, Avg: {sum(acc_values)/len(acc_values):.1f}")
    acc_buckets = Counter()
    for a in acc_values:
        if a == 1:
            acc_buckets["1 (single access)"] += 1
        elif a <= 3:
            acc_buckets["2-3"] += 1
        elif a <= 5:
            acc_buckets["4-5"] += 1
        elif a <= 10:
            acc_buckets["6-10"] += 1
        else:
            acc_buckets["10+"] += 1
    for bucket in sorted(acc_buckets.keys()):
        lines.append(f"    {bucket}: {acc_buckets[bucket]}")

    # Importance distribution (active)
    lines.append(f"\nImportance (active):")
    imp_active = Counter(m["importance"] for m in active)
    for imp in ["critical", "high", "medium", "low"]:
        count = imp_active.get(imp, 0)
        pct = count / len(active) * 100 if active else 0
        lines.append(f"  {imp}: {count} ({pct:.0f}%)")

    # Domain distribution (active)
    lines.append(f"\nDomains (active):")
    domain_active = Counter()
    no_domain_active = 0
    for m in active:
        if m["domains"]:
            for d in m["domains"]:
                domain_active[d] += 1
        else:
            no_domain_active += 1
    for d, count in domain_active.most_common():
        lines.append(f"  {d}: {count}")
    lines.append(f"  (no domain): {no_domain_active}")

    # Multi-domain active memories (bridges)
    multi_domain_active = [m for m in active if len(m["domains"]) >= 2]
    lines.append(f"\nMulti-domain active: {len(multi_domain_active)}")
    for m in multi_domain_active[:5]:
        lines.append(f"  [{m['sharpe']:.2f}] acc={m['acc']} domains={m['domains']} | {m['preview'][:60]}...")

    # Sharpe distribution (active)
    sharpes_active = [m["sharpe"] for m in active]
    lines.append(f"\nSharpe (active): Avg={sum(sharpes_active)/len(sharpes_active):.2f}, Min={min(sharpes_active):.2f}, Max={max(sharpes_active):.2f}")

    # --- DORMANT MEMORIES ANALYSIS ---
    lines.append("\n" + "=" * 50)
    lines.append("DORMANT MEMORIES (acc = 0)")
    lines.append("=" * 50)

    # Importance distribution (dormant)
    lines.append(f"\nImportance (dormant):")
    imp_dormant = Counter(m["importance"] for m in dormant)
    for imp in ["critical", "high", "medium", "low"]:
        count = imp_dormant.get(imp, 0)
        pct = count / len(dormant) * 100 if dormant else 0
        lines.append(f"  {imp}: {count} ({pct:.0f}%)")

    # Domain distribution (dormant)
    lines.append(f"\nDomains (dormant):")
    domain_dormant = Counter()
    no_domain_dormant = 0
    for m in dormant:
        if m["domains"]:
            for d in m["domains"]:
                domain_dormant[d] += 1
        else:
            no_domain_dormant += 1
    for d, count in domain_dormant.most_common():
        lines.append(f"  {d}: {count}")
    lines.append(f"  (no domain): {no_domain_dormant}")

    # Sharpe distribution (dormant)
    sharpes_dormant = [m["sharpe"] for m in dormant]
    lines.append(f"\nSharpe (dormant): Avg={sum(sharpes_dormant)/len(sharpes_dormant):.2f}, Min={min(sharpes_dormant):.2f}, Max={max(sharpes_dormant):.2f}")

    # Dormant but critical/high importance (potentially sleeping giants)
    sleeping_giants = [m for m in dormant if m["importance"] in ("critical", "high")]
    lines.append(f"\nSleeping Giants (dormant + critical/high): {len(sleeping_giants)}")
    sleeping_giants.sort(key=lambda x: -x["sharpe"])
    for m in sleeping_giants[:10]:
        lines.append(f"  [{m['sharpe']:.2f}] imp={m['importance']} domains={m['domains']} | {m['preview'][:60]}...")

    # --- CROSS-ANALYSIS ---
    lines.append("\n" + "=" * 50)
    lines.append("CROSS-ANALYSIS: What makes a memory get used?")
    lines.append("=" * 50)

    # Compare importance distributions
    lines.append(f"\nImportance -> Access Rate:")
    for imp in ["critical", "high", "medium", "low"]:
        total_imp = sum(1 for m in mems if m["importance"] == imp)
        active_imp = sum(1 for m in active if m["importance"] == imp)
        rate = active_imp / total_imp * 100 if total_imp > 0 else 0
        lines.append(f"  {imp}: {active_imp}/{total_imp} ({rate:.0f}%) are active")

    # Compare domain distributions
    lines.append(f"\nDomain -> Access Rate:")
    all_domains = set()
    for m in mems:
        all_domains.update(m["domains"])
    for d in sorted(all_domains):
        total_d = sum(1 for m in mems if d in m["domains"])
        active_d = sum(1 for m in active if d in m["domains"])
        rate = active_d / total_d * 100 if total_d > 0 else 0
        lines.append(f"  {d}: {active_d}/{total_d} ({rate:.0f}%) are active")

    # Theme analysis
    lines.append(f"\nTheme -> Access Rate (top 10):")
    theme_total = Counter()
    theme_active = Counter()
    for m in mems:
        for t in m["themes"]:
            theme_total[t] += 1
            if m["acc"] > 0:
                theme_active[t] += 1
    for theme, total in theme_total.most_common(15):
        act = theme_active.get(theme, 0)
        rate = act / total * 100 if total > 0 else 0
        lines.append(f"  {theme}: {act}/{total} ({rate:.0f}%)")

    # --- RECURSIVE INSIGHT ---
    lines.append("\n" + "=" * 50)
    lines.append("RECURSIVE INSIGHT: Top-5 by Sharpe - What am I valuing most?")
    lines.append("=" * 50)

    top5 = sorted(mems, key=lambda x: -x["sharpe"])[:5]
    for i, m in enumerate(top5, 1):
        lines.append(f"\n  #{i} [Sharpe={m['sharpe']:.2f}] acc={m['acc']} imp={m['importance']}")
        lines.append(f"     Domains: {m['domains']}")
        lines.append(f"     Content: {m['preview']}...")

    # Concentration: how many unique memories make up 80% of total access
    lines.append("\n" + "=" * 50)
    lines.append("ACCESS CONCENTRATION (Pareto analysis)")
    lines.append("=" * 50)
    active_sorted = sorted(active, key=lambda x: -x["acc"])
    total_access = sum(m["acc"] for m in active)
    cumulative = 0
    for i, m in enumerate(active_sorted):
        cumulative += m["acc"]
        if cumulative >= total_access * 0.80:
            lines.append(f"  80% of all access comes from {i+1}/{len(active)} memories ({(i+1)/len(active)*100:.0f}%)")
            break
    lines.append(f"  Total access events: {total_access}")
    lines.append(f"  Top accessor: acc={active_sorted[0]['acc']} | {active_sorted[0]['preview'][:60]}...")

    return "\n".join(lines)


if __name__ == "__main__":
    print("Scrolling all memories...")
    mems = scroll_all_memories()
    print(f"Loaded {len(mems)} memories. Analyzing...\n")
    report = analyze(mems)
    print(report)

    # Save to file
    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "memory_introspection_report.txt")
    with open(out_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {out_path}")
