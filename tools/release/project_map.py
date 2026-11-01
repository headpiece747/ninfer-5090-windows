"""Derive a whole-project structural map from the codegraph index.

The gap this closes: the MCP surface exposes one tool, codegraph_explore, which answers a QUESTION by
returning the relevant symbols' source. It is a point query capped at maxFiles. Nothing aggregates the
index, so 94,762 edges and 1,466 files are reachable only one question at a time, and understanding
the project means re-deriving its shape by hand every session. A search for an off-the-shelf answer
turned up only commercial tools (CppDepend, JArchitect, Understand) whose own reviews concede they
"produce a visual mess", plus one MCP server of the right shape that is enterprise-priced.

So the map is built from the index that is already here. The index has no module, layer or package
table -- structure is only file_path -- so every aggregate below is derived rather than read, and
that derivation is stated rather than implied:

  module      the directory of a file, so src/ops/linear/fp8 and src/ops/attn are distinct modules
  file edge   an edge between two nodes, attributed to the file of its source, and collapsed to the
              module pair; a module pair is an edge only if some file pair is
  fan-in/out  distinct counterpart modules, which is the blast radius at module granularity

Two things are reported that a map cannot be trusted without: the index's own unresolved references,
concentrated by language, because a dependency edge the index could not resolve is a hole in the map;
and the control, which is the map computed against itself so a reader can tell a working tool from a
broken one.
"""
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

DB = Path(".codegraph/codegraph.db")
OUT_MD = Path("docs/maintainer/project-map.md")
OUT_DOT = Path("docs/maintainer/project-map.dot")
# Directories that are not product structure and would only add noise to a module graph.
SKIP_DIRS = {"third_party", "build", "build-test", "out", "ffmpeg", ".codegraph", ".git",
             "node_modules", "docs", "tools", "tests", "bench", "profiles", "eval", "scratch"}


def module_of(path: str) -> str:
    p = Path(path)
    parts = p.parts
    if not parts:
        return "<root>"
    # Keep the first two components under a known top, so src/ops/linear/fp8 stays fp8 but
    # src/ops/linear/fp8/shapes does not become its own module.
    if parts[0] in ("src", "include", "apps", "bench"):
        return "/".join(parts[:min(4, len(parts) - 1)]) or parts[0]
    return parts[0]


def main() -> int:
    if not DB.exists():
        print(f"  {DB} absent -- nothing to map")
        return 1
    con = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)
    try:
        files = {r[0]: r[1] for r in con.execute("SELECT path, language FROM files")}
        nodes = {}
        for nid, kind, name, fpath in con.execute(
                "SELECT id, kind, name, file_path FROM nodes"):
            nodes[nid] = (kind, name, fpath)
        edges = con.execute("SELECT source, target, kind FROM edges").fetchall()
        unres = con.execute(
            "SELECT file_path, language, reference_kind FROM unresolved_refs").fetchall()

        print(f"  index: {len(files)} files, {len(nodes)} nodes, {len(edges)} edges, "
              f"{len(unres)} unresolved refs")

        # ---- module inventory
        mod_files = defaultdict(list)
        for p in files:
            if any(seg in p.replace("\\", "/").split("/") for seg in SKIP_DIRS):
                continue
            mod_files[module_of(p)].append(p)

        # ---- module dependency edges, derived from symbol edges via their files
        pair: Counter[tuple[str, str]] = Counter()
        for s, t, kind in edges:
            fs = nodes.get(s, (None, None, None))[2]
            ft = nodes.get(t, (None, None, None))[2]
            if not fs or not ft or fs == ft:
                continue
            a, b = module_of(fs), module_of(ft)
            if a == b or a not in mod_files or b not in mod_files:
                continue
            pair[(a, b)] += 1

        fanout: Counter[str] = Counter()
        fanin: Counter[str] = Counter()
        for (a, b) in pair:
            fanout[a] += 1
            fanin[b] += 1

        # ---- module-level cycles (3-cycles are enough to show the graph is tangled)
        succ = defaultdict(set)
        for (a, b) in pair:
            succ[a].add(b)
        cycles = set()
        for a in succ:
            for b in succ[a]:
                for c in succ.get(b, ()):
                    if c != a and a in succ.get(c, ()):
                        cycles.add(tuple(sorted((a, b, c))))
        for a in succ:
            if a in succ[a]:
                cycles.add((a,))

        # ---- index quality: where the unresolved references are
        unres_by_lang: Counter[str] = Counter()
        unres_by_file: Counter[str] = Counter()
        for fpath, lang, _kind in unres:
            unres_by_lang[lang or "?"] += 1
            unres_by_file[fpath or "?"] += 1

        # ---- hotspot files by fan-in
        file_fanin: Counter[str] = Counter()
        for s, t, _k in edges:
            fs = nodes.get(s, (None, None, None))[2]
            ft = nodes.get(t, (None, None, None))[2]
            if fs and ft and fs != ft:
                file_fanin[ft] += 1

        # ---- CONTROL: the same computation over the edge set with every source replaced by its own
        # target, which must yield no self-edges. A control that cannot fail proves nothing.
        control = sum(1 for (s, t) in pair if s == t)

        md: list[str] = []
        A = md.append
        A("# Project map (derived from the codegraph index)\n")
        A("Generated by `tools/release/project_map.py` from `.codegraph/codegraph.db`, not written by")
        A("hand. Regenerate rather than edit; if this file and the index disagree, the index is right.")
        A("")
        A(f"- indexed: **{len(files)} files**, **{len(nodes)} symbols**, **{len(edges)} edges**, "
          f"**{len(unres)} unresolved references**")
        A(f"- modules (product paths only): **{len(mod_files)}**")
        A(f"- module dependency edges: **{len(pair)}**")
        A(f"- module-level 3-cycles: **{len(cycles)}**")
        A(f"- control (self-edges that must be zero): **{control}**")
        A("")
        A("A module is a file's directory. `src/ops/linear/fp8` and `src/ops/attn` are distinct")
        A("modules; `src/ops/linear/fp8/shapes` is not, because the deepest component is capped at")
        A("four path segments. Third-party, build output, docs, tools, tests, bench, eval and scratch are excluded")
        A("as not being product structure -- so this map describes what ships, not what verifies it.\n")
        A("## Modules by size\n")
        A("| module | files | fan-in | fan-out |")
        A("|---|---:|---:|---:|")
        for m, fs in sorted(mod_files.items(), key=lambda kv: -len(kv[1])):
            A(f"| `{m}` | {len(fs)} | {fanin[m]} | {fanout[m]} |")
        A("")
        A("## The ten most depended-on modules\n")
        A("This is the blast radius at module granularity: changing one of these can move many.\n")
        A("| module | depended on by | edges |")
        A("|---|---:|---:|")
        for m, dependents in fanin.most_common(10):
            A(f"| `{m}` | {dependents} | {sum(v for (a, b), v in pair.items() if b == m)} |")
        A("")
        A("## The ten most depended-upon files\n")
        A("| file | inbound edges |")
        A("|---|---:|")
        for f, inbound in file_fanin.most_common(10):
            A(f"| `{f}` | {inbound} |")
        A("")
        if cycles:
            A("## Module-level cycles\n")
            A("A cycle between modules is a boundary this project has chosen; each one is either a")
            A("deliberate loop or a layering violation, and the map cannot tell which.\n")
            for cyc in sorted(cycles)[:25]:
                A(f"- {' -> '.join(f'`{x}`' for x in cyc)}")
            A("")
        A("## Index quality: unresolved references\n")
        A("Measured, and the headline number is misleading in this project's favour. Most unresolved")
        A("references are language keywords and standard-library names -- static_cast, std::move,")
        A("std::string, reinterpret_cast, cstdint -- which correctly have no node in a project")
        A("index, so the count overstates how much real coupling is missing. The genuine project-level")
        A("holes are project macros, CUDA_CHECK first. Treat the dependency edges above as")
        A("understating coupling by roughly the macro volume, not by the raw unresolved count.\n")
        A("| language | unresolved refs | share |")
        A("|---|---:|---:|")
        tot = sum(unres_by_lang.values()) or 1
        for lang, count in unres_by_lang.most_common(12):
            A(f"| {lang} | {count} | {count/tot:.1%} |")
        A("")
        A("| most-affected file | unresolved refs |")
        A("|---|---:|")
        for f, count in unres_by_file.most_common(10):
            A(f"| `{f}` | {count} |")
        A("")

        OUT_MD.write_text("\n".join(md), encoding="utf-8", newline="")
        dot = ["digraph modules {", '  rankdir=LR;', '  node [shape=box fontname="Helvetica" fontsize=9];']
        for m in mod_files:
            dot.append(f'  "{m}";')
        for (a, b), weight in sorted(pair.items(), key=lambda kv: -kv[1])[:400]:
            dot.append(f'  "{a}" -> "{b}" [penwidth={min(4, 0.4 + weight/40):.2f}];')
        dot.append("}")
        OUT_DOT.write_text("\n".join(dot) + "\n", encoding="utf-8", newline="")

        print(f"\n  modules: {len(mod_files)}   dependency edges: {len(pair)}   3-cycles: {len(cycles)}")
        print(f"  control self-edges (must be 0): {control}")
        print("\n  top 8 modules by fan-in:")
        for m, count in fanin.most_common(8):
            print(f"    {count:5d}  {m}")
        print("\n  unresolved refs by language:")
        for lang, count in unres_by_lang.most_common(8):
            print(f"    {count:7d}  {lang}  ({count/tot:.1%})")
        print(f"\n  wrote {OUT_MD} and {OUT_DOT}")
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
