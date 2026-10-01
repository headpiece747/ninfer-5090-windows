# Prior art for documentation-drift detection

Research subagent deliverable. Scope: what tooling and prior practice exists for three
classes of documentation drift, and what the literature measures about the underlying
problem. Primary sources only (project repositories, official docs, published papers),
each cited.

Constraints honoured: **no recommendations are made.** Every item reports what exists, what
it checks, and what it does not check, tagged as *maintained* / *unmaintained* / *does not
exist*.

Method note: every tool's behaviour claim below was read out of that tool's own source or
official documentation, not from a comparison table or a blog post. Where a tool ships a
comparison table (lychee does), the claim was re-derived from the tool's code.

---

## Headline findings

1. **No maintained tool validates a `path:line` citation in prose.** Every link checker
   surveyed (lychee, linkinator, markdown-link-check, remark-validate-links, markdownlint)
   either never extracts a bare `path:line` from prose or — where it does parse a
   line-shaped fragment — deliberately *discards* it without checking. This is a clean
   negative finding.
2. **No tool verifies that a cited line number is still within the file's range.** One
   tool (markdownlint MD051) recognises the `#L20` syntax well enough to *exclude* it from
   validation, which is a deliberate design choice, not an oversight.
3. **The Linux kernel — the closest comparable — has exactly one script for this class
   (`tools/docs/documentation-file-ref-check`), and it checks file *existence* only, only
   for `Documentation/` paths and Sphinx `:doc:` roles, and never a line number.** It is
   `git grep` plus `glob`. That is the whole mechanism.
4. **Class 1 (prose count vs authoritative source) has exactly one strong, maintained
   instance of prior art: Rust's `tidy` `unstable_book` check**, which cross-checks a
   prose book's section filenames against a structured metadata file in both directions.
   It does not check numbers embedded in sentences, and its reverse direction is
   commented out.
5. **Class 3 (comment asserts a mechanism the code disproves) is not a lint class
   anywhere.** The established technique is not detection but *substitution*: bind the
   claim to a `static_assert`, or make the document literally the file
   (`#[doc = include_str!]`). LLVM's `SmallVector.h` is a readable instance of the former.
6. **Decay rates are measured, and they are large.** Tan et al. (EMSE, from
   arXiv:2212.01479) found **82.3 % (658/800)** of the 800 most popular GitHub projects had
   at least one outdated code-element reference at some point in history, **28.9 %** of
   them were outdated *at the time of measurement*, and outdated references had roughly a
   **55 % chance of still being present one month later**. Wen et al. (ICPC 2019) found
   **only 13–20 % of code changes trigger a comment update at all**. Rani et al.'s SLR of
   2353 papers found only **4 of 21 identified quality attributes** are commonly studied,
   with "consistency" dominant, and that the field still "rel[ies] on manual assessment and
   specific heuristics rather than automated assessment."

---

## Part A — Existing tooling, per class

### A.1 Markdown / link checkers: do any validate `path:line` into source files?

**Short answer: no. Not one of the six.**

#### lychee — maintained (★3966, last push 2026-09-28)

- Scope: "A fast, async, stream-based link checker … Finds broken hyperlinks and mail
  addresses in websites and Markdown, HTML, and other file formats!"
  ([README](https://github.com/lycheeverse/lychee/blob/master/README.md)). Ships as CLI,
  as `lychee-lib`, and as `lycheeverse/lychee-action`.
- Local-file validation exists and is real: `lychee-lib/src/checker/file.rs` resolves a
  `file://` URI, applies `--fallback-extensions`, applies `--index-files` for directory
  links, and then `check_file`. Source:
  <https://raw.githubusercontent.com/lycheeverse/lychee/master/lychee-lib/src/checker/file.rs>.
- **What it does not do:** the fragment checker is scoped to **HTML only**. From the
  `FileChecker` doc comment: *"checks if the file exists, and optionally checks for the
  existence of fragments in HTML files."* A `.cu`, `.py`, `.json` or `.md` file is checked
  for existence and nothing else.
- **What it does not do:** its markdown extractor (`lychee-lib/src/extract/markdown.rs`)
  only emits a `RawUri` from actual link constructs — `Tag::Link`, autolinks, wikilinks,
  HTML attributes, CSS urls. A bare `` `src/runtime/foo.cpp:42` `` in a prose sentence or a
  code span is never a `Link` event, so it is never checked. Plaintext fallback uses
  `linkify` (`lychee-lib/src/extract/plaintext.rs` → `url::find_links`), which extracts
  URLs, not filesystem paths.
- `--include-verbatim` (`lychee --help`) extends extraction into `pre-`/`code` blocks — but
  still only to *link syntax*, not to bare paths.
- Relevant knobs that exist and are *not* the needed one: `--include-fragments`
  (`none|anchor-only|text-only|full`, default off), `--root-dir`, `--base-url`,
  `--fallback-extensions`, `--index-files`.

#### linkinator — maintained (★1269, last push 2026-09-29)

- Scope: a crawler. "Supports redirects, absolute links, relative links, all the things";
  `--markdown` parses markdown when scanning from disk
  ([README](https://github.com/JustinBeckwith/linkinator/blob/main/README.md)).
- `--check-fragments` exists: "Validate fragment identifiers (URL anchors like
  `#section-name`) exist on the target **HTML page**. … Only checks server-rendered HTML
  (not JavaScript-added fragments)." Default **false**.
- Local scanning stands up a temporary static web server and crawls it as HTTP. A
  `file:line` in prose is not a link target it would ever see.
- **What it does not do:** no source-file reference validation, no line-number validation.

#### markdown-link-check — maintained (★718, v3.15.0, last push 2026-07-28)

- Scope, stated plainly in its README: "Extracts links from markdown texts and checks
  whether each link is alive (`200 OK`) or dead. `mailto:` links are also validated."
  This is an **HTTP liveness** checker.
- Config surface: `aliveStatusCodes`, `retryCount`, `ignorePatterns`,
  `replacementPatterns`, `projectBaseUrl`, `timeout`, `httpHeaders`. No file-existence
  check, no anchor check, no line check.
- Escape hatches are comment markers only: `<!-- markdown-link-check-disable -->`,
  `-disable-next-line`, `-disable-line`.

#### remark-validate-links — **unmaintained-ish** (★126, last push **2025-02-21**, ~18
months stale relative to every tool above)

- Scope: "check that markdown links and images point to existing local **files and
  headings** in a Git repo" ([readme](https://github.com/remarkjs/remark-validate-links/blob/main/readme.md)).
  Offline; deliberately Git-repo-specific.
- It *does* resolve local files and headings in other markdown documents
  (`missing-file`, `missing-heading`, `missing-heading-in-file` messages).
- **Decisive negative finding.** `lib/index.js` line 181 defines
  `const lineExpression = /^#l\d/i`, and `normalize()` at lines 778–786:

  ```js
  // Ignore the hash if it references lines in a file or doesn’t start
  // with a heading prefix.
  else if (
    prefix &&
    ((lines && lineExpression.test(hash)) ||
      hash.slice(0, prefix.length) !== prefix)
  ) {
    hash = undefined
  }
  ```

  A line-shaped fragment is *dropped from the check*, by design. (`lineLinks` defaults
  `{github: true, gitlab: true}`.) Source:
  <https://raw.githubusercontent.com/remarkjs/remark-validate-links/main/lib/index.js>.
- It also does not read non-markdown files for anchors, and its Node API (as opposed to the
  CLI) does not check headings in other files at all.

#### markdownlint — maintained (★6363, last push 2026-10-01)

- Scope: markdown *style* rules (`MD001`–`MD059`), not link liveness. No network checks at
  all.
- `MD051` ("link-fragments") is the closest any tool gets to line references, and it
  operates **within a single document** — it builds the set of fragments from that
  document's own headings, `id` attributes and `<a name>` attributes. See
  <https://raw.githubusercontent.com/DavidAnson/markdownlint/main/doc/md051.md>.
- **Decisive negative finding.** `lib/md051.mjs` line 11:
  `const lineFragmentRe = /^#(?:L\d+(?:C\d+)?-L\d+(?:C\d+)?|L\d+)$/`, used at line 129 as
  `!lineFragmentRe.test(encodedText)` inside the "is this fragment missing?" test. The
  effect is that a `#L20` or `#L19C5-L21C11` fragment is **never reported**, regardless of
  whether line 20 exists or whether the document has 20 lines. Source:
  <https://raw.githubusercontent.com/DavidAnson/markdownlint/main/lib/md051.mjs>.
- `MD052` checks that reference-style links have a defined label. Not related.

#### vale — maintained (★6181, last push 2026-10-01)

- Scope: "A command-line linter for prose." It "parses your markup instead of guessing at
  it" and "lifts comments and docstrings out of source code with tree-sitter grammars, so a
  comment marker inside a string literal stays code"
  ([README, v3](https://github.com/errata-ai/vale/blob/v3/README.md)). That code-comment
  awareness is real and relevant to class 3, though it lints *style*, not truthfulness.
- Rule types live in `internal/check/`: `action, anchor, capitalization, conditional,
  consistency, definition, editops, existence, expand, filter, inherit, matchcase, metric,
  occurrence, readability, repetition, scope, script, sequence, spellfilter, spelling,
  substitution, upos, variables`
  (<https://api.github.com/repos/errata-ai/vale/contents/internal/check?ref=v3>).
- **Boundary that matters for class 1:** `internal/check/script.go` compiles a rule body as a
  [Tengo](https://github.com/d5/tengo) script, and the import set is closed:
  `program.SetImports(stdlib.GetModuleMap("text", "fmt", "math"))`. There is **no `os`
  and no `fs` module**. A Vale rule cannot open the authoritative JSON/CSV to compare
  against. It sees the text block it is linting and nothing else.

#### codespell — maintained (★2430, last push 2026-09-28)

- Scope, from its own README: "Fix common misspellings in text files. It's designed
  primarily for checking misspelled words in source code (backslash escapes are skipped)"
  (<https://raw.githubusercontent.com/codespell-project/codespell/master/README.rst>).
- "It does not check for word membership in a complete dictionary, but instead looks for a
  set of common misspellings." Shipped dictionaries are "an improved version of the one
  available on Wikipedia … applied in projects like Linux Kernel, EFL, oFono."
- **What it does not do:** anything structural. No paths, no numbers, no references.

#### Verdict for A.1

| Tool | checks local file exists | checks `#anchor` into md | checks `#anchor` into source | checks `path:line` | checks line in range |
|---|---|---|---|---|---|
| lychee | yes | yes (`--include-fragments`) | no (HTML only) | no | no |
| linkinator | no (HTTP crawl) | yes, HTML only | no | no | no |
| markdown-link-check | no | no | no | no | no |
| remark-validate-links | yes | yes (md headings) | no | no — **deliberately ignored** | no |
| markdownlint MD051 | no | yes (same doc only) | no | no — **deliberately ignored** | no |
| vale | no | no | no | no | no |
| codespell | no | no | no | no | no |

A comment-identifier `alignas(64)` claim, or a sentence asserting "the test suite is 135
tests", is invisible to all six. **Class 1 and class 2 have no off-the-shelf solution.**

---

### A.2 General documentation-drift detection: "prose claims X, code says Y"

This is the class where the prior art is thinnest and most academic.

- **Rust `tidy`** (in-tree, `src/tools/tidy/src/`, part of the compiler repo, ★119406,
  pushed 2026-10-01) is the only tool found that performs a general two-way cross-check
  between prose and a structured source of truth. See D.1 below for the mechanism, which
  is the strongest prior art for class 1.
- **`rustdoc`** — main maintained mechanism for *references*, not counts.
  `rustdoc::broken_intra_doc_links` "**warns by default**. This lint detects when an
  intra-doc link fails to be resolved", and is `deny`-able:
  <https://doc.rust-lang.org/rustdoc/lints.html>. Note the doc's own caveat: "except for
  `missing_docs`, these lints are only available when running `rustdoc`, not `rustc`."
  So this is a *doc tool* check, not a compile check.
- **Sphinx nitpicky mode** — `nitpicky = True` (default `False`): "Enables nitpicky mode if
  `True`. In nitpicky mode, Sphinx will warn about *all* references where the target cannot
  be found. This is recommended for new projects as it ensures that all references are to
  valid targets." Escape hatches `nitpick_ignore` and `nitpick_ignore_regex` mark missing
  references as "known missing".
  <https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-nitpicky>. Also
  available per-invocation as `--nitpicky` / `-n`
  (<https://www.sphinx-doc.org/en/master/man/sphinx-build.html>).
  **What it does not do:** resolve a reference into a *source file*; it resolves Sphinx
  domain targets (labels, Python objects, C domain entities), not `file.c:42`.
- **Doxygen** — maintained (★6588, pushed 2026-09-30). Provides
  `WARN_IF_UNDOCUMENTED`, `WARN_NO_PARAMDOC`, `WARN_IF_DOC_ERROR`,
  `WARN_IF_INCOMPLETE_DOC`, and `WARN_AS_ERROR` (whose own value space includes
  `FAIL_ON_WARNINGS`; Doxygen's own `doc/Doxyfile` sets `WARN_AS_ERROR = FAIL_ON_WARNINGS`,
  so upstream dogfoods it). **What it does not do:** detect a *wrong* claim. Every one of
  those knobs is about *absence* (undocumented parameter, undocumented member), never
  *falsity*. `\anchor` and `\ref` create resolvable anchors inside the generated docs; a
  broken `\ref` is a warning, which `WARN_AS_ERROR` can promote.
- **Javadoc `@link` / `-Xdoclint`** — the equivalent Java mechanism; resolves references
  between documented elements. Same limitation: reference resolution, not truth.
- **comment-testing / doc-test frameworks where the prose is executed.** These exist and are
  maintained, and they are the closest thing to a compiler-enforced prose claim for
  *behaviour*:
  - `mdBook test` — "mdBook supports a `test` command that will run all available tests in a
    book. At the moment, only Rust tests are supported." Rust code blocks are compiled and
    run; `rust,ignore` and non-Rust languages are skipped; an unlabelled block **is** tested.
    <https://raw.githubusercontent.com/rust-lang/mdBook/master/guide/src/cli/test.md>.
  - `sphinx.ext.doctest` — directives `testsetup`, `testcode`, `testoutput`, `doctest`,
    with `:skipif:`, `:options:`, `doctest_test_doctest_blocks`.
    <https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html>.
  - `pytest --doctest-glob` for `.md`/`.rst`.
  - **What none of them do:** execute or check a *prose assertion about code state*. They
    check that a code example still compiles/runs. A sentence claiming "the suite is 137
    tests" is not code and is never executed.
- **GitHub repository search** for tools in this space returned essentially nothing
  durable. Searches run: `detect stale documentation references code` (0 relevant),
  `documentation code consistency checker` (only a 2-star gem and a 0-star
  non-project), `doc-comment consistency checker` (0), `check documentation references
  source code symbols` (0), `topic:documentation topic:linter stale` (6 results, all
  0–5 stars; the largest, `AloizioMacedo/pystaleds` ★5, checks Python docstrings for
  stale-ness). **These are experiments, not prior art.**

---

### A.3 Keeping comments truthful: where the COMPILER enforces a comment claim

This is the most important section for class 3, because the answer is that the field's
answer is *not detection*.

#### The established technique: bind the claim to a `static_assert`

The idiom is documented next to the assertion, and the compiler enforces the invariant
while the comment carries the rationale and the consequence. LLVM's
`llvm/include/llvm/ADT/SmallVector.h` is a clean instance (lines 1185–1230):

```cpp
template <typename T> struct CalculateSmallVectorDefaultInlinedElements {
  // Parameter controlling the default number of inlined elements
  // for `SmallVector<T>`.
  //
  // The default number of inlined elements ensures that
  // 1. There is at least one inlined element.
  // 2. `sizeof(SmallVector<T>) <= kPreferredSmallVectorSizeof` unless
  // it contradicts 1.
  static constexpr size_t kPreferredSmallVectorSizeof = 64;

  // static_assert that sizeof(T) is not "too big".
  // ... [rationale, including the portability caveat] ...
  static_assert(
      sizeof(T) <= 256,
      "You are trying to use a default number of inlined elements for "
      "`SmallVector<T>` but `sizeof(T)` is really big! Please use an "
      "explicit number of inlined elements with `SmallVector<T, N>` to make "
      "sure you really want that much inline storage.");
```

Two properties worth naming, because they are exactly what the audit's failing comment
(`"inherits alignas(64)"`, where MSVC reports `__cplusplus == 199711` and the header gates
the attribute on `>= 201103L`, so it inherits 8) lacked:

- the claim is stated as a *rule*, not a measurement;
- the rule is enforced by a construct the compiler rejects on violation.

The `static_assert` message is itself a machine-readable statement of the claim. The
portability wrinkle is documented rather than hidden.

#### The other established technique: make the document *be* the file

Rust's `#[doc = include_str!(...)]` splices a file's contents into the rustdoc for a
module, so there is no second copy to drift. Live instances in the current tree
(`library/std/src/lib.rs`):

```rust
#![doc(rust_logo)]
#[doc = include_str!("../../portable-simd/crates/core_simd/src/core_simd_docs.md")]
pub mod simd { … }

#[doc = include_str!("../../core/src/autodiff.md")]
pub mod autodiff { … }

#[doc = include_str!("../../core/src/offload.md")]
pub mod gpu_offload { … }

#[doc = include_str!("../../stdarch/crates/core_arch/src/core_arch_docs.md")]
pub mod arch { … }
```

<https://raw.githubusercontent.com/rust-lang/rust/master/library/std/src/lib.rs> (lines
234, 664, 674, 681, 698). This is a *Rust-only* facility (`#[doc = …]` is an attribute);
C/C++ has no direct equivalent, though `include`-based literate systems are the same idea.

#### What a compiler *can* check in a comment, in C/C++

Clang's `-Wdocumentation` family is the one real compiler-side comment checker. Its
diagnostics, read from the diagnostic reference:

> `warning: parameter 'A' not found in the function declaration`
> `warning: template parameter 'A' not found in the template declaration`
> `warning: ' \ @ param' command used in a comment that is not attached to a function declaration`
> `warning: parameter 'A' is already documented`
> `warning: '\@ class interface protocol struct union' command should not be used in a comment attached to a non-class … declaration`
> `warning: '\@ function functiongroup method methodgroup callback' command should be used in a comment attached to a function … declaration`

<https://clang.llvm.org/docs/DiagnosticsReference.html#wdocumentation>. Also
`-Wdocumentation-deprecated-sync`, `-Wdocumentation-html`,
`-Wdocumentation-pedantic`, `-Wdocumentation-unknown-command`.

**Scope boundary:** this validates Doxygen *command syntax and parameter names against the
declaration*. It does not and cannot evaluate whether the described *behaviour* is what the
body does. GCC has no equivalent flag. MSVC (this project's toolchain) has no
documentation-comment validation warning as of VS 2022 / MSVC 14.5x — I found no primary
source for one.

#### Literate programming — the pre-compiler answer

It exists and is long-lived, with mixed maintenance:

- **noweb** — `nrnrnr/noweb` ★308, **last push 2026-08-10** (maintained). The tool Knuth
  designed for this: prose and code in one file, both extracted from the same source.
- **lhs2tex** — `kosmikus/lhs2tex` ★107, last push 2025-12-30 (maintained), but it is a
  *LaTeX typesetting* preprocessor for Haskell sources; the language it supports is the
  limiting factor, not the age.
- Others (`spockz/lhs2texhl` ★4, last push 2013) are unmaintained forks.

**What it does not do:** it makes prose and code *co-located and mechanically entangled*,
which prevents some drift classes, but it does not check that a prose claim about
behaviour is true, and it imposes a whole-file authoring discipline.

#### Comment-executing test frameworks

Searched GitHub for `comment testing framework execute code in comments` — total=2, both
irrelevant (an unrelated repo, a SQL-injection tool). **No maintained framework exists that
executes a C/C++ comment as a test.** The doc-test frameworks listed in A.2 (mdBook,
`sphinx.ext.doctest`, pytest `--doctest-glob`) are the closest and are all
Markdown/reStructuredText-oriented, not C/C++-comment-oriented.

---

### A.4 CI patterns: "docs must match code"

- The pattern that actually exists in large projects is **not** a generic action; it is a
  **project-local script invoked from a build system**, plus **`-W`/warning-as-error on the
  docs build**. Linux kernel: `Documentation/Makefile` line 10–11 runs
  `tools/docs/documentation-file-ref-check --warn` under `CONFIG_WARN_MISSING_DOCUMENTS`,
  and line 74–75 provides a standalone `refcheckdocs:` target. `linkcheckdocs` is the
  external-link target (and is *not* run by default; `dochelp` describes it as "check for
  broken external links / (will connect to external hosts)"). **There is no pre-built
  GitHub Action for this in the kernel**; there is no CI in the kernel at all for docs
  references, only make targets.
- GitHub Actions exist for the *link* half only: `lycheeverse/lychee-action`,
  `JustinBeckwith/linkinator-action`. Neither validates source references or counts.
- `pre-commit` ecosystem: `pre-commit/pre-commit-hooks` provides `check-json`,
  `check-yaml`, `check-toml`, `check-xml`, `trailing-whitespace`, `check-merge-conflict`,
  `check-vcs-permalinks`, etc. All **syntax**, none cross-referencing prose to code.
- **Does not exist:** any widely-used "docs must match code" CI action.

---

## Part B — What the literature says

Comment decay **is** measured, and the measurements are large enough to be uncomfortable.
The relevant numbers, each read from the paper or its abstract:

### B.1 Tan, Wagner & Treude — *Detecting Outdated Code Element References in Software
Repository Documentation*, EMSE (arXiv:2212.01479, submitted 2022-12-02)

The single most directly relevant paper: it targets exactly class 2 (documentation naming
code elements that no longer exist). Its method is a *history* analysis — a reference counts
as outdated when the element existed in source when the doc was last updated but the source
instances are now all gone.

Scale and rates:

- **82.3 % (658/800)** of the top-1000 GitHub projects, **40.7 % (2878/7071)** of documents,
  and **12.3 % (23588/191849)** of code-element references were outdated **at some point in
  history**.
- **28.9 %** of the most popular projects currently contain at least one outdated reference.
- On the 1907 Google projects: 29.7 % (567/1907) of projects, 30.6 % (925/3018) documents,
  7.1 % (4176/58805) references.
- **Decay rate: outdated references have "around 55 % chance of surviving in top1000 projects
  and 45 % in google projects after a month."** So roughly half of what is found broken
  remains broken a month later.
- **Recurrence:** 1.3 % (2431/191849) of references were outdated *again* after being fixed
  (0.4 % on the Google set) — i.e. fixing is not durable.
- **How it gets fixed:** of the 73.6 % (17368/23588) of top1000 references that were
  resolved, **47.6 % were "fixed" by changing the source code back** — not by fixing the
  documentation. Only 39.1 % were fixed by updating the doc, 13.3 % by deleting it.
- **False positives are real:** of 19 reported instances across 15 projects, 4 projects
  responded positively and 4 reported false positives; 7 had not responded at the time of
  writing.

Why this matters for the "is a lint worth building" question: it says (a) drift is
near-universal, (b) it is not self-healing, and (c) the largest single resolution path is
*changing the code to match the doc*, which is the opposite of what a doc-drift report is
trying to encourage. Sources:
<https://arxiv.org/abs/2212.01479>, full text <https://arxiv.org/html/2212.01479v1>.

### B.2 Wen, Nagy, Bavota & Lanza — *A Large-Scale Empirical Study on Code-Comment
Inconsistencies*, ICPC 2019 (DOI 10.1109/ICPC.2019.00019)

Scale: "1.3 Billion AST-level changes from the complete history of **1,500 systems**", plus
manual analysis of 500 commits to build a taxonomy (author preprint:
<https://csnagy.github.io/research/pdfs/2019/Wen2019-preprint.pdf>; published version
paywalled at IEEE and ACM DL).

The headline number is a negative one, read from the preprint's RQ1 answer:

> "We confirm previous findings in the literature, showing that **between 13% and 20% of
> code changes trigger comment updates**. This does not imply that in the remaining ~80% of
> cases code-comment inconsistencies are introduced, but they represent a possibility…"

And the paper is explicit about the field's structural limitation:

> "**Code-comment traceability is still an open problem.** … a major research challenge is
> the code-comment traceability (i.e., automatically identifying the code instructions
> documented by a given comment). As of today … popular programming languages are still
> bound to line and block comments. There are many research opportunities here both for
> language designers and researchers…"

Of the 500 candidate commits they hand-labelled, 138 were false positives and 362 were
"actually related to comment changes"; they identified 69 types of comment change, 25 of
which were relevant to inconsistencies. (Note the 27.6 % false-positive rate on that
corpus.)

### B.3 Rani, Blasi, Stulova, Panichella, Gorla & Nierstrasz — *A Decade of Code Comment
Quality Assessment: A Systematic Literature Review*, JSS 2023 (arXiv:2209.08165)

> "Our evaluation, based on the analysis of **2353 papers** and the actual review of **47
> relevant ones**, shows that (i) most studies and techniques focus on comments in **Java**
> code, thus may not be generalizable to other languages, and (ii) the analyzed studies
> focus on **four main QAs of a total of 21 QAs** identified in the literature, with a clear
> predominance of checking consistency between comments and the code. **We observe that
> researchers rely on manual assessment and specific heuristics rather than the automated
> assessment of the comment quality attributes.**"

Also: "several QAs are often assessed manually rather than with the automated approaches",
and "metrics are defined or used for only 10 QAs out of 21". Sources:
<https://arxiv.org/abs/2209.08165>, <https://arxiv.org/html/2209.08165v1>.

### B.4 Earlier and adjacent evidence

- **Chen, Lo et al., *iComment: Bugs or Bad Comments?*** (HSE, JHU) measured on Linux and
  Mozilla: "Linux contains about 1.0 million lines of comments for 5.0 million lines of
  source code, and Mozilla has 0.51 million lines of comments for 3.3 million lines of
  code". Their rule checker reported 98 inconsistencies, 60 true (33 new bugs, 27 bad
  comments) — a **38.8 % false-positive rate**, which they state.
  <https://www.cs.jhu.edu/~huang/cs624/spring21/readings/icomment.pdf>.
  *Reported by a third party*; not independently verified here.
- **Aghajani et al., *Software Documentation Issues Unveiled*, ICSE 2019**: "up-to-dateness
  problems" account for **39 %** of documentation content issues (as cited by Tan et al.,
  §1). Cited via Tan et al.; not read directly.
- **Huang, Chen, Chen & Zhou — *Are your comments outdated? Towards automatically detecting
  code-comment consistency*, arXiv:2403.00251.** A detection tool, not a study of rates.
  Reported precision 92.1 %, recall 78.9 %, F1 0.850 on their labelled set; and a
  practitioner check: "of the 25 outdated comments in the latest version detected by CoCC,
  93.6 % of comments programmers think that they are outdated comments". Also reports
  mutation-sensitivity: adding/deleting `WHILE` statements yields "> 45 % probability of
  outdated comments"; adding `FOR` and deleting `CATCH` "> 40 %".
  <https://arxiv.org/abs/2403.00251>, full text <https://arxiv.org/html/2403.00251>.
- **C4RLLaMA — *Code Comment Inconsistency Detection and Rectification Using a Large
  Language Model*, ICSE 2025**, pp. 1832–1843, DOI 10.1109/ICSE55347.2025.00035. Reports
  rectification "accuracy" of **65.0 % (just-in-time) and 55.9 % (post hoc)**. Author PDF:
  <https://people.cs.umass.edu/~brun/class/2024Fall/CS692P/idllm.pdf>. Read as a lead only —
  the DL abstract is paywalled.
- **Stulova, Blasi, Gorla & Nierstrasz**, *Towards detecting inconsistent comments in Java
  source code automatically*, SCAM 2020, pp. 65–69 (cited in the C4RLLaMA reference list;
  not read directly).

**Answers to the two questions asked of the literature:**

1. *Are decay rates measured?* Yes, and the strongest measurement (Tan et al.) is on
   **documentation** references rather than inline comments: 28.9 % of top projects stale
   *now*, ~55 % of stale references surviving one month, 1.3 % regressing after a fix. For
   inline comments the best available figure is Wen et al.'s complement — 80–87 % of code
   changes do not touch the related comment.
2. *Is drift inevitable and only cadence matters?* The evidence supports "decay is the
   default state and no shipped mechanism prevents it", but it does **not** support
   "detection cadence is the only lever": the three techniques that genuinely work are
   (a) making the doc the file (`include_str!`), (b) binding the claim to a
   `static_assert`, and (c) co-locating prose and code (literate programming). All three are
   *substitution*, none are detection. Detection is what the academic field does, and its
   own systematic review says that field still relies "on manual assessment and specific
   heuristics."

---

## Part C — What mature projects actually do

### C.1 Linux kernel (the closest comparable: C, heavy `file:line` usage)

**Policy on line references in comments: there is none.** I read
`Documentation/process/coding-style.rst` (1295 lines) in full. Its §8 "Commenting"
(beginning line 598) says: "Comments are good, but there is also a danger of
over-commenting. **NEVER try to explain HOW your code works in a comment** … Generally, you
want your comments to tell **WHAT** your code does, not HOW." It mandates kernel-doc format
for API functions and points at `tools/docs/kernel-doc` and `make W=n` for verification
("the documentation format of `.c` files is also verified by the kernel build when it is
requested to perform extra gcc checks … However, the above command does not verify header
files"). There is **no statement permitting or forbidding `file.c:123` citations**, and no
rule about counts. The document's only numerical rule is the 80-column line limit.
<https://raw.githubusercontent.com/torvalds/linux/master/Documentation/process/coding-style.rst>.

**Tooling that exists — exactly one script, and it is small.** `tools/docs/documentation-file-ref-check`,
5,839 bytes of Perl, invoked as `make refcheckdocs` or automatically under
`CONFIG_WARN_MISSING_DOCUMENTS`. Read in full
(<https://raw.githubusercontent.com/torvalds/linux/master/tools/docs/documentation-file-ref-check>):

- It is two `git grep` passes. Pass 1 greps `` :doc:`...` `` inside `Documentation/`.
  Pass 2 greps `Documentation/` across the tree.
- Existence test: `next if (grep -e, glob("$f"));` then `next if (grep -e, glob("$ref $fulref"));`
  — i.e. `File::Glob` against the filesystem.
- It normalises a small amount of noise: strips footnotes (`txt[1]`), trailing punctuation,
  `Documentation/output` paths, `http` URLs, `Makefile`/`*.sh`/`*.py`/`*.pl`, hidden files,
  `$(...)` make expressions, brace expansion.
- It carries a hard-coded `%false_positives` hash with a comment explaining the policy:
  "only add things here when the file was gone, but the text wants to mention a past
  documentation file, for example, to give credits for the original work."
- It has an `--fix` mode that guesses a replacement via `find -iname` and `sed -i`.
- **Explicitly out of scope:** any `file.c:123` form. It only looks at the `Documentation/`
  prefix, so a citation to `src/runtime/engine/context_cache.cc:42` is never extracted,
  let alone range-checked. Line numbers are never parsed.
- It also never fails on its own: the main body ends `exit 0 if (!$fix);`, so the finding is
  printed to stderr and the exit status is 0 unless you are in `--fix` mode. A caller must
  parse stderr.

**The rest of the kernel's doc tooling, and what it does:**

| Tool | What it checks | Does it check prose claims? |
|---|---|---|
| `tools/docs/documentation-file-ref-check` | `Documentation/` paths and `:doc:` roles resolve to existing files | **no line numbers, no counts** |
| `tools/docs/get_abi.py` (`validate`) | ABI files; run under `CONFIG_WARN_ABI_ERRORS` from `Documentation/Makefile` line 16 | no |
| `tools/docs/sphinx-build-wrapper` | wraps every Sphinx build; `sphinx-pre-install` manages the venv | no |
| `linkcheckdocs` target | external HTTP links only, "will connect to external hosts" | no |
| `scripts/checkpatch.pl` | **224 distinct check types** (I enumerated every `WARN("…")`/`CHK("…")`/`ERROR("…")` literal in the 7,985-line script) | **no.** Closest is `TYPO_SPELLING` — "spelling/typos", driven by `scripts/spelling/` plus `--codespell`. Nothing validates a comment's semantic claim. |
| `tools/docs/kernel-doc` + `make W=n` | kernel-doc *format* in `.c` files | format only; and header files are excluded from `W=n` |

The `checkpatch.pl` type list is worth recording as a negative: it contains
`BLOCK_COMMENT_STYLE`, `BAD_COMMENT_SEPARATOR`, `TRAILING_WHITESPACE`, `SPDX_LICENSE_TAG`,
`DT_SPLIT_BINDING_PATCH`, `MAINTAINERS_STYLE`, `MEMORY_BARRIER`, `OBSOLETE` — and nothing
that compares a comment's content to the code it sits on.

The kernel does have a **new** and closely relevant document that I would not have expected:
`Documentation/process/generated-content.rst` (4,666 bytes, added recently). It is a policy
about *AI and tool-generated contributions* — "be transparent about the origin of content",
name the tools, include the prompts, and "expect additional scrutiny in proportion to how
much of it was generated". It is about **provenance and review burden, not correctness**. No
tool enforces it; it is prose guidance in the patch-submission process.
<https://raw.githubusercontent.com/torvalds/linux/master/Documentation/process/generated-content.rst>.

### C.2 Chromium

**Tooling: none found for any of the three classes.** What exists is *policy prose*, and it
is notably defeatist:

- `docs/documentation_guidelines.md` opens: "Chromium's code base is large. Very large.
  Like most large places, it can be hard to find your way around … **It also changes a lot.
  Lots of people work on Chromium and refactoring, componentization, addition or removal of
  layers, etc. means that knowledge one has can quickly get out of date.**" and states the
  guiding principle: "It works from the principle that **all documentation is wrong and out
  of date**, but in-code documentation is less so and valuable."
  <https://raw.githubusercontent.com/chromium/chromium/main/docs/documentation_guidelines.md>.
  The rest of the document is a three-tier taxonomy (module README / interface comment /
  implementation comment) and exhortation. **No line-reference rule. No count rule.**
- `docs/documentation_best_practices.md`: "A small set of fresh and accurate docs is better
  than a large assembly of 'documentation' in various states of disrepair … **Docs work best
  when they are alive but frequently trimmed, like a bonsai tree**"; "**Change your
  documentation in the same CL as the code change**"; "**Delete dead documentation**"; "The
  documentation is **the contract of how your code must behave** … It is often reasonable to
  say that any behavior documented here should have a test verifying it."
  <https://raw.githubusercontent.com/chromium/chromium/main/docs/documentation_best_practices.md>.
  That last sentence is the closest thing to a mechanism, and it is a suggestion to the
  author, not a check.
- `docs/README.md` links markdown documents to a style guide at
  `https://chromium.googlesource.com/chromium/src/+/HEAD/styleguide/markdown/markdown.md`
  — a style guide, not a validator.
- The Google C++ style guide (`google/styleguide`, `cppguide.html`, 242 KB) has a large
  "Comments" section; I searched it for "line number" and got **zero hits**.
- Chromium's presubmit/check tooling (`depot_tools`, Gerrit) validates `DEPS`, style
  formatting, and licenses. I found no documentation-reference or documentation-count check.

### C.3 Rust project (the one project with a real class-1 mechanism)

`sccache`-class aside, the relevant artifact is `src/tools/tidy/src/` in `rust-lang/rust` —
a bespoke linter run in-tree. Of its ~30 modules, the ones relevant here:

- **`unstable_book.rs`** — the class-1 cross-check, described in D.1.
- **`pal.rs`** — enforces that `cfg(unix|windows|target_os|target_env)` appear only in
  allowed `std` paths. A code-organisation policy enforced by a dedicated linter.
- **`codegen.rs`** — codegen backend TODO policy, scanning `rs, py, js, sh, c, cpp, h, md,
  css, ftl, toml, yml, yaml`.
- **`extra_checks/mod.rs`** — wraps third-party formatters/linters (Python via a managed
  venv, shell), supports `--bless`.
- **No `documentation.rs`** — I checked the module list; there is no such file in the
  current tree.

Rust's *other* class-2 mechanism is the compiler/rustdoc pair: `rustdoc` lints
(`broken_intra_doc_links`, warn-by-default, `deny`-able) resolve intra-doc links, and
`tidy` handles the rest. Neither touches numbers in prose.

---

## Part D — Prior art specifically for "prose count must equal an authoritative value"

### D.1 The one strong instance: Rust `tidy`'s `unstable_book` check

Read in full at
<https://raw.githubusercontent.com/rust-lang/rust/master/src/tools/tidy/src/unstable_book.rs>.
Its structure is a **two-directional set difference** between a structured source of truth
and a prose artefact:

```rust
/// Retrieves names of all unstable features.
pub fn collect_unstable_feature_names(features: &Features) -> BTreeSet<String> {
    features.iter()
        .filter(|&(_, f)| f.level == Status::Unstable)
        .map(|(name, _)| name.replace('_', "-"))
        .collect()
}

/// Retrieves file names of all library feature sections in the Unstable Book ...
pub fn collect_unstable_book_section_file_names(dir: &Path) -> BTreeSet<String> { … }
```

and then in `check()`:

```rust
// Check for Unstable Book sections that don't have a corresponding unstable feature
for feature_name in &unstable_book_lib_features_section_file_names - &unstable_lib_feature_names {
    check.error(format!(
        "The Unstable Book has a 'library feature' section '{feature_name}' which doesn't \
                     correspond to an unstable library feature"
    ));
    maybe_suggest_dashes(&unstable_lib_feature_names, &feature_name, &mut check);
}
```

with the reverse direction — features that lack a book section — present but **commented
out**, with the note "Remove the comment marker if you want the list printed."

Why this is the best prior art, stated factually:

- it is maintained (lives in the compiler repo, pushed 2026-10-01);
- it derives the authoritative set from *structured* metadata (`features`, i.e. the
  compiler's own tracking file), not by parsing the prose;
- it normalises before comparing (`_`→`-`, `.md` stripped) and, on mismatch, attempts a
  *diagnosis* (`maybe_suggest_dashes` tells you the dash form exists);
- it operates on **filenames as the unit of correspondence**, not on numbers embedded in
  sentences.

Why it does not solve the audited problem: a filename *is* the claim, so there is nothing
to parse. "The suite is 137 tests", "three required real-model tests", "all six call sites",
and a JSON `note` saying 28 while `sites` sums to 27 are all **numbers in free text**, and
this mechanism has no representation for them.

### D.2 The four candidate patterns from the brief, assessed factually

**JSON Schema `$comment`.** Per the JSON Schema 2020-12 core spec §8.3: "$comment … This
keyword **reserves a location for comments from schema authors** … The value of this keyword
MAY be used in debug or error output which is intended for developers making use of
schemas." and "Vocabularies **MUST NOT** specify any effect of "$comment" beyond what is
described in this specification."
<https://json-schema.org/draft/2020-12/json-schema-core.html#section-8.3>.
**Verdict: the specification forbids exactly the validation you would want.** `$comment` is
documentation by construction; no validator may check it, and no schema keyword can relate
its content to a sibling numeric field. A schema can validate that `sites` has 27 entries;
it cannot validate that a `note` string says "28".

**Golden/snapshot tests over documentation files.** Exists as a *technique* with no
canonical tool; the nearest maintained instances are indirect:
- `llvm/utils/update_test_checks.py` (397 lines) regenerates expected test output from the
  source rather than hand-maintaining it. Widely used in LLVM.
- `mdbook test` and `sphinx.ext.doctest` (see A.2) execute documentation content.
- `pre-commit`'s `--bless`-style hooks and `tidy --extra-checks --bless` (see
  `extra_checks/mod.rs`) provide the "regenerate and commit" ergonomics.
**Verdict: nothing snapshot-tests the *text* of a doc against a computed value.** Snapshots
prove "the doc did not change since last time", which is a different guarantee from "the
number in the doc is right". There is no maintained snapshot tool that recomputes the
expected content from an authoritative source and diffs.

**`codespell` / `vale` custom rules.**
- `codespell` is dictionary-only ("looks for a set of common misspellings"); it has no
  numeric or cross-file capability and no rule-authoring mechanism at all.
- `vale` *does* have the most expressive rule model of anything surveyed — regex +
  part-of-speech (`existence`), cross-file/scoped `conditional` (whose `In` field selects
  which View to search for the second pattern), readability formulas (`metric`), and
  `script` rules compiled as Tengo programs. It also parses C/C++ comments via tree-sitter,
  which is directly relevant to class 3.
  **But the `script` sandbox is closed**: `stdlib.GetModuleMap("text", "fmt", "math")`. No
  `os`, no `fs`, no network. A Vale rule therefore **cannot read the authoritative file** it
  would need to compare against. `conditional` compares patterns within and across Vale's
  own *parsed* scopes, not against a JSON baseline. This is the single most decisive
  technical finding for class 1: **the best prose linter in existence is architecturally
  prevented from doing the check you need**, by a sandbox decision.

**Generated-from-source-of-truth + `<!-- generated -->` marker + drift check.** This is the
pattern with the strongest real-world evidence, and it is the one the Linux kernel and Rust
both use — but always with **generation at a different level** than the audit's failures:
- the kernel regenerates `Documentation/ABI` (`get_abi.py`) and generates from source, rather
  than asserting a hand-written number;
- Rust's `unstable_book` derives correspondence from structured metadata;
- LLVM's `update_test_checks.py` regenerates expected output.
In every case the fix is *remove the hand-written value*, not *check the hand-written value*.
Note the trade-off visible in the kernel: `documentation-file-ref-check` needs a
hand-maintained `%false_positives` allow-list precisely because generation is not available
for the prose that mentions past files.

### D.3 Direct evidence on class 1 itself

GitHub repository search for this exact problem found **nothing**:
`markdown linter verify numbers in documentation match code` → total 0;
`docstring count assert source of truth` → total 0;
`generated documentation drift check CI` → total 4, of which the three real hits are
0-star personal projects from 2026. The one with a non-trivial star count
(`nikolajflojgaard/code-doc-pipeline`, ★0) is an agent skill for "documentation drift
checks", which is a prompt, not a checker.

**Conclusion for class 1: no maintained tool exists that asserts a number written in prose
equals a value computed from an authoritative source.** The three real mechanisms in
mature projects all avoid the problem by construction (generate the value; make the file
name the claim; or make the file the document).

---

## Summary table

| Class | Maintained tool that does it? | Name | Unmaintained/partial | Does not exist |
|---|---|---|---|---|
| 1. Prose count vs authoritative value | no | — | Rust `tidy unstable_book` (filename correspondence only; reverse direction commented out) | any numeric assertion of a hand-written prose count |
| 2. `file:line` citation resolves | **no** | — | kernel `documentation-file-ref-check` (file existence, `Documentation/` only, never a line) | any maintained tool that parses `path:line` out of prose |
| 2b. Line number still in range | **no** | — | — | everything; `remark-validate-links` and `markdownlint MD051` both *exclude* line fragments on purpose |
| 3. Comment asserts a falsified mechanism | no (detection) | — | `clang -Wdocumentation` (Doxygen syntax + parameter names only) | any semantic truthfulness check |
| — (alternative to 3) | yes | `static_assert` bound to a documented invariant (LLVM `SmallVector.h`); `#[doc = include_str!]` (Rust std) | noweb (★308, active); lhs2tex (★107, Haskell only) | a framework that executes a C/C++ comment as a test |
| — (wider drift) | yes | `rustdoc::broken_intra_doc_links`; Sphinx `-n`/nitpicky + `-W`; Doxygen `WARN_AS_ERROR` | — | a generic "docs must match code" CI action |
| — (narrower class) | yes | `codespell` (dictionary-only, no structure) | — | — |

## Reproducing the key claims

Everything above is checkable from these, in the order used:

- lychee file checker scope: `lychee-lib/src/checker/file.rs`, `lychee-lib/src/extract/markdown.rs`
  at <https://github.com/lycheeverse/lychee/tree/master/lychee-lib/src>.
- remark-validate-links dropping line fragments: `lib/index.js` lines 181, 778–786.
- markdownlint MD051 exempting line fragments: `lib/md051.mjs` lines 11, 129.
- Vale's sandbox: `internal/check/script.go`, `program.SetImports(stdlib.GetModuleMap("text", "fmt", "math"))`.
- kernel reference checker: `tools/docs/documentation-file-ref-check`; wiring in
  `Documentation/Makefile` lines 9–16 and 74–75.
- kernel checkpatch check inventory: 224 distinct `WARN`/`CHK`/`ERROR` type literals in
  `scripts/checkpatch.pl`.
- Rust's class-1 cross-check: `src/tools/tidy/src/unstable_book.rs`.
- LLVM's `static_assert`-bound claim: `llvm/include/llvm/ADT/SmallVector.h` lines 1185–1230.
- Clang's comment diagnostics: <https://clang.llvm.org/docs/DiagnosticsReference.html#wdocumentation>.
- JSON Schema's prohibition on validating `$comment`:
  <https://json-schema.org/draft/2020-12/json-schema-core.html#section-8.3>.
- Literature: arXiv:2212.01479 (28.9 %, 82.3 %, 55 %-a-month survival, 47.6 % fixed by
  changing the source), arXiv:2209.08165 (2353 → 47 papers, 4 of 21 QAs), ICPC 2019 preprint
  at <https://csnagy.github.io/research/pdfs/2019/Wen2019-preprint.pdf> (13–20 % of code
  changes trigger a comment update).

## Labels used, and one caveat about them

Every repository was tagged *maintained* only on the basis of a `pushed_at` timestamp read
from the GitHub API on 2026-10-01: lychee 2026-09-28, linkinator 2026-09-29, markdownlint
2026-10-01, vale 2026-10-01, codespell 2026-09-28, markdown-link-check 2026-07-28,
remark-validate-links **2025-02-21**. A push timestamp shows the repository is not
archived; it does not show the maintainers' intent, and `remark-validate-links` is the one
case where I applied a judgement ("unmaintained-ish") beyond the timestamp.

Three things I could **not** verify and am flagging rather than asserting:

1. **MSVC's comment-validation surface.** I found no primary source for a Microsoft C++
   compiler warning that validates documentation comments, and I did not exhaustively read
   the MSVC warning catalogue (the index page 404s for the versioned URL I tried). The
   claim "MSVC has no `-Wdocumentation` equivalent" is an **absence of evidence**, not a
   demonstrated absence.
2. **The IEEE and ACM published versions of Wen et al. 2019** are paywalled; every quoted
   figure comes from the author's own preprint PDF hosted on a co-author's site, which
   states it is "the author's version of the work". The preprint and the proceedings paper
   could in principle differ.
3. **Chromium's tooling absence** is a negative claim established by reading two docs and
   the Google style guide, and by `docs/README.md`'s index — not by reading
   `depot_tools`'s full presubmit rule set. Chromium may have a private or
   less-visible check I did not find.