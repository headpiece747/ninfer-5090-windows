# CodeGraph `init` replaces a junction/symlink with an empty directory

**Target:** `@colbymchenry/codegraph` 1.5.0 (github.com/colbymchenry/codegraph)
**Platform:** Windows 11, PowerShell 5.1, Node via npm global install
**Severity:** data-layout destructive (the symlink is destroyed; no files are deleted)

## Summary

Running `codegraph init <path>` where `<path>` is a **directory junction** (or symlink)
replaces the link with an empty real directory, then reports `No files found to index`. The
link target itself survives, but the path the user has been using all along becomes an
empty directory, and the indexing silently does nothing.

## Reproduction

```powershell
# The real tree
C:\AI\ninfer-v3-windows\            # full source checkout, .git present

# A junction pointing at it, which is the path everything else uses
C:\AI\infer-v3-windows  -> C:\AI\ninfer-v3-windows

codegraph init "C:\AI\infer-v3-windows"
```

Output:

```
Scanning files...
Parsing code...
|  !  No files found to index
|  -  Done
```

Afterwards:

```powershell
Get-ChildItem "C:\AI\infer-v3-windows" -Force
#   .codegraph          <- the only entry

(Get-Item "C:\AI\infer-v3-windows").LinkType
#   (empty)             <- it is no longer a link
```

`codegraph status "C:\AI\infer-v3-windows"` then reports `Files: 0, Nodes: 0, Edges: 0`,
while the real tree still has ~850 source files.

Running `codegraph init "C:\AI\ninfer-v3-windows"` directly (the real path) works fine:
**1,292 files, 29,663 nodes, 88,908 edges in 2.9s**.

## Impact

- The junction is silently destroyed, so every other tool that resolved through it starts
  pointing at an empty directory.
- The user gets a successful-looking "Done" and an empty index, with no hint that the path
  they indexed was not a real directory.
- On Windows, junctions are the normal way to keep a canonical checkout path while the real
  tree lives elsewhere, so this is not an exotic layout.

## Suggested fix

Before creating `.codegraph/`, resolve the reparse point and either:

1. refuse with a clear message ("`<path>` is a junction to `<target>`; run `codegraph init`
   on the target"), or
2. index the **resolved** target and create `.codegraph/` there, leaving the link intact.

Detection: `fs.lstatSync(path).isSymbolicLink()` on Node, or check
`FILE_ATTRIBUTE_REPARSE_POINT` via `GetFileAttributesW` — noting that Windows junctions
report as reparse points and are exposed by `fs.lstat` as symbolic links.

Either behaviour is fine; the important part is never replacing the link.

## Evidence

The substitution is reproducible and leaves the target untouched. I recovered by deleting
the empty directory and re-running `init` against the real path, which indexed normally —
so the tool itself is working; only the link handling is wrong.
