Review the current changes with this repo's own review skill. NInfer's review skill is
`cpp-cuda-review`; the project-local `code-review` skill is a .NET/Roslyn import and does not apply
to C++/CUDA work here.

## Usage

```
/cpp-cuda-review [commit|branch|pr|path]
```

## Instructions

1. Call the `skill` tool for `cpp-cuda-review` before reading any diff. Do not review from memory of
   the skill's contents.
2. Default to the uncommitted changes. With an argument, review that revision range, branch, or path
   instead (`git diff <arg>`, `git log -p <arg>` as appropriate).
3. Report findings by severity, each with file and line, and state which checks you could not run.
4. This command reviews; do not modify code unless the user asks for a fix afterwards.
