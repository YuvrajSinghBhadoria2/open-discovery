# Open Discovery in Claude Code

Read and follow @AGENTS.md. Treat the workflows under `.agents/skills/` as
repository-local instructions. When the user names a workflow or their request
clearly matches one, read its complete `SKILL.md` before acting.

For onboarding and examples, use @WORKFLOWS.md.

For local credentials, GitHub identity, and per-repository push/PR permissions,
read @LOCAL-ACCESS.md. It is machine-local and gitignored — never copy its
contents into a commit, patch, PR body, or chat message, and never print a
token value. Before claiming GitHub access is unavailable, check §4 of that
file: the blocker is per-repo permission, not missing credentials.
