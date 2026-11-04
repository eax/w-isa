# Security and trust boundary

W-ISA edits model memory in the caller's Python process. The application supplies
trusted programs, callbacks, checkpoint paths and model objects. Application
authentication and authorization belong to that calling application.

The compiler checks parameter names, operand dimensions and finite values before
execution. Numeric NPZ operands are loaded with `allow_pickle=False`. Runtime
snapshots preserve the entry weights of each edited parameter.

Request scopes coordinate operations through a shared per-model lock. Route all
inference sharing that model through the same lock. Restoration executes through
Python's `finally` mechanism for completion, exceptions and cancellation. The
scope owns runtime weight operations for the lifetime of the running process.

For a security report, use the repository's Security tab and its private reporting
option when enabled. Otherwise, request a private contact channel from the
maintainer. Provide a minimal reproduction, affected versions and redacted logs
through that private channel.