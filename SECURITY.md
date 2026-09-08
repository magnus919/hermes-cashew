# Security Policy

## Supported Versions

Security fixes are made against `main` and released for the latest published
version. Older releases are not supported unless a maintainer explicitly says
otherwise. Check whether the latest release resolves the problem, but report it
privately if you are unsure.

## Reporting a Vulnerability

Please do not disclose suspected vulnerabilities in a public issue, discussion,
or pull request. Use GitHub's private vulnerability reporting form instead:

<https://github.com/magnus919/hermes-cashew/security/advisories/new>

If the private form is unavailable, contact the maintainer through the
[GitHub profile](https://github.com/magnus919) without including sensitive
details and ask for a private reporting channel.

Include enough information to reproduce and assess the report safely:

- the affected version or commit;
- impact and realistic attack scenario;
- prerequisites and a minimal reproduction;
- relevant logs, traces, or screenshots with secrets and personal data removed;
- any proposed mitigation or patch, if available; and
- whether the vulnerability has been disclosed anywhere else.

Reports about secret exposure, unsafe path handling, untrusted workflow
execution, dependency or supply-chain compromise, and unauthorized access to
conversation or knowledge-graph data are in scope. Reports that require access
to another person's systems or data without permission are not.

## What to Expect

The maintainer aims to acknowledge a complete report within five business days.
After initial triage, the reporter will receive the current assessment and a
proposed next step. Remediation timing depends on severity and complexity, so no
fixed resolution deadline is promised.

Please allow time for a fix and coordinated disclosure. When appropriate, the
project will publish a GitHub security advisory, credit the reporter if desired,
and describe affected versions and remediation steps.

## Safe Research

Good-faith research should use systems and data you own or are authorized to
test. Avoid privacy violations, service disruption, persistence, data
destruction, and unnecessary access or retention. Stop testing and report
privately if you encounter user data, credentials, or an active compromise.
