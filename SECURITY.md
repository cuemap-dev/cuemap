# Security Policy

## Reporting a vulnerability

Please do not report security vulnerabilities in public GitHub issues, pull requests, or discussions.

Use GitHub's private vulnerability reporting for this repository:

[Report a private vulnerability](https://github.com/cuemap-dev/cuemap/security/advisories/new)

Include the affected version or commit, operating system, configuration, reproduction steps, and potential impact. Remove API keys, credentials, personal data, and other secrets from reports and reproduction cases.

If private vulnerability reporting is unavailable, contact the maintainers through the private communication channel listed in the repository or organization profile.

## Scope and deployment guidance

CueMap can expose an HTTP server and supports API-key authentication, encryption at rest, and cloud backups. Before deploying it outside a trusted local environment:

- enable API-key authentication;
- restrict network access to trusted clients;
- protect encryption keys, cloud credentials, and snapshot files;
- avoid placing secrets in ingested content or checked-in configuration;
- keep the engine and companion packages updated.

The Apache-2.0 license is described in [LICENSE](LICENSE). Earlier engine releases may use the BSL-1.1 license noted in their respective release artifacts.

## Supported versions

Security fixes target the latest released version. When reporting an issue, include the exact CueMap engine version and the versions of any SDK or MCP package involved.
