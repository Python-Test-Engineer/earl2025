# Position Requirements

Based on what Laura posted in Slack, here are the requirements for the position:
- Help establish and implement robust CI/CD pipelines
- Advise on and enforce best practice coding standards (testing, code security, etc.)
- Guide deployment of production-ready Python applications (versioning, library compatibility)
- Mentor the team with an experience-based, opinionated approach

# TLDR;

This is handled by GitHub Actions using pre-commit, Conventional Commits, semantic versioning and python-semantic-release.

This is a well established approach that can be used to ensure that the codebase is clean, consistent, and up-to-date before every commit.

With the ability to apply rules at the 'atomic level' to ensure for example that only the deployment branch, (usually main) is deployed.

Implementation would be in stages to ensure that the team is well trained and able to use this approach effectively as well as provide an incremental approach to developing the final workflow, tested thoroughly at each stage on a replica system.

The attraction for the team is that they are freed from 'blame' if a developer were to break the requirements of the agreed workflow. It is the job of the GitHub Actions Administrator to ensure gatekeeping.

## Pre-commit

Using [Precommit](https://pre-commit.com/) with additional tooling for ease like [Husky](https://github.com/typicode/husky) to run pre-commit hooks on commit, we can ensure that our code is clean, consistent, and up-to-date before every commit. 

We can also enforce the structure of the commit message with [Conventional Commits]
(https://www.conventionalcommits.org/en/v1.0.0/) to ensure that the commits are well-structured and easy to understand. It also enables versioning to be automatically determined based on the commit history.

This will enforce a consistent and clean codebase, making it easier to collaborate and maintain.

## GitHub Actions

Using [GitHub Actions](https://github.com/features/actions) to automate the build, test, and deployment process.

We can use GitHub Actions to run tests on different versions of Python, perform code quality checks, and generate coverage reports.

This will ensure that the codebase is always up-to-date and ready for deployment.

We can also apply rules at the 'atomic' level to ensure for example that only the deployment branch, (usually main) is deployed.

## Team inspiration for using this approach

**Developer freed from 'blame'**
The GitHub Actions should be sufficiently rigorous that if a develop where to break the requirements of the agreed CI/CD rules, they are free of all blame. *This is a very powerful attraction for its use by developers as they are freed from 'blame'*

**Opinionated == Agreed and Desired by Team**

A different frame of reference from the developer perspective. It is the job of the GitHub Actions Administrator to ensure gatekeeping.

## Automated releases

If needed we can use [python-semantic-release](https://github.com/python-semantic-release/python-semantic-release) to automatically create releases and publish to PyPI as needed.

## Semantic Versioning

Using [Semantic Versioning](https://semver.org/) to determine the version of our code. 



## Core Components Overview

### GitHub Actions
GitHub Actions serves as our CI/CD orchestration platform, providing native integration with GitHub repositories and supporting complex workflows through YAML configuration files.

### Semantic Versioning (SemVer)
SemVer provides a structured approach to version numbering using the format MAJOR.MINOR.PATCH, where increments indicate the nature of changes (breaking changes, new features, or bug fixes).

### Conventional Commits
A standardized commit message format that enables automated changelog generation and version bumping based on commit types (feat, fix, docs, etc.).

### Pre-commit Library
A framework for managing and maintaining multi-language pre-commit hooks that catch issues before they enter version control.

## Implementation Strategy

### Phase 1: Pre-commit Hook Setup

Pre-commit hooks serve as the first line of defense, catching issues locally before code reaches the repository. The implementation includes:

**Configuration (.pre-commit-config.yaml):**
- Code formatting, quality etc with Ruff
- use `uv` rather then pip over time
- Security scanning 
- Dependency vulnerability checking with safety

### Phase 2: Conventional Commits Integration

Commit messages are structured for both human and machine use and automated changelog generation.

Implementing conventional commits requires team training and tooling support:

**Commit Format Structure:**
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Primary Types:**
- feat: New features (triggers MINOR version bump)
- fix: Bug fixes (triggers PATCH version bump)
- docs: Documentation changes
- style: Code style changes
- refactor: Code refactoring without feature changes
- test: Adding or modifying tests
- chore: Build process or auxiliary tool changes

**Breaking Changes:**
- Include "BREAKING CHANGE:" in commit footer or use "!" after type
- Triggers MAJOR version bump


## Team Inspiration and Adoption Strategy

### Addressing Common Concerns

**"This adds complexity to our workflow"**
While initial setup requires investment, the automation eliminates far more complexity than it introduces. Teams typically see productivity gains within weeks of implementation.

**"We don't need this level of automation"**
The system scales to team size and project complexity. Even small projects benefit from consistent quality checks and automated releases, while larger projects find the automation essential.

**"Learning curve will slow us down"**
The conventional commit format and pre-commit workflow become natural within days. The immediate feedback from automated tools accelerates learning rather than hindering it.

**Developer freed from 'blame'**
The GitHub Actions should be sufficiently rigorous that if a develop where to break the requirements of the agreed CI/CD rules, they are free of all blame. *This is a very powerful attraction for its use by developers as they are freed from 'blame'*
