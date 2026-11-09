# Conventions


Use PascalCase Python module/class names; keep conventional `__init__.py` package markers. Use descriptive snake_case variables and parameters. Avoid one-letter local names; `_` may mark an intentionally unused value. External library aliases and established JSON/SQL keys retain their compatibility contracts.

Every class has a responsibility docstring. Functions/methods carry parameter and return annotations and docstrings with Args/Returns where applicable. `Any` denotes a dynamic boundary, not an assertion that arbitrary values are accepted at runtime. Runtime validation remains authoritative.

Use objects for related state and behavior: the command runner, model/transport adapters, scoring service, retriever and CPU worker each own a concrete responsibility. Stateless transformations remain functions. Add an abstraction only when actual callers require it. Keep presentation, inference, data preparation and persistence separate.

Changes to scoring contracts must update every public/local/research consumer and its meaningful checks. Changes to notebook source require matching ZIP/manifest/notebook regeneration. Preserve accepted data, source labels, raw outputs and immutable historical releases.


[Docs](../README.md)
