# llm-plugin-utils
LLM plugin utilities for axum-based projects.

This library is specifically intended for OpenAI-style plugins served with [axum](). The `manifest` struct provides a description conforming to [OpenAI's spec](https://platform.openai.com/docs/plugins/getting-started/plugin-manifest). The `serve_plugin_info` function returns a very simple `Router` that serves the required plugin resources.

The library is currently incomplete.
