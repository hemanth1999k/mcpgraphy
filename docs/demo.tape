# VHS tape for the README demo: `vhs docs/demo.tape` writes docs/demo.gif and docs/demo.mp4
# https://github.com/charmbracelet/vhs
#
# The downstream is the official filesystem MCP server (npx @modelcontextprotocol/server-filesystem),
# so Node must be installed. Nothing else is needed: no keys, no accounts.

Output docs/demo.gif
Output docs/demo.mp4

Set Shell "bash"
Set FontSize 13
Set FontFamily "Menlo"
Set Width 1280
Set Height 900
Set Padding 22
Set Theme "Catppuccin Mocha"
Set TypingSpeed 25ms

Hide
Type "export PATH=$PWD/.venv/bin:$PATH; export PS1='$ '; rm -rf /tmp/mcpgraphy-demo; clear"
Enter
Sleep 500ms
Show

Type "# mcpgraphy in front of the official filesystem MCP server. Act 1: a naive agent works, mcpgraphy only watches."
Enter
Sleep 1.5s
Type "python examples/filesystem_demo.py /tmp/mcpgraphy-demo 2>/dev/null"
Enter
Sleep 30s
Type "# It learned from the agent's own 7 calls. Act 2 (same run, above): the proxy restarts in block mode and the same mistakes never reach the server."
Enter
Sleep 6s

Hide
Type "clear"
Enter
Show
Type "# Everything it learned is a graph with evidence on every edge."
Enter
Sleep 1.2s
Type "mcpgraphy graph export --server filesystem --db /tmp/mcpgraphy-demo/.mcpgraphy.sqlite --format mermaid"
Enter
Sleep 5s
