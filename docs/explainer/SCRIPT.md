# Explainer video: script and how to re-render

`docs/explainer.mp4` is a 76 second, 1080p animated explainer. It is rendered from
`explainer.html` in this folder, frame by frame, so every number on screen is typed into one
HTML file and the video never drifts from the README or the benchmark.

## Narration (for a voice track)

Timestamps match the scenes. Read at a normal pace; each line fits its scene with a beat to spare.

| at | say |
|---|---|
| 0:00 | mcpgraphy. Your agent's tools, mapped before it acts. |
| 0:06 | An MCP server hands your agent a list of tools. Names, descriptions, parameters. It does not say when a tool works. So the agent edits a file that isn't there, publishes a release that was never tested, moves something that moved already. Errors, retries, and sometimes damage. |
| 0:18 | mcpgraphy sits between the agent and the server. The tools pass through unchanged. But before and after every call it reads the resource back, and it keeps the evidence. |
| 0:28 | From that evidence it builds a graph. Nodes are the states it actually observed. Edges are what happened from each state, with the number of times it happened, and a status that is honest: supported, contradicted, or unknown. Unknown is not valid. Unknown is not invalid. |
| 0:41 | Ask it about a resource and you get offers for its current state. Run the tests: supported, twelve observations. Publish: not supported from here, it failed every time. Archive: uncertain, it worked twice and failed once, and it needs approval. |
| 0:52 | Three policies. Observe just learns. Warn passes everything through and adds a note. Block stops the calls the evidence says will fail, and anything destructive, before the server ever sees them. Every decision is audited. |
| 1:03 | We measured it. On two hundred stateful tasks, raw tool calling made two point two invalid calls per task. With mcpgraphy, point six seven. And the honest part: just logging the agent's own calls gets you most of that. Which is exactly what the proxy does. |
| 1:12 | pip install mcpgraphy. Put it in front of any MCP server with a five line adapter. Map first, act second. |

## Re-render

Needs Chrome, Node, `@playwright/test` and ffmpeg. From a folder where `@playwright/test` is
installed:

```bash
node docs/explainer/frames.mjs "$PWD/docs/explainer/explainer.html" video docs/explainer.mp4 76 30
node docs/explainer/frames.mjs "$PWD/docs/explainer/explainer.html" stills docs/explainer 3 32 58   # single frames
```

The page exposes `window.__seek(seconds)`, which places every animation at that moment and sets
the counters and typed text. The renderer steps through time at 30 frames per second and pipes
PNGs into ffmpeg, so the output is sharp and never drops a frame. Change the numbers or the copy
in `explainer.html` and render again.

To add a voice track:

```bash
ffmpeg -i docs/explainer.mp4 -i voice.m4a -c:v copy -c:a aac -shortest docs/explainer-voiced.mp4
```
