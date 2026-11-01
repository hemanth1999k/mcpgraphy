# Explainer video: script and how to re-render

`docs/explainer.mp4` is a 76 second, 1080p animated explainer (silent; the narration below is timed to it). It is rendered from
`explainer.html` in this folder, frame by frame, so every number on screen is typed into one
HTML file and the video never drifts from the README or the benchmark.

## Narration (for a voice track)

Timestamps match the scenes. Read at a normal pace; each line fits its scene with a beat to spare.

| at | say |
|---|---|
| 0:00 | MCPgraphy. Your agent's tools, mapped before it acts. |
| 0:06 | MCP tells your agent what tools exist, never when they work. So it edits a file that isn't there. Retries. And sometimes, does damage. |
| 0:18 | MCPgraphy sits between agent and server. The tools pass through, unchanged. But around every call, it reads the resource back, and keeps the evidence. |
| 0:28 | From that evidence, it builds a graph. Nodes are states it actually observed. Edges are what happened from each one, how often, and whether later calls agreed. Unknown stays unknown. |
| 0:41 | Ask about a resource, and you get offers for its state right now. Run tests? Supported. Publish? Failed every time from here. Archive? Uncertain, and it needs approval. |
| 0:52 | Three policies. Observe just learns. Warn passes everything through, with a note. Block stops the calls the evidence says will fail, before the server ever sees them. |
| 1:03 | With a real GPT agent: raw tools finished seventy-four percent of tasks. Behind MCPgraphy, ninety-nine, with no false successes. More tokens per task, fewer per job done. |
| 1:16 | pip install mcpgraphy. Put it in front of any MCP server. Map first. Act second. |

Pronunciation: "M-C-P graphy", with "graphy" rhyming with "coffee". For a text-to-speech voice, spell it "MCP graph-ee".

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
