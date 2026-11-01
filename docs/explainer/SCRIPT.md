# Explainer video: script and how to re-render

`docs/explainer.mp4` is a 107 second, 1080p animated explainer (silent; the narration below is timed to it). It is rendered from
`explainer.html` in this folder, frame by frame, so every number on screen is typed into one
HTML file and the video never drifts from the README or the benchmark.

## Narration (for a voice track)

Timestamps match the scenes. Read at a normal pace; each line fits its scene with a beat to spare.

| at | say |
|---|---|
| 0:00 | Hey. This is MCPgraphy. It maps what your agent's tools can actually do, before the agent acts. |
| 0:07 | Here's the thing about tool calling. Your agent gets a list of tools: edit a file, publish a release, move a ticket. What it doesn't get, is when those work. So it tries. Edits a file that isn't there. Publishes a release nobody tested. Gets an error, tries again. And sometimes the retry is the one that breaks something. |
| 0:28 | MCPgraphy sits in the middle. Every tool call passes straight through, nothing changes for your server. But around each call, it quietly reads the thing the agent touched, and remembers what happened. |
| 0:41 | Do that for a while and you get a map. Here's every state we've actually seen a release in. Here's what worked from each one, and how many times. And when we haven't seen something? We say unknown. Not safe. Not broken. Unknown. |
| 0:57 | So now, when your agent reads a release, we add one line. From here, running tests works. Publishing has failed every time. Archiving? Needs approval. The agent doesn't have to guess anymore. |
| 1:09 | You pick how hard we push. Observe just learns. Warn adds a note. Block stops the calls we've seen fail, before your server ever sees them, and tells the agent what to do instead. |
| 1:21 | We put a real GPT agent on two hundred jobs. On its own, it finished seventy-four percent, and claimed success on ten it never finished. Behind MCPgraphy: ninety-nine percent. Zero false claims. And per job actually done, fewer tokens. Same budget. More work. |
| 1:41 | pip install MCPgraphy. Put it in front of any MCP server. Map first. Act second. |

Pronunciation: "M-C-P graphy", with "graphy" rhyming with "coffee". For a text-to-speech voice, spell it "MCP graph-ee".

## Re-render

Needs Chrome, Node, `@playwright/test` and ffmpeg. From a folder where `@playwright/test` is
installed:

```bash
node docs/explainer/frames.mjs "$PWD/docs/explainer/explainer.html" video docs/explainer.mp4 107 30
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
