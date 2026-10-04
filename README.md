# RiverVision

RiverVision is a browser-first, human-reviewed stream photo assessment demo.

## Run locally

```sh
npm install
npm run dev
```

The production bundle is built with `npm run build` and served with `npm run preview`.

## What the demo does

- Accepts JPG, PNG, and WebP photos and checks decoded size, approximate blur, and brightness in the browser.
- Loads TensorFlow.js and its WebGL backend on demand. The current assessment suggestions are deterministic demo outputs, not predictions from a trained model.
- Draws clearly labeled illustrative overlay regions on the image. They are not segmentation results.
- Requires an answer to each observation before saving. Confirmed records preserve the AI suggestions, human decisions, final decisions, timestamp, and disagreement state in browser local storage.

Replace the demo classifiers in `src/ai/` with validated, locally appropriate models before using the app for real environmental assessments. Connectors for server storage and optional ONNX inference are not configured.
