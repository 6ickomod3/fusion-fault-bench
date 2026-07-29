# Data and model terms

The Apache-2.0 license in this repository covers only the original software and
documentation contributed to Fusion Fault Bench. It does not cover nuScenes or
any other third-party dataset, model, checkpoint, or generated payload.

## nuScenes

Users must obtain nuScenes directly from its official distributor, accept the
applicable terms, and configure a local dataset root. This repository does not
redistribute nuScenes archives, images, point clouds, maps, metadata tables, or
credentials.

Fusion Fault Bench uses the mini split for the released M2 local geometry and
metadata-grounding check. The tracked M2 release contains only sanitized
aggregate attestations and a summary figure; it contains no source row,
per-frame payload, image, point cloud, map, archive, token, filename, or local
path. Planned M5 replay will use recorded latent geometry, motion,
calibration, and timing while estimator errors remain simulated. The project
does not claim endorsement by Motional or the nuScenes authors.

The current [Motional Dataset Terms](https://www.nuscenes.org/terms-of-use)
state that data derived from nuScenes, including tables and charts, remains
subject to those terms and to CC BY-NC-SA 4.0 unless separately licensed.
Accordingly:

- project code and repository-owned synthetic fixtures remain Apache-2.0;
- any tracked release record or figure derived from a local nuScenes run is
  marked `CC BY-NC-SA 4.0 plus Motional Dataset Terms`, with attribution and
  non-endorsement language;
- the Apache-2.0 license does not relicense those derived records or figures;
  and
- commercial users must obtain appropriate permission from Motional.

This repository is prepared for personal, educational, and non-commercial
research evaluation. Review the controlling upstream terms for your own use;
terms may change after this document is published.
