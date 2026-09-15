# Mars Rover Terrain Lab

All browser logic, styles, Three.js and OrbitControls CDN imports are in `dist/index.html`. Serve `dist` over HTTP; direct `file://` loading cannot fetch the JSON assets.

## Assets

- `terrain.json`: 256×256 normalized heights, row-major, 1 m grid, original physical elevation range and crop metadata. The mesh restores meters with no vertical exaggeration.
- `heightmap.png`: 8-bit grayscale representation of the same normalized heights; JSON retains higher precision.
- `perception.json`: measured pixel confusion counts, row-normalized probabilities, checkpoint SHA-256, validation metrics and real camera replay metadata.
- `nav-*.jpg`, `overlay-*.jpg`: real AI4MARS validation frames and actual best-checkpoint predictions at 128×128, enlarged to 384×384. These frames are not spatially registered to the HiRISE crop.

`scripts/prepare_assets.py` uses Python, rasterio, NumPy, Pillow and PyTorch. It reads the downloaded HiRISE PDS IMG and the existing neighboring `ai4mars_unet` training outputs. Rebuild from the parent workspace with `.venv/bin/python mars-rover-demo/scripts/prepare_assets.py`.

DTM source: NASA/JPL-Caltech/University of Arizona, DTEEC_009149_1750_009294_1750_U01, Gale Crater inverted riverbed. Source pixels are 1.0117467945795 m; a window beginning at column 2812, row 3616 spans 256 m and is resampled with rasterio bilinear interpolation. Pixel-center mesh spans 255 m. The exported values restore -4057.949462890625 to -4024.263427734375 m elevation (offset to zero for rendering).

## Explicit simulation assumptions

This is geometry, not terramechanics or rigid-body dynamics. No slip, sinkage dynamics, forces, suspension constraints or real rock detection are simulated.

The terrain classes are slope proxies on coherent 4 m patches: below the 22nd slope percentile is sand; above the 94th is an obstacle proxy; above the 58th is bedrock; remaining patches are soil. These are not known geological labels. Seed 42 assigns a stable predicted class to each patch from the measured validation confusion matrix (true class row, predicted class column). Pixel-level statistics transferred to terrain patches are an explicit modeling assumption. No U-Net inference is performed on orbital data.

Detected obstacle patches stop the preset traverse. Missed obstacle patch overlaps increment the geometric impact counter once per patch per lap. A coarse 3×3 footprint sample detects overlaps; the counter is not a contact force calculation. DTM geometry is never altered by perception mode. Sub-meter rocks are not represented.

Wheel centers sample bilinear terrain elevation plus radius. Body elevation is the mean of all six centers plus 0.65 m, with pitch from front/back means and roll from left/right means. Thin box links are illustrative. Body width is 1.6 m and body length is wheel span + 1.44 m, giving 4.14 m at the default 2.7 m span. Underbody clearance is the minimum vertical distance at 45 underside samples.

Equal static load is 2000×3.71/6 = 1236.67 N per wheel. Contact area uses a circular chord with an assumed 20 mm indentation: width × 2√(2rδ−δ²). Pressure is load/area. The default 9 kPa allowable pressure is illustrative and adjustable, not a measured bearing-capacity limit.

## Validation

The recomputed confusion matrix matches the best model's stored validation mIoU within 0.0001. All four classes have validation support; the model predicts no big-rock pixels. Browser checks exercised real rendering, default pressure 8.25 kPa, narrow/small wheels at 10.60 kPa with warning, body-length change to 4440 mm, perfect-perception stopping, model-mode travel, positive missed-impact count, camera replay and invalid configuration rejection.
