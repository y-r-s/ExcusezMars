# Mars Rover Terrain Lab

All browser logic, styles, Three.js and OrbitControls CDN imports are in `dist/index.html`. Serve `dist` over HTTP; direct `file://` loading cannot fetch the JSON assets.

## Assets

- `terrain.json`: 256×256 normalized heights, row-major, 1 m grid, original physical elevation range and crop metadata. The mesh restores meters with no vertical exaggeration.
- `heightmap.png`: 8-bit grayscale representation of the same normalized heights; JSON retains higher precision.
- `perception.json`: measured pixel confusion counts, row-normalized probabilities, checkpoint SHA-256, validation metrics and real camera replay metadata.
- `models/mars_export.glb`: rover body mesh (Blender glTF export, 6.5k triangles, metres, Y-up, symmetric fore/aft; 2040 × 4272 mm footprint, 1970 mm tall, underside 805 mm above origin). Loaded with three.js GLTFLoader; wheels and suspension links stay procedural so the sliders can resize them. If the file fails to load, a box body is shown.
- `nav-*.jpg`, `overlay-*.jpg`: real AI4MARS validation frames and actual best-checkpoint predictions at 128×128, enlarged to 384×384. These frames are not spatially registered to the HiRISE crop.

`scripts/prepare_assets.py` uses Python, rasterio, NumPy, Pillow and PyTorch. It reads the downloaded HiRISE PDS IMG and the existing neighboring `ai4mars_unet` training outputs. Rebuild from the parent workspace with `.venv/bin/python mars-rover-demo/scripts/prepare_assets.py`.

DTM source: NASA/JPL-Caltech/University of Arizona, DTEEC_009149_1750_009294_1750_U01, Gale Crater inverted riverbed. Source pixels are 1.0117467945795 m; a window beginning at column 2812, row 3616 spans 256 m and is resampled with rasterio bilinear interpolation. Pixel-center mesh spans 255 m. The exported values restore -4057.949462890625 to -4024.263427734375 m elevation (offset to zero for rendering).

## Explicit simulation assumptions

This is geometry, not terramechanics or rigid-body dynamics. No slip, sinkage dynamics, forces or real rock detection are simulated.

The terrain classes are rule-based proxies on each 1 m DTM cell, computed in the browser from slope and topographic position (elevation minus the 7×7 local mean): the top 0.8% by slope or local relief is Big rock; slope above the 72nd percentile or relief above the 90th is Bedrock; low-lying cells (relief below the 14th percentile, slope below the 45th) are Sand; the rest is Soil, followed by one 3×3 majority pass over ground classes. These are not known geological labels. Connected Big-rock cells form rock clusters.

In My model mode each cell draws a predicted class from its true class's row of the measured validation confusion matrix. The random draws come from ~7 m value noise, rank-transformed to exactly uniform, so errors are spatially correlated and a cluster tends to be detected or missed as a whole. The default seed is 42; "Resample errors" advances it. Pixel-level statistics transferred to terrain cells are an explicit modeling assumption. No U-Net inference is performed on orbital data. This checkpoint never predicts Big rock, so every rock cluster is missed.

Preset waypoints sit next to the largest rock clusters (at least 45 m apart, away from the crop edge), visited in angular order. A* plans between them on the current perception map only: Big rock is inflated by 3 m and blocked; Soil costs 1, Bedrock 1.6, Sand 4. The route is smoothed and resampled every 0.5 m. With perfect perception the rover detours around rocks; with the model it drives through missed clusters. Wheel and body-centerline samples that enter a true rock cluster increment the impact counter once per cluster per lap and mark the site; the counter is not a contact force calculation. DTM geometry is never altered by perception mode. Sub-meter rocks are not represented.

Wheel centers sit on the rendered triangle mesh plus radius. Body attitude uses pitch from front/rear wheel means and roll from left/right means; the model underside is 0.405 m above the mean axle height (its designed 805 mm ride height on 800 mm wheels). Thin box links are illustrative. The body is the imported model: 2.04 m wide, stretched along its length only to 4.272 m + (span − 2.7 m). Wheels sit outboard of the chassis rail: track is 2.04 m + 2×0.08 m gap + wheel width. Underbody clearance is the minimum vertical distance at 45 samples across the 1.9 m wide underside; the route minimum is sampled every 1 m along the planned route.

Equal static load is 2000×3.71/6 = 1236.67 N per wheel. Contact area uses a rigid-wheel chord at an assumed sinkage (default 20 mm): width × 2√(D·z − z²). Pressure is load/area. The default 9 kPa allowable pressure is illustrative and adjustable, not a measured bearing-capacity limit; pressures at or above 90% of the limit are flagged as near.

## Validation

The recomputed confusion matrix matches the best model's stored validation mIoU within 0.0001. All four classes have validation support; the model predicts no big-rock pixels. `window.demo.simulateLap(mode, seed)` drives a full lap without rendering: seeds 42–44 give 0 impacts with perfect perception and 5–6 with the model (20/20 clusters missed); route minimum clearance is 682 mm with perfect perception at default geometry. Browser checks also covered default pressure 8.25 kPa (near limit), 700 × 500 mm wheels at 10.60 kPa (over limit), body length 4572 mm at 3000 mm span, GLB model loading, camera replay and the `configure_rover` tool.
