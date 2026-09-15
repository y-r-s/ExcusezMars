# ExcusezMars
This is the repo of team ExcusezMars for Mars City Hackathon - GirlWhoML x PhysicsX. Team member Julia Hong, data scientist/designer and Youran Song, 2D/3D designer.

## Mars terrain perception and rover demo

This repository contains the completed AI4MARS U-Net baseline, reproducible training artifacts, English result visualization, and a single-page Three.js rover geometry demo on real HiRISE terrain.

## ARMA - The VW Beetle ON MARS

ARMA explores a simple question: what would a compact, two-person utility vehicle for Mars look like if its design were informed by terrain perception and first-order engineering calculations? We started with the proportions of a VW Beetle and the six-wheel, rocker-bogie concept used by Mars rovers, then defined a **2,000 kg total operating mass**, including the vehicle, cargo and two astronauts. The goal is to connect vehicle geometry, terrain conditions and mobility requirements in one demonstrator.

### Design reference and working configuration

The Beetle provides our compact body reference: **4,140 mm long, 1,600 mm wide, with a 2,400 mm reference wheelbase**. Perseverance provides a planetary mobility reference: approximately **3,000 mm long, 2,700 mm wide, 2,200 mm high and 1,025 kg**, with six wheels and rocker-bogie suspension. These are design references, not interchangeable specifications or evidence that a scaled vehicle will perform the same way. See [NASA's Perseverance rover components](https://science.nasa.gov/mission/mars-2020-perseverance/rover-components/).

Our current demo uses an **800 mm wheel diameter, 600 mm wheel width and 2,700 mm front-to-rear wheel span** as an adjustable baseline. Users can compare diameters of 700-900 mm, widths of 500-700 mm, and spans of 2,400 / 2,700 / 3,000 mm. These values are candidate design choices, not an ML-derived optimum.

### What we actually built

1. **Camera-based terrain perception.** We trained a U-Net from scratch on an AI4MARS subset of 1,813 images, split into 1,631 training and 182 validation images. It predicts soil, bedrock, sand and big rock. The best checkpoint reached **51.88% validation mIoU and 86.73% pixel accuracy** after a ten-epoch run at 128 × 128 resolution. Its big-rock IoU and recall are **0%**, so this baseline cannot establish that a route is obstacle-free.
2. **Real terrain geometry.** Using Python and rasterio, we extracted a 256 × 256 height sample from a HiRISE DTM of Gale Crater, resampled it to a 1 m grid, and exported normalized JSON heights and a grayscale PNG. Three.js reconstructs the terrain in physical meters with no vertical exaggeration. The current scene uses our imported 3D body model with six terrain-following wheels. Body height follows the six-wheel average, with pitch and roll estimated from the wheel contacts. Suspension links illustrate the rocker-bogie concept rather than solve its mechanical constraints.
3. **A perception-error experiment.** The terrain's displayed classes are simulated from DTM slope and local relief. A switch applies the U-Net's measured validation confusion matrix with spatially correlated errors. An A* planner connects preset waypoints, treating perceived rocks and a 3 m surrounding buffer as blocked while assigning different costs to other terrain classes. Switching perception replans the route. Missed rock proxies lose their warning color, can enter the planned route, and increment a geometric encounter counter when crossed. These route costs are illustrative, not calibrated energy costs; the smoothed route is not a certified collision-free trajectory. The side panel separately replays real AI4MARS camera images with actual U-Net prediction overlays. Those images are not geographically registered to the DTM.

Together, these components let us explore how design choices and perception failures could affect mobility. **U-Net is not applied directly to orbital data, and the demo does not confirm real-world traversability.** The 1 m grid does not represent sub-meter rocks; wheel-soil forces, slip, structural strength and vehicle dynamics are not simulated.

### Vehicle calculations: wheel loading and contact pressure

We use SI units and Mars gravity `g = 3.71 m/s²`. For total mass `m = 2,000 kg`:

- Total gravitational force: `W = mg = 7,420 N`.
- Average static load per wheel: `N = W / 6 = 1,236.7 N`.

Equal load sharing is a flat-ground approximation. Individual wheel loads change with slope, cargo placement and obstacle encounters.

For the demo's contact-area comparison, a circular wheel of radius `r` and width `b` uses a **default assumed 20 mm indentation**, `δ = 0.02 m` (adjustable in the interface):

```text
Contact length L = 2 × sqrt(2rδ - δ²)
Contact area   A = b × L
Mean pressure  p = N / A
```

| Candidate | Wheel diameter × width | Contact area per wheel | Mean contact pressure |
| --- | --- | --- | --- |
| Smaller | 700 × 500 mm | 0.117 m² | 10.60 kPa |
| Current baseline | 800 × 600 mm | 0.150 m² | 8.25 kPa |
| Larger | 900 × 700 mm | 0.186 m² | 6.66 kPa |

These figures compare geometry at the **same assumed indentation**; they do not predict actual sinkage or prove that the soil can support the rover. The demo's adjustable **9 kPa pressure threshold is illustrative**, not a measured Martian soil limit. A larger contact patch lowers pressure in this approximation, but wider wheels also affect vehicle width, mass and steering resistance. Wheel-soil testing or a calibrated terramechanics model is needed to resolve those trade-offs; see [NASA's terramechanics modeling white paper](https://ntrs.nasa.gov/citations/20220010732).

### Vehicle calculations: body dimensions and clearance

The original Beetle-inspired reference was **4,140 × 1,600 mm**. The current imported 3D body has a **4,272 × 2,040 mm bounding box** at the default 2,700 mm wheel span. These should not be presented as the same geometry.

Using the current body's 2,040 mm width, 600 mm external wheels and the code's 80 mm body-to-wheel gap on each side:

```text
Wheel-center track = 2,040 + 2 × 80 + 600 = 2,800 mm
Overall wheel envelope = 2,040 + 2 × (600 + 80) = 3,400 mm
```

This is the calculated wheel envelope before steering, not a swept-clearance assessment. If 1,600 mm is intended as an overall vehicle-width limit, the current model does not meet it.

The demo links body length to front-to-rear wheel span using `body length = 4,272 mm + (span - 2,700 mm)`. This gives **3,972 / 4,272 / 4,572 mm** body lengths for the three span settings. It is an explicit geometry rule, not a mechanical requirement; a later design could hold body length fixed and vary its overhangs instead.

The modeled underside is **405 mm above the mean axle height**. At an 800 mm wheel diameter, nominal flat-ground clearance is `400 + 405 = 805 mm`. Current clearance is the smallest vertical gap at **45 underside sample points** against the terrain; route-minimum clearance repeats this calculation every 1 m along the planned path. These sampled values change with wheel size, span and terrain, and do not establish continuous collision clearance or account for actual soil sinkage. The contact-pressure indentation setting is not a wheel-soil displacement simulation.

A longer span may improve longitudinal static stability but can reduce breakover clearance. A full stability assessment also needs the loaded center-of-mass height and actual wheel track.

### Vehicle calculations: drive torque and energy

For a first-order, constant-speed uphill estimate:

```text
Required traction force F = mg × (sin θ + Crr × cos θ)
Average wheel-end torque = F × r / 6
Electrical drive power  = F × v / η
Drive energy per km     = F / (3.6 × η) Wh/km
```

Here `θ` is slope angle, `Crr` is an assumed effective rolling-resistance coefficient, `v` is speed in m/s, and `η` is battery-to-wheel efficiency. The following scenarios use **800 mm wheels, 0.5 m/s speed and 70% efficiency**. The rolling-resistance coefficients are sensitivity assumptions, not values measured or inferred by our U-Net.

| Scenario | Average wheel-end torque | Electrical drive power | Drive energy |
| --- | --- | --- | --- |
| Flat, Crr = 0.05 | 24.7 N·m | 0.265 kW | 147 Wh/km |
| Flat, Crr = 0.10 | 49.5 N·m | 0.530 kW | 294 Wh/km |
| Flat, Crr = 0.20 | 98.9 N·m | 1.060 kW | 589 Wh/km |
| 10° uphill, Crr = 0.10 | 134.6 N·m | 1.442 kW | 801 Wh/km |

These are **planning calculations documented here, not a power simulation implemented in the demo**. The 0.5 m/s assumption is independent of the animation playback speed. They exclude acceleration, turning, significant slip and obstacle climbing, and assume the soil can provide the required traction. They are average wheel-end demands, not motor-shaft ratings or peak motor requirements.

Auxiliary loads must be added separately. For example, a hypothetical continuous **300 W** equipment load adds `300 / (3.6 × 0.5) = 167 Wh/km` at the assumed speed. The flat `Crr = 0.10` scenario would therefore use approximately **461 Wh/km** including that example load. Life support, heating, battery temperature effects and operational reserves have not been sized, so these numbers cannot yet establish battery capacity or range.

### What this means for ARMA

Our current recommendation is to retain **800 × 600 mm wheels and a 2,700 mm span as the comparison baseline**, then test the smaller and larger configurations against the same terrain and mission assumptions. The next engineering inputs are the overall-width constraint, loaded center of mass, target speed and slope, soil properties, drivetrain efficiency and auxiliary power budget. In parallel, big-rock perception needs improvement before its predictions can inform obstacle decisions. The present work provides a reproducible way to compare assumptions and expose failures, rather than a validated final vehicle design.

### Quick start: rover demo

```sh
python3 -m http.server 8000 --directory mars-rover-demo/dist
```

Open <http://localhost:8000>. Three.js and OrbitControls load from a CDN, so an internet connection is required. All browser logic and styles are in one HTML file; terrain and camera assets are local static files.

[Hosted demo (owner-private)](https://mars-rover-terrain-lab.finn-zhang626.chatgpt.site) · [Download demo bundle](mars-rover-demo.zip)

The demo includes six terrain-following wheels, adjustable diameter/width/wheel span, estimated contact pressure, sampled underbody clearance, perfect/model perception, missed-obstacle encounter counts, and real AI4MARS camera/prediction replay.

### Training result

| Setting / metric | Value |
| --- | --- |
| Original train split | 18,130 images |
| Eligible after label checks | 15,899 images |
| Selected subset | 1,813 images (10% of original split) |
| Train / validation | 1,631 / 182 |
| Input / training | 128 × 128, 10 epochs, seed 42 |
| Best checkpoint | Epoch 9 |
| Validation mIoU | 51.88% |
| Validation pixel accuracy | 86.73% |

![Training results](ai4mars_unet/run/results_summary.png)

These are validation results, not official test-set results. Big-rock IoU is 0%; the model does not predict that class. The small input resolution and short run are baseline limitations.

### Repository contents

| Path | Contents |
| --- | --- |
| `ai4mars_unet/` | Training, inference and visualization scripts |
| `ai4mars_unet/data/` | Exact sampled image/mask arrays, source revision and sample indices |
| `ai4mars_unet/run/` | Best and last checkpoints, configuration, metrics, example predictions, PNG/PDF report |
| `mars-rover-demo/dist/` | Single HTML demo plus DTM JSON/PNG, confusion matrix and camera assets |
| `mars-rover-demo/scripts/` | Python + rasterio asset preparation |
| `track-b-vehicles-ai4mars/` | Original supplied terrain pack and simulated logistics data |
| `Perseverance_Panorama_8k-2.jpg` | Original team repository panorama, preserved |

### Reproduce training and figures

Python 3.9 was used for the recorded run. The root requirements include rasterio for DTM processing; the training folder also records its earlier environment.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python ai4mars_unet/train.py
.venv/bin/python ai4mars_unet/visualize_results.py
.venv/bin/python ai4mars_unet/predict.py ai4mars_unet/run/example_input.png
```

The exact 1,813-image subset is included, so the default training run does not need the full source dataset. Training again overwrites `ai4mars_unet/run/`. If the subset is absent, preparation downloads approximately 6.1 GB of training shards. Changing the seed or image size generates a new subset.

### Rebuild DTM and perception assets

The processed 256 × 256 height JSON/PNG and measured confusion matrix are already included. To regenerate them, download the original DTM first:

```sh
curl -L --fail --retry 3 \
  'https://www.uahirise.org/PDS/DTM/PSP/ORB_009100_009199/PSP_009149_1750_PSP_009294_1750/DTEEC_009149_1750_009294_1750_U01.IMG' \
  -o ai4mars_unet/HiRISE_Gale.IMG
.venv/bin/python mars-rover-demo/scripts/prepare_assets.py
```

Original DTM (~391 MB), downloaded dataset shards and the virtual environment are intentionally excluded from Git. The two checkpoints and sampled data are included. The original Sites repository metadata is excluded from this GitHub copy; it is not needed to run the static demo.

### Interpretation and sources

- Terrain colors are slope-based simulated classes, with errors sampled from the measured validation confusion matrix. U-Net is **not applied directly to orbital imagery**.
- The grid is 1 m, resampled from approximately 1.012 m HiRISE pixels. Rocks smaller than 1 m are not represented.
- This is a **geometric demonstration, not a dynamics simulation**. Contact area assumes 20 mm indentation; the adjustable pressure threshold is illustrative. Obstacle encounters are proxies, not collision-force estimates.
- Navigation frames are real AI4MARS validation images with actual U-Net overlays, but are not geographically aligned with the DTM.
- [AI4MARS Hugging Face source](https://huggingface.co/datasets/hassanjbara/AI4MARS); exact source revision and selected row IDs are recorded in `ai4mars_unet/data/selection.json`.
- [HiRISE DTM: Gale Crater inverted riverbed](https://www.uahirise.org/dtm/dtm.php?ID=PSP_009149_1750), NASA/JPL-Caltech/University of Arizona. Product, crop and elevation metadata are in `terrain.json`.
- The original terrain pack includes synthetic dust frames and simulated logistics; see its `DATA_NOTES.txt`. It was not used as the training subset for this baseline.

See [training notes](ai4mars_unet/README.md) and [demo assumptions](mars-rover-demo/README.md) for details. Source data retain their respective attribution and usage terms; no new blanket license is asserted for third-party assets.
