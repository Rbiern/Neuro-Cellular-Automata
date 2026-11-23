# Cellular Studio

Cellular Studio is an interactive Tkinter lab for neural cellular automata. Grow and repair sprites, generate still and animated textures, or explore a learned reaction–diffusion field. Playback runs locally on the CPU.

## Setup

In your project’s Python environment, install the dependencies and check that Tkinter is available.

```sh
python -m pip install -r requirements.txt
python -m tkinter
```

## Run the demo

```sh
python src/app.py
```

Studio opens with an empty canvas. Use **Open model…** to load a checkpoint from `models/`, then press **Play**. Use the [training recipes](#reproduce-the-experiments) to create example checkpoints. On a compatible Sprite checkpoint, changing **Design** with **Switch in place** enabled gives the existing cells a new target. Hover over an info icon for a tooltip, or click it to open the explanation in a dialog.

For sprite models, move **Growth angle** or enter a value and press Enter to restart growth at that orientation. Positive angles are clockwise. Drag the eraser or enable **Hover to erase** to make a cut while the simulation continues. Double-click to plant a seed or reseed a patch.

| Control | Behavior |
| --- | --- |
| Open model | Opens a saved `.pt` checkpoint. Training saves to `models/` by default. Source images are in `assets/`. |
| Show cell grid | Overlays the real simulation cells, including in Neural detail mode. |
| Growth angle | Rotates the Sobel perception filters or the isotropic structured seed. Shapes at intermediate angles are approximations on the square cell grid. |
| Play, Step, Reset to seed | Runs the simulation, advances a single update while paused, or resets the current design and orientation. |
| Speed and brush size | Set the playback rate and eraser diameter in cells. Living patterns can advance several simulation steps per displayed frame. |
| Eraser cursor | Hover over the image to see the round brush cursor at the selected size. Drag to erase, or enable Hover to erase. |
| Synchronous updates | Updates all cells in each step for comparison with the default stochastic update rule. It can change a model’s stability. |
| Neural detail | Uses the learned decoder for the preview and PNG export when available. The preview queries the decoder at up to 384 pixels per side during playback and 768 when paused. |
| Crisp edges | Makes the visible outline opaque at 50% alpha and transparent below that. The preview and PNG export use the same rule. Disable it for naturally translucent artwork. |

**Export PNG** saves a snapshot at a chosen square resolution up to 2048 × 2048. With **Neural detail** enabled, the exporter queries the decoder at the requested pixel positions. With it disabled, PNG export scales the simulation cells using nearest-neighbor sampling.

Passing a checkpoint path on the command line opens it paused. Earlier single-image checkpoints still open in the viewer.

## Research references

| Follow-up | What is implemented |
| --- | --- |
| [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/), Mordvintsev et al., 2020 | The original growth, persistence and regeneration experiments inform the shared update rule, random cell updates, state pools and damage training. |
| [Growing Isotropic Neural Cellular Automata](https://arxiv.org/abs/2205.01681), Mordvintsev, Randazzo, and Fouts, 2022 | Optional symmetric Laplacian perception and a three-point structured seed. Rotating the seed changes the direction of growth while the update rule stays the same. |
| [Neural Cellular Automata Can Respond to Signals](https://arxiv.org/abs/2305.12971), Stovold, 2023 | A design signal selects the target that the shared update rule grows. Studio supplies the signal each step and uses a training state pool for each design. |
| [Goal-Guided Neural Cellular Automata](https://arxiv.org/abs/2205.06806), Sudhakaran et al., 2022 | Live morphing changes the input goal during a rollout. Studio trains on mature states from other design pools alongside fresh seeds and damaged states. |
| [Complex Patterns in a Simple System](https://doi.org/10.1126/science.261.5118.189), Pearson, 1993, and [Learning Spatio-temporal Patterns with Neural Cellular Automata](https://doi.org/10.1371/journal.pcbi.1011589), Richardson et al., 2024 | The Living patterns checkpoint learns a local approximation from simulated Gray–Scott trajectories. Studio uses a periodic grid, a mass-action inductive bias, and one fixed parameter regime. |
| [Neural Cellular Automata: From Cells to Pixels](https://arxiv.org/html/2506.22899v3), Pajouheshgar et al., SIGGRAPH 2026 | The 2D decoder uses interpolated cell states and periodic sine/cosine coordinates. Training combines a coarse alpha silhouette term with sampled L1 and L2 RGBA reconstruction. |
| [Learning Continuous Image Representation With Local Implicit Image Function](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Learning_Continuous_Image_Representation_With_Local_Implicit_Image_Function_CVPR_2021_paper.html), Chen et al., CVPR 2021 | The same local decoder is queried directly at the requested output pixel centers. This supports sizes that are not integer multiples of the cell grid. |
| [Self-Organising Textures](https://distill.pub/selforg/2021/textures/), Niklasson et al., 2021, and [Texture Generation with Neural Cellular Automata](https://arxiv.org/abs/2105.07299), Mordvintsev et al., 2021 | An image exemplar supervises an NCA that synthesizes a texture across the field. |
| [A Sliced Wasserstein Loss for Neural Texture Synthesis](https://arxiv.org/abs/2006.07229), Heitz et al., CVPR 2021 | Studio matches distributions of random linear projections of local color patches. |
| [DyNCA](https://arxiv.org/abs/2211.11417), Pajouheshgar et al., CVPR 2023 | Studio adopts separate appearance and motion references. It transports the learned texture with an estimated or supplied flow field. |
| [The Phase Correlation Image Alignment Method](https://ndlsearch.ndl.go.jp/en/books/R100000136-I1573668924758608512), Kuglin and Hines, 1975, and [An Iterative Image Registration Technique with an Application to Stereo Vision](https://idl.uw.edu/living-papers-paper/lucas-kanade/index.pdf), Lucas and Kanade, 1981 | Phase correlation estimates the frame-to-frame translation. A local Lucas–Kanade residual adds spatial variation to the motion field. |
| [Adam](https://arxiv.org/abs/1412.6980), Kingma and Ba, ICLR 2015 | All trainers use Adam to update the learned model parameters. |

The design picker selects trained targets. This project adapts the 2D growth method with a lightweight ReLU decoder and sampled reconstruction. The simulation remains on its coarse grid, so repair acts on simulation cells. The detail renderer can express finer edges and color within cells, with fidelity limited by the source and training resolution.

## Train and open a model

Click **Train model…**, select **Sprite from images**, then click **Add images…**. You can train up to 32 target images in the same checkpoint. The list order is the order of the designs in the saved model. Hover over an info icon for a quick explanation or click it to open the explanation in a dialog.

To train the wave-like model, select **Living patterns (Gray–Scott)** instead. Set the training iterations and seed, then choose **Save and train…**. This preset generates its own training trajectories on CPU using the built-in Gray–Scott parameter regime. Once saved, open its checkpoint in Studio.

### Train a texture from references

Select **Texture from image** for an automaton that learns the colors and local structure of one image and produces a repeating texture across the canvas. Select **Animated texture** to supply an appearance image plus a motion reference.

- Use an animated GIF, APNG, or WebP, a folder of numbered frames, or a video such as MP4 with FFmpeg on PATH. Studio estimates a short loop of motion fields from consecutive frames.
- Supply an authored flow field with a NumPy `.npy` file shaped `H × W × 2` or `F × H × W × 2`, with up to 32 fields. The two components are horizontal rightward and vertical downward motion in source pixels per step. Studio rescales the vectors for the simulation grid.
- **Direction** steers the local texture rule without resetting the canvas. It also turns the motion vectors in animated models. **Synchronous updates** changes which cells apply the learned rule, while reference motion advances the whole field. Double-click to reseed a patch.
- **Neural detail** works on texture checkpoints trained with the decoder. **Crisp edges** is available when the texture target has transparency. Gray–Scott uses its own opaque palette and a symmetric, synchronous update rule.

Studio matches local color-patch distributions for appearance and transports the learned texture using the reference flow. The decoder queries a periodic cell grid for finer output.

### Sprite training settings

| Tab | Settings |
| --- | --- |
| Setup | Add, remove, and reorder target images. Choose the training mode, grid width, number of iterations, and optional pixel-art sampling, neural detail, flat-background removal, and live morphing. |
| Advanced | Choose Sobel or isotropic perception, auto/CPU/CUDA, batch size, training seed, rollout lengths, detail scale and sample count, or a checkpoint to warm-start from. The default rollout length scales with grid size. |

A transparent PNG gives the trainer a clean silhouette. For an opaque image with a uniform matte, enable **Remove flat background**. **Preserve hard pixel edges in targets** controls how the target is sampled, while **Train neural detail renderer** controls whether the resulting model can render between cells. The two options can be used together.

For live morphing, add at least two targets, select **persistent** or **regenerating**, and use a batch size of at least 3. To fine-tune a checkpoint, choose it under **Advanced** and supply its target images in the indicated order. Choose **Save and train…** to select the output in `models/`. The viewer stays usable during training. When training finishes, use **Open model…** to load the new checkpoint.

### CPU and GPU settings

Sprite and texture training use CUDA automatically when available. To select a device, pass `--device cpu` or `--device cuda`. Gray–Scott training and viewer playback run on CPU. For NVIDIA GPUs, use the CUDA build from the [PyTorch installation selector](https://pytorch.org/get-started/locally/).

Set **CPU threads** in the training dialog or pass `--cpu-threads N` to a trainer. At **0 (Auto)**, small sprite and texture jobs and Gray–Scott use one thread, and larger sprite and texture batches use up to four. Sprite and texture training use a channels-last tensor layout on CPU. Playback defaults to one thread and accepts its own `--cpu-threads` override. The training log shows the chosen thread count.

To train a multi-design model with the decoder:

```sh
python src/train.py --target assets/sprout.png --target assets/flower.png --output models/my-studio-demo.pt --size 20 --batch-size 3 --min-steps 20 --max-steps 28 --iterations 2000 --mode regenerating --neural-detail
python src/app.py models/my-studio-demo.pt
```

To train on a higher-resolution transparent PNG:

```sh
python src/train.py --target assets/my-high-res-sprite.png --output models/hi-res.pt --mode regenerating --size 48 --neural-detail --detail-scale 12 --batch-size 3
python src/app.py models/hi-res.pt
```

`--size` is the simulation size in cells. `--detail-scale` multiplies it to set the square training canvas for the decoder, with a range of 2 to 16 and a default of 8. For example, 48 cells at scale 12 gives a 576-pixel training canvas, with the sprite centered and padded. `--detail-samples` controls the number of pixel queries per training step. Raising the training canvas resolution does not increase the number of simulation cells or the configured pixel query count. Use **Export PNG** to choose the output dimensions.

For a 512 × 512 source on a 96 × 96 grid, start with `--size 96 --neural-detail --detail-scale 9 --detail-samples 8192 --batch-size 2`. This gives an 864 × 864 decoder canvas with up to 540 × 540 pixels available for the sprite after padding. Use a transparent PNG with a clear silhouette. Enable **Neural detail** for the live preview, and export at 864 pixels to inspect the render at the training canvas size.

To train an isotropic model:

```sh
python src/train.py --target assets/sprout.png --output models/my-sprout-isotropic.pt --size 20 --batch-size 3 --min-steps 20 --max-steps 28 --iterations 1200 --mode regenerating --perception isotropic
python src/app.py models/my-sprout-isotropic.pt
```

| Training mode | What it trains |
| --- | --- |
| `growing` | Start from the initial seed state and grow toward the target. |
| `persistent` | Reuse states from earlier rollouts so the pattern can learn to maintain itself. |
| `regenerating` | Reuse and damage rollout states so the pattern can learn to repair missing cells. |

Supply one transparent PNG for a single design or repeat `--target` for a model with multiple designs. Distinct filenames become the design names. The trainer crops transparent margins, preserves aspect ratio, and leaves room around the sprite for growth and repair. Run `python src/train.py --help` for the full CLI.

For an opaque sprite with a flat background, add `--remove-background` to remove the corner color where it connects to the image border. This gives the automaton the character silhouette as its growth target.

Use `--pixel-art` for nearest-neighbor training targets. Choose a grid large enough to fit the native sprite in the central 5/8 of the canvas. The example flower is 17 pixels wide, so a 28-cell grid can preserve every source pixel. A larger export of a coarse checkpoint scales its cells, and a detail decoder's fidelity depends on its training data.

## Reproduce the experiments

Train the crisp Sprout and Flower model on the original sprite pixels:

```sh
python src/train.py --target assets/sprout.png --target assets/flower.png --output models/my-studio-pixel.pt --size 28 --batch-size 3 --min-steps 28 --max-steps 44 --iterations 2100 --mode regenerating --pixel-art --live-morph
```

Fine-tune an existing demo for in-place design switching. Target order, grid size, perception, and decoder settings must match the warm-start checkpoint. Use the multi-design checkpoint from the sprite training example, or omit `--init-model` to train from scratch.

```sh
python src/train.py --target assets/sprout.png --target assets/flower.png --output models/my-studio-morph.pt --init-model models/my-studio-demo.pt --size 20 --batch-size 4 --min-steps 24 --max-steps 40 --iterations 1800 --mode regenerating --neural-detail --detail-scale 4 --detail-samples 1024 --live-morph
python src/app.py models/my-studio-morph.pt
```

Train the reaction–diffusion demo from generated trajectories. The trainer uses 48 × 48 fields, and the viewer runs the learned rule on a 72 × 72 field. Long rollouts may diverge from the numerical reference.

```sh
python src/train_patterns.py --output models/my-studio-patterns.pt
python src/app.py models/my-studio-patterns.pt
```

## Code style

The repository's Ruff configuration applies to all Python files. With Ruff installed, run `ruff check .` and `ruff format --check .` before sharing a change.
