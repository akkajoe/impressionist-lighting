# impressionist-lighting
Computational pipeline for estimating and analyzing pictorial lighting in Impressionist paintings, as described in "Quantifying Lighting Consistency in Impressionist Painting" (CHR 2026).

## Dependencies

This pipeline uses DiffusionLight-Turbo for chrome ball synthesis. Clone and install it first:

git clone git@github.com:DiffusionLight/DiffusionLight-Turbo.git
conda env create -f environment.yml
conda activate diffusionlight-turbo
pip install -r requirements.txt

Additional packages needed:

pip install orjson scikit-image pandas tqdm

## Scripts

ball2pointcloud.py
  Takes chrome ball images from DiffusionLight-Turbo (Inpainting the chrome ball) and exports a 3D point cloud (positions, normals, light directions, luminance) as JSON or gzipped JSON.

ransac_light.py
  Runs weighted multi-hypothesis RANSAC on the point cloud to estimate up to 3 dominant light directions per painting. Outputs results to a CSV.

brightness_metrics.py
  Computes peak-to-mean ratio, brightness concentration, and angular spread from chrome ball images. Outputs per-painting, per-artist, and per-scene CSVs.

## Usage

# Step 1: run DiffusionLight-Turbo on your paintings to get chrome ball images

# Step 2: export point clouds
python ball2pointcloud.py --ball_dir balls/ --envmap_dir points/ --zip_json

# Step 3: estimate light directions
python ransac_light.py --input points/painting_points.json.gz --csv results.csv

# Step 4: compute luminance metrics
python brightness_metrics.py --root_dir balls/ --out_dir metrics/

## Interactive explorer

Available at [URL provided upon acceptance — withheld for anonymous review].
