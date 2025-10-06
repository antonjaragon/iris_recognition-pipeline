# 🧠 Iris Recognition Pipeline (Segmenter + Matcher)

This project provides an **automated two-step pipeline** for iris image processing and matching using Docker containers.  
It performs:

1. **Segmentation** – extracts the iris region and generates masks using a segmentation container.  
2. **Matching** – encodes, matches, and computes statistics between iris codes using a matcher container.

The pipeline automatically handles file organization, data copying, and result collection on the host system.

---

## 🚀 Requirements

Before running the pipeline, make sure the following are installed on your computer:

- **[Docker](https://docs.docker.com/get-docker/)**  
  You must also **start the Docker Engine** before executing the script.  
  > 💡 On Linux, Docker usually starts automatically.  
  > 💡 On Windows/macOS, make sure **Docker Desktop** is running.

- **Python 3.8+**

---

## 🧩 Setting Up the Python Environment

To isolate dependencies, it’s recommended to use a **Python virtual environment**.

### 1️⃣ Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

Once activated, you should see (venv) at the beginning of your terminal prompt, indicating the virtual environment is active.


### 2️⃣ Install required dependencies

Install all Python dependencies from the requirements.txt file:


### 3️⃣ Verify setup

Check that Python and Docker are correctly installed:

```bash
python --version
docker --version
```

If both commands return version numbers, your setup is ready.


### 📁 Project Structure

```bash
.
├── .env                    # Environment variables for Docker containers
├── requirements.txt        # Python dependencies
├── main.py             # Main pipeline script (segmenter + matcher)
├── analyze_roc.py          # ROC analysis script for matching results
├── distances/              # Output folder for matcher results
├── masks/                  # Output folder for generated masks
└── DB_SYNTIRIS_vero_0_5_selectedframes/   # Example input images folder

```

### ⚙️ Environment Configuration (.env)

Create a .env file (or rename your configuration .txt file to .env) in the project root.
It defines all Docker container and folder configurations.

```bash
# === SEGMENTER CONFIGURATION ===
SEGMENTER_CONTAINER_NAME=iris_segmenter
SEGMENTER_IMAGE=my-segmenter-image
SEGMENTER_FOLDER=segmenter
SEGMENTER_INPUT_FOLDER=data/input
SEGMENTER_OUTPUT_FOLDER=data/output
SEGMENTER_COMMAND="python run_segmentation.py"

# === MATCHER CONFIGURATION ===
MATCHER_CONTAINER_NAME=iris_matcher
MATCHER_IMAGE=antoniounimi/usit
MATCHER_FOLDER=usit
MATCHER_INPUT_IMAGES=data/input_images
MATCHER_INPUT_MASKS=data/masks
MATCHER_OUTPUT_FOLDER=data/distances
MATCHER_COMMAND="python run_matching.py"

```

Each container must read and write inside the mounted /host directory.


### ▶️ Running the Pipeline
Run the full pipeline by providing your input images directory as a command-line argument:

```bash
python pipeline.py /path/to/input_images
```

What the pipeline does

1. Copies input images into the segmenter’s input directory.

2. Runs the segmentation container (Docker).

3. Collects generated masks into masks/.

4. Copies images + masks into the matcher’s input directories.

5. Runs the matcher container to encode, match, and compute distances.

6. Saves results to the distances/ folder.


### 🧪 Example Usage
```bash
python pipeline.py DB_SYNTIRIS_vero_0_5_selectedframes
```
Expected output:

```bash
[STEP 1] Preparing inputs for segmenter...
[DOCKER] Running: docker run -it --rm ...
[STEP 1] Segmentation finished.
[STEP 2] Preparing inputs for matcher...
[DOCKER] Running: docker run -it --rm ...
✅ Pipeline complete (2 steps: Segmenter, Matcher).
```

### 📊 Post-Processing and Evaluation
Once the matcher completes, you’ll find a results file like matching_results.csv (depending on the matcher) inside the distances/ folder.

You can analyze these results using the included ROC analysis script:

```bash
python distributions_plot.py distances/matching_results.csv
```

The script displays:

- AUC (Area Under the Curve)

- EER (Equal Error Rate)

- ROC Curve

- DET Curve

- Genuine vs. Impostor Distributions

Example output:

```bash
AUC: 0.9852
EER: 0.0321
Threshold @ EER (distance space): 0.416253
```