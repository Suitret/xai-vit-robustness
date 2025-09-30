# Explainable AI Robustness Study

This repository contains the code and data for a research study evaluating the robustness of 2 XAI methods.

## Repository Contents

- `src/`: Python scripts for generating saliency maps, computing metrics, and running statistical tests.
- `data/`: Sample dataset including clean and corrupted images (ImageNet-C severity-5 corruptions).
- `predictions.csv`: Model prediction data for clean and corrupted images.
- `results/`: Generated saliency maps and quantitative metric summaries.
- `requirements.txt`: Dependencies required to run the code.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Suitret/xai-vit-robustness.git
   cd xai-vit-robustness
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the main script to generate saliency maps and compute metrics:

   ```bash
   python src/main.py
   ```
2. Analyze results in the `results/` directory or use provided scripts for statistical analysis.

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- numpy
- pandas
- matplotlib
- scipy

(Listed in `requirements.txt` for easy installation.)

## Contributing

Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request with detailed changes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This work builds on research by Petsiuk et al. (2018) and Englebert et al. (2023), with data support from the ImageNet-C dataset by Hendrycks and Dietterich (2019).
