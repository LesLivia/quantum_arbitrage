# Implied Density Function Estimation

This project provides tools to estimate the implied risk-neutral density function from options data using non-parametric techniques. It includes functionality for data processing, density estimation, and visualization.

## Project Structure

- `main.py`: The main script to run the estimation pipeline. It handles input, calls the estimation function, and generates output.
- `implied_density_function.py`: Contains core functions for computing the implied density function, including interpolation, differentiation, and smoothing techniques.

## Usage

Modify the input data and parameters inside `main.py` as needed, then run:

```bash
python main.py
```

The script will output the estimated density function and save any plots or data as configured.

## License

This project is released under the MIT License.

## Disclaimer

This code is for academic and research purposes only. It is not intended for financial advice or production use.
