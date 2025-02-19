# cs4080-group-project

### Josh Edwards, Austin Weingart, Carlos Guerra, Emily Ng

# Random Geometric Graph Generator

This program generates random geometric graphs of various sizes and saves them in different formats along with visualizations.

## Requirements

- Python 3.6+
- NetworkX
- Matplotlib
- NumPy

## Installation

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install networkx matplotlib numpy
   ```

## Usage

1. Run the script:  
   ```bash
   python random_geo_graphs.py
   ```

2. Choose the output format:
   - `adj`: Adjacency list  
   - `json`: JSON format  
   - `edge`: Edge list format

## Graph Formats

When prompted, choose one of three output formats:
- `adj`: Adjacency list format
  ```
  num_nodes num_edges
  n node_id x_pos y_pos
  e source target weight
  ```
- `json`: JSON format with nodes and edges
  ```json
  {
    "nodes": {"0": {"pos": [x, y]}, ...},
    "edges": [{"source": "0", "target": "1", "weight": w}, ...]
  }
  ```
- `edge`: Edge list format with node positions in comments
  ```
  # Node positions:
  # node_id: x_pos y_pos
  # Edge list (source target weight):
  source target weight
  ```

## Output

The program will create a directory named `random_geometric_graphs_{format}` containing:
- Graph files in the chosen format for each size (10, 100, 200, ..., 1000 nodes)
- PNG visualizations of each graph
- A summary text file with graph statistics

## Example

```bash
$ python random_geo_graphs.py
Choose output format (adj/json/edge): adj
Generating graph with 10 nodes...
Graph with 10 nodes saved successfully
Edges: 12
Average degree: 2.40
```


## Graph Sizes Generated

The program generates graphs with the following numbers of nodes:
- 10 nodes
- 100 nodes
- 200 nodes
- 300 nodes
- 400 nodes
- 500 nodes
- 600 nodes
- 700 nodes
- 800 nodes
- 900 nodes
- 1000 nodes

Each graph is saved with:
1. The graph data in your chosen format
2. A PNG visualization of the graph
3. Statistics in the summary file

## Output Directory Structure

```
random_geometric_graphs_{format}/
├── random_geometric_10.{format}
├── random_geometric_10.png
├── random_geometric_100.{format}
├── random_geometric_100.png
...
├── random_geometric_1000.{format}
├── random_geometric_1000.png
└── graph_summary.txt
Where `{format}` is one of: `adj`, `json`, or `edge`
```

## Graph Visualization

The program also generates PNG images of each graph. These images are saved in the `random_geometric_graphs_{format}` directory.    

## Graph Statistics

The program also generates a summary text file with graph statistics. This file is saved in the `random_geometric_graphs_{format}` directory.
