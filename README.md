# svebm
Unsupervised meta-learning model from [Pang et al](https://openreview.net/forum?id=-pLftu7EpXz), with Lightning and Transformer layers.

## Setup

Requires conda, (for easy packaging), and any C++ compiler, (for TorchInductor). Otherwise, install/uninstall is automated with:
```bash
make install # Create conda environment SV-EBM
make uninstall # Removes conda environment SV-EBM
```

## Developing

**Recommended**: Use the dev environment and tmux for the best experience:

```bash
make dev                    # Start development session
tmux attach-session -t svebm_dev  # Attach to session
```

**JIT Note**: Top-level JIT compilation only happens in `main.py`, so you can use breakpoints freely.

**Commands**:
```bash
make test     # Run tests with logging
make logs     # View test logs  
make format   # Format code
make lint     # Run linting
```

## Ref.

- [Original SV-EBM](https://openreview.net/forum?id=-pLftu7EpXz)