# Glossary

## Base optimizer

A standard PyTorch optimizer, such as `torch.optim.SGD` or `torch.optim.Adam`, wrapped by an INFTY optimizer.

## Closure

A function supplied by the user that re-runs the model forward pass and returns `(logits, loss_list)`.

## Geometry reshaping

An optimizer family that modifies local loss geometry or flatness behavior through perturbation and gradient aggregation.

## Gradient filtering

An optimizer family that modifies gradients from multiple objectives to reduce interference.

## Zeroth-order update

An update that estimates a descent direction without ordinary reverse-mode backpropagation.

## Loss list

A list of scalar tensors returned by a closure. Single-objective training uses one loss. Multi-objective training may use multiple losses.

## Similarity threshold

A threshold used by gradient-filtering methods to decide whether objectives are sufficiently aligned.
