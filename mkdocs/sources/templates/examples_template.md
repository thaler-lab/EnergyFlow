# Examples

There are {num_example_files} examples provided for the EnergyFlow package. They currently focus on demonstrating the various architectures included as part of EnergyFlow (see [Architectures](../docs/archs)). For examples involving the computation of EFPs or EMDs, see the [Demos](../demos).

To install the examples to the default directory, `~/.energyflow/examples/`, simply run 

```python
python3 -c "import energyflow; energyflow.utils.get_examples()"
```

See the [`get_examples`](../docs/utils/#get_examples) function for more detailed information. Some examples require [Tensorflow](https://tensorflow.org), [matplotlib](https://matplotlib.org/), or [scikit-learn](https://scikit-learn.org/stable/) to be installed.

{example_files}