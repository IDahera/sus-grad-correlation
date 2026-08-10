"""Custom exceptions raised by the training utilities."""


class DimensionMismatchError(ValueError):
    """Raised when a model and a dataset have incompatible shapes.

    e.g. the per-sample input shape does not match ``model.input_shape``, or the
    number of classes in the labels exceeds ``model.output_dim``.
    """


class DataNotNormalizedError(ValueError):
    """Raised when input data does not look normalised for the model.

    This is a sanity heuristic that catches the common mistake of feeding raw,
    un-scaled data (e.g. 0-255 pixel values, or un-standardised tabular columns)
    into a model that expects normalised inputs.
    """
