"""
Wrapper that provides lenient deserialization behavior.

If a serialized object has missing fields present in the example object, the missing fields will be filled with the current value from the example object.
"""

import dataclasses
from flax import serialization
import jax


class SerializationWrapper:
  """Wrapper that provides lenient deserialization behavior."""

  def __init__(self, wrapped_obj):
    self._wrapped = wrapped_obj
    self._setup_serialization()

  def __getattr__(self, name):
    return getattr(self._wrapped, name)

  def replace(self, **updates):
    new_wrapped = self._wrapped.replace(**updates)
    return SerializationWrapper(new_wrapped)

  def _setup_serialization(self):
    """Register serialization behavior for this wrapper."""
    original_class = type(self._wrapped)

    # Get data fields
    if hasattr(original_class, "__dataclass_fields__"):
      self._data_fields = [
        f.name
        for f in dataclasses.fields(original_class)
        if f.metadata.get("pytree_node", True)
      ]
    else:
      self._data_fields = [
        attr
        for attr in dir(self._wrapped)
        if not attr.startswith("_") and not callable(getattr(self._wrapped, attr))
      ]

    # Register serialization if not already done
    if SerializationWrapper not in serialization._STATE_DICT_REGISTRY:
      serialization.register_serialization_state(
        SerializationWrapper, self._to_state_dict, self._from_state_dict
      )

      # Register with JAX tree_util
      jax.tree_util.register_pytree_node(
        SerializationWrapper, self._tree_flatten, self._tree_unflatten
      )

  def _to_state_dict(self, x):
    return serialization.to_state_dict(x._wrapped)

  def _from_state_dict(self, x, state):
    """Use example object values for missing fields."""
    state = state.copy()
    updates = {}

    for name in x._data_fields:
      if name in state:
        value = getattr(x._wrapped, name)
        value_state = state.pop(name)
        # Apply lenient deserialization to nested flax structs
        if hasattr(value, "_flax_dataclass") or hasattr(
          type(value), "__dataclass_fields__"
        ):
          wrapped_value = SerializationWrapper(value)
          updates[name] = serialization.from_state_dict(
            wrapped_value, value_state, name=name
          )
        else:
          updates[name] = serialization.from_state_dict(value, value_state, name=name)
      else:
        # Use current value from wrapped object for missing fields
        updates[name] = getattr(x._wrapped, name)

    # Ignore unknown fields in state
    if hasattr(x._wrapped, "replace"):  # flax struct
      return x.replace(**updates)
    elif hasattr(x._wrapped, "_replace"):  # NamedTuple
      return x._replace(**updates)
    else:
      raise ValueError(f"Object {x._wrapped} has no replace method")

  def _tree_flatten(self, x):
    wrapped_children, wrapped_aux = jax.tree_util.tree_flatten(x._wrapped)
    return wrapped_children, wrapped_aux

  def _tree_unflatten(self, aux, children):
    wrapped_obj = jax.tree_util.tree_unflatten(aux, children)
    return SerializationWrapper(wrapped_obj)
