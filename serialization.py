"""
Wrapper that provides lenient deserialization behavior.

If a serialized object has missing fields present in the example object, the missing fields will be filled with the current value from the example object.
"""

import dataclasses
import re
from flax import serialization
import jax


def lenient_from_bytes(target, data):
  """Like flax's from_bytes but skips unknown fields with a warning."""
  state_dict = serialization.from_bytes(None, data)
  if isinstance(target, list) and isinstance(state_dict, dict):
    # flax serializes lists as dicts with string keys {"0": ..., "1": ...}
    cleaned = {
      str(i): _clean_state_dict(t, state_dict[str(i)], path=f"[{i}]")
      for i, t in enumerate(target)
    }
  else:
    cleaned = _clean_state_dict(target, state_dict)
  result = serialization.from_state_dict(target, cleaned)
  for pattern, count in _warned_fields.items():
    if count > 1:
      print(f"\033[91m  ... '{pattern}' dropped {count} times total\033[0m")
  _warned_fields.clear()
  return result


_warned_fields = {}


def _clean_state_dict(target, state, path=""):
  """Recursively remove keys from state that don't exist in target."""
  if isinstance(state, dict):
    # figure out which keys the target expects
    if hasattr(target, "__dataclass_fields__"):
      expected = set(target.__dataclass_fields__.keys())
    elif isinstance(target, dict):
      expected = set(target.keys())
    else:
      # leaf-like target but dict state — just return state
      return state

    cleaned = {}
    for k, v in state.items():
      if k not in expected:
        # deduplicate warnings by stripping indices from path
        pattern = re.sub(r"\[\d+\]", "[*]", f"{path}.{k}")
        _warned_fields[pattern] = _warned_fields.get(pattern, 0) + 1
        if _warned_fields[pattern] == 1:
          print(f"\033[91mWarning: dropping unknown field '{path}.{k}'\033[0m")
        continue
      # get the sub-target for recursion
      if hasattr(target, "__dataclass_fields__"):
        sub_target = getattr(target, k)
      else:
        sub_target = target[k]
      cleaned[k] = _clean_state_dict(sub_target, v, path=f"{path}.{k}")
    return cleaned

  if isinstance(state, (list, tuple)):
    if isinstance(target, (list, tuple)) and len(target) == len(state):
      return type(state)(
        _clean_state_dict(t, s, path=f"{path}[{i}]")
        for i, (t, s) in enumerate(zip(target, state))
      )
    return state

  # leaf
  return state


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
    if state is None:
      return None
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
