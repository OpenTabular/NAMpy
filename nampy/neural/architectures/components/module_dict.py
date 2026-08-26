"""Module registration keyed by unmodified external feature names.

PyTorch's :class:`torch.nn.ModuleDict` uses mapping keys as module attribute
names.  Consequently, otherwise legitimate dataframe column names such as
``"children"`` collide with ``nn.Module.children``, while names containing a
dot are rejected because dots delimit paths in ``state_dict``.  NAMpy must not
impose those implementation details on the user's feature schema.
"""

from __future__ import annotations

from collections.abc import ItemsView, KeysView, Mapping, ValuesView

import torch.nn as nn


class RawKeyModuleDict(nn.ModuleDict):
    """A ``ModuleDict`` facade that exposes raw keys and registers safe keys.

    Ordinary keys keep their original registration name, preserving the
    state-dict paths of existing artifacts.  Only keys that PyTorch cannot
    register directly are mapped to a deterministic, injective UTF-8 hex name.
    The reserved internal prefix starts with ``":"``; raw neural feature
    names cannot contain colons because that character owns the interaction
    term grammar.

    Indexing, iteration, ``keys()``, ``items()``, and ``values()`` all expose
    the caller-supplied keys.  The encoding is solely an implementation detail
    of PyTorch module registration and state-dict traversal.
    """

    _ENCODED_PREFIX = ":nampy_raw_key:"

    def __init__(self, modules: Mapping[str, nn.Module] | None = None) -> None:
        # Do not pass ``modules`` to ModuleDict.__init__: it dispatches to our
        # update() before the bidirectional key maps exist.
        super().__init__()
        self._raw_to_internal: dict[str, str] = {}
        self._internal_to_raw: dict[str, str] = {}
        if modules is not None:
            self.update(modules)

    def _registration_key(self, raw_key: str) -> str:
        if not isinstance(raw_key, str):
            raise TypeError(
                "RawKeyModuleDict keys must be strings, "
                f"but received {type(raw_key).__name__}."
            )
        directly_registerable = (
            bool(raw_key)
            and "." not in raw_key
            and not raw_key.startswith(self._ENCODED_PREFIX)
            and not hasattr(self, raw_key)
        )
        if directly_registerable:
            return raw_key
        return f"{self._ENCODED_PREFIX}{raw_key.encode('utf-8').hex()}"

    def __getitem__(self, key: str) -> nn.Module:
        return self._modules[self._raw_to_internal[key]]

    def __setitem__(self, key: str, module: nn.Module) -> None:
        internal_key = self._raw_to_internal.get(key)
        if internal_key is None:
            internal_key = self._registration_key(key)
            # The reserved prefix makes encoded keys disjoint from every key
            # that can be registered verbatim.  Retain an explicit guard so a
            # future grammar change cannot silently alias two raw names.
            other_raw_key = self._internal_to_raw.get(internal_key)
            if other_raw_key is not None and other_raw_key != key:
                raise RuntimeError(
                    "RawKeyModuleDict internal key collision between "
                    f"{other_raw_key!r} and {key!r}."
                )
            self._raw_to_internal[key] = internal_key
            self._internal_to_raw[internal_key] = key
        super().__setitem__(internal_key, module)

    def __delitem__(self, key: str) -> None:
        internal_key = self._raw_to_internal.pop(key)
        del self._internal_to_raw[internal_key]
        super().__delitem__(internal_key)

    def __iter__(self):
        return iter(self._raw_to_internal)

    def __len__(self) -> int:
        return len(self._raw_to_internal)

    def __contains__(self, key: object) -> bool:
        return key in self._raw_to_internal

    def keys(self) -> KeysView[str]:
        return KeysView(self)

    def items(self) -> ItemsView[str, nn.Module]:
        return ItemsView(self)

    def values(self) -> ValuesView[nn.Module]:
        return ValuesView(self)

    def clear(self) -> None:
        super().clear()
        self._raw_to_internal.clear()
        self._internal_to_raw.clear()

