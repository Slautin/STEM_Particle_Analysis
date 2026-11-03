"""Fake microscope for tests. Replaces mcp.ops; no network.

Planned:
    class FakeMicroscope
        synthetic particle image whose blur depends on stage z, C1 and A1
        (known optimum, so searches can be checked)
        records every call, can be told to fail on a chosen call
"""
