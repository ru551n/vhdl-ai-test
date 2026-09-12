"""Test package for the QSPI flash model.

Present so pytest's rootdir-insertion import mode walks up past this
package to `modules/flash_model/python` (which is not a package) and
inserts *that* on `sys.path` -- which is what lets these tests import both
`flash_model.*` and the bridge module `flash_model_bridge` exactly the way
the simulator's embedded interpreter does, with no path juggling in any
test file.
"""
