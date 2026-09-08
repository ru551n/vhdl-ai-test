"""Test package for `accel_v2`.

Present so pytest's rootdir-insertion import mode walks up past
`accel_v2/tests` and `accel_v2` (both packages) to `modules/cnn_accel`
(which is not a package) and inserts *that* directory onto `sys.path`,
letting these tests `import cnn_accel_model` / `cnn_accel_constants`
directly, exactly like `modules/cnn_accel/test_cnn_accel_model.py` does.
"""
