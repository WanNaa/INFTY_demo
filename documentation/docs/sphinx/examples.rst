Examples
========

PILOT demo:

.. code-block:: bash

   cd examples/PILOT
   python main.py --config=exps/memo_scr.json --inftyopt=c_flat --workdir ../../workdirs
   python main.py --config=exps/ease.json --inftyopt=zo_sgd_conserve --workdir ../../workdirs
   python main.py --config=exps/icarl.json --inftyopt=unigrad_fs --workdir ../../workdirs

Minimal examples are provided under ``examples/infty_minimal/``.
