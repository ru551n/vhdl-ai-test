from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tsfpga.module import BaseModule

if TYPE_CHECKING:
    from vunit.ui import VUnit


class Module(BaseModule):
    """tsfpga module declaration for the ``flash_model`` VHDL library."""

    def setup_vunit(self, vunit_proj: VUnit, **kwargs: Any) -> None:  # noqa: ANN401, ARG002
        """Run the QSPI master suite at two bus speeds.

        Every timing expectation in ``tb_qspi_master`` is derived from
        ``g_sck_period_ns`` rather than written as a literal, so running the
        whole suite at a second period is a real check that the VC's SCK
        generation, CS framing and dummy-cycle timing scale with the
        configured period instead of happening to work at one of them.
        """
        test_bench = vunit_proj.library(self.library_name).test_bench("tb_qspi_master")

        for sck_period_ns in (20, 33):
            self.add_vunit_config(
                test=test_bench,
                generics={"g_sck_period_ns": sck_period_ns},
            )
