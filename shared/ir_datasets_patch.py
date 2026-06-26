"""Windows-safe UTF-8 reading for ir_datasets TSV streams (MS MARCO)."""

import io


def patch_ir_datasets_tsv_utf8() -> None:
    """Force UTF-8 when reading ir_datasets TSV on Windows (avoids cp1252 errors)."""
    import ir_datasets.formats.tsv as tsv_mod

    if getattr(tsv_mod.FileLineIter, "_utf8_patched", False):
        return

    def _open_utf8_stream(_self, raw_stream):
        return io.TextIOWrapper(raw_stream, encoding="utf-8", errors="replace")

    def _patched_next(self):
        if self.stop is not None and self.start >= self.stop:
            self.ctxt.close()
            raise StopIteration
        if self.stream is None:
            if isinstance(self.dlc, list):
                self.stream = _open_utf8_stream(
                    self, self.ctxt.enter_context(self.dlc[self.stream_idx].stream())
                )
            else:
                self.stream = _open_utf8_stream(
                    self, self.ctxt.enter_context(self.dlc.stream())
                )
        line = ""
        while self.pos < self.start:
            line = self.stream.readline()
            if line != "\n":
                self.pos += 1
        if line == "":
            if isinstance(self.dlc, list):
                self.stream_idx += 1
                if self.stream_idx < len(self.dlc):
                    self.stream = _open_utf8_stream(
                        self,
                        self.ctxt.enter_context(self.dlc[self.stream_idx].stream()),
                    )
                    line = self.stream.readline()
                else:
                    raise StopIteration()
            else:
                raise StopIteration()
        self.start += self.step
        return line

    tsv_mod.FileLineIter.__next__ = _patched_next
    tsv_mod.FileLineIter._utf8_patched = True
