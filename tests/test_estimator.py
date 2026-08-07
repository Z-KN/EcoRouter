import unittest
from unittest.mock import MagicMock, patch

from peqrouter.estimator import EstimatorUnavailableError, _load_embedder

_MINILM = "sentence-transformers/all-MiniLM-L6-v2"
_PINNED_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


class LoadEmbedderTests(unittest.TestCase):
    def setUp(self) -> None:
        _load_embedder.cache_clear()
        self.addCleanup(_load_embedder.cache_clear)

    def test_pinned_model_loads_pinned_revision_from_local_cache_only(self) -> None:
        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=MagicMock()) as tok,
            patch("transformers.AutoModel.from_pretrained") as model_cls,
        ):
            model_cls.return_value.eval.return_value = MagicMock()
            _load_embedder(_MINILM)

        tok.assert_called_once_with(_MINILM, revision=_PINNED_REVISION, local_files_only=True)
        model_cls.assert_called_once_with(_MINILM, revision=_PINNED_REVISION, local_files_only=True)

    def test_unpinned_model_name_loads_with_no_revision_but_still_local_only(self) -> None:
        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=MagicMock()) as tok,
            patch("transformers.AutoModel.from_pretrained") as model_cls,
        ):
            model_cls.return_value.eval.return_value = MagicMock()
            _load_embedder("some/other-model")

        tok.assert_called_once_with("some/other-model", revision=None, local_files_only=True)

    def test_missing_local_cache_raises_estimator_unavailable_not_a_raw_oserror(self) -> None:
        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            side_effect=OSError("not cached locally"),
        ):
            with self.assertRaisesRegex(EstimatorUnavailableError, "not cached locally"):
                _load_embedder(_MINILM)


if __name__ == "__main__":
    unittest.main()
