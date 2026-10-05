import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from research_agent.utils import convert_pdf_url_to_markdown


class DocumentConversionTests(unittest.TestCase):
    @patch("research_agent.utils.DocumentConverter")
    @patch("research_agent.utils.DocumentStream")
    @patch("research_agent.utils.requests.get")
    def test_downloads_authenticated_pdf_and_returns_markdown(
        self, get, document_stream, converter_class
    ):
        get.return_value = Mock(content=b"pdf bytes")
        converter = converter_class.return_value
        converter.convert.return_value = SimpleNamespace(
            document=SimpleNamespace(export_to_markdown=lambda: "# Paper")
        )

        markdown = convert_pdf_url_to_markdown(
            "https://content.openalex.org/works/W123.pdf", "test-api-key"
        )

        get.assert_called_once_with(
            "https://content.openalex.org/works/W123.pdf",
            headers={"Authorization": "Bearer test-api-key"},
            timeout=30,
        )
        get.return_value.raise_for_status.assert_called_once_with()
        document_stream.assert_called_once()
        self.assertEqual(document_stream.call_args.kwargs["name"], "document.pdf")
        self.assertEqual(document_stream.call_args.kwargs["stream"].read(), b"pdf bytes")
        converter.convert.assert_called_once_with(document_stream.return_value)
        self.assertEqual(markdown, "# Paper")

    @patch("research_agent.utils.DocumentConverter")
    @patch("research_agent.utils.DocumentStream")
    @patch("research_agent.utils.requests.get")
    def test_propagates_conversion_errors(
        self, get, document_stream, converter_class
    ):
        get.return_value = Mock(content=b"pdf bytes")
        error = RuntimeError("conversion failed")
        converter_class.return_value.convert.side_effect = error

        with self.assertRaisesRegex(RuntimeError, "conversion failed"):
            convert_pdf_url_to_markdown("https://example.org/paper.pdf")


if __name__ == "__main__":
    unittest.main()
