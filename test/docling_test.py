from docling.document_converter import DocumentConverter

source = "https://inovio.com/for-patients-clinical-trials/"  # document per local path or URL
converter = DocumentConverter()
result = converter.convert(source)
print(result.document.export_to_markdown())  # output: "## Docling Technical Report[...]"

