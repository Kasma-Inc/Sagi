from langchain_community import document_loaders

from Sagi.doc_db.loaders.base_loader import BaseLoader


class PowerPointLoader(BaseLoader):
    """Based on
    https://python.langchain.com/docs/integrations/document_loaders/microsoft_powerpoint/
    """

    def __init__(self):
        self.loader_type = document_loaders.UnstructuredPowerPointLoader
