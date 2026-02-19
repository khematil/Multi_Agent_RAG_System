from pathlib import Path
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import Distance, VectorParams, PointStruct
import argparse

from config import *
from utils import get_qdrant_client, get_embedding_model


