# === Imports ===
import os
import uuid
import logging
from tqdm import tqdm
from pymilvus import connections, db, utility, Collection
from langchain_milvus import Milvus
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

# === Setup Logging ===
logging.basicConfig(
    filename="milvus_debug.log",
    filemode="w",
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

logger.info("🚀 Starting Milvus multi-vector pipeline")

# === Connect to Milvus Server ===
logger.info("🔌 Connecting to Milvus...")
connections.connect(host="localhost", port="19530")
db_name = "multivector_demo"

if db_name in db.list_database():
    logger.info(f"🧹 Dropping old database '{db_name}' and its collections...")
    db.using_database(db_name)
    for name in utility.list_collections():
        Collection(name=name).drop()
        logger.info(f"   Dropped collection: {name}")
    db.drop_database(db_name)

db.create_database(db_name)
db.using_database(db_name)
logger.info(f"✅ Created and switched to database '{db_name}'")

# === Simulate 10 Long Texts ===
texts = [
    """The global climate crisis is accelerating faster than anticipated, with rising sea levels threatening major coastal cities. Immediate international collaboration is required to reduce carbon emissions and invest in renewable energy sources.""",
    """In recent developments in artificial intelligence, multi-modal models are gaining traction. These models can simultaneously process text, images, and audio, improving capabilities in tasks like medical diagnostics and autonomous navigation.""",
    """The history of quantum computing dates back to the 1980s, but only recently have we begun to realize practical applications. Quantum supremacy promises exponential speedups for specific problems, potentially disrupting cryptography and logistics.""",
    """In the field of neuroscience, researchers are developing brain-computer interfaces that allow paralyzed individuals to control robotic limbs. This technology uses neural decoding algorithms to interpret brain signals in real time.""",
    """Modern agriculture is being transformed by AI-powered precision farming. With the help of satellite data and IoT sensors, farmers can optimize irrigation, fertilization, and pest control to improve yield and reduce environmental impact.""",
    """Space exploration is entering a new era with missions to Mars and the Moon becoming more frequent. Reusable rockets and international space stations are reducing costs and making deep space travel a reality.""",
    """The concept of universal basic income (UBI) is gaining popularity as automation threatens jobs across industries. UBI could provide financial security and reduce poverty but raises questions about cost and societal impact.""",
    """In the wake of the COVID-19 pandemic, global supply chains are being redesigned for resilience. Companies are adopting near-shoring, automation, and digital twins to manage disruptions more effectively.""",
    """Cybersecurity threats have evolved with the advent of AI. From deepfake videos to adaptive malware, organizations must deploy AI-based defense mechanisms and continuous monitoring systems to protect critical infrastructure.""",
    """Education technology is revolutionizing learning. Adaptive learning platforms use AI to tailor content to each student's needs, improving retention and engagement in both primary and higher education."""
]

docs = [Document(page_content=txt, metadata={"uuid": str(uuid.uuid4())}) for txt in texts]
logger.info("📄 Created 10 sample documents.")

# === Load Embedding Model ===
logger.info("🔍 Loading embedding model: BAAI/bge-m3")
bge_m3 = SentenceTransformer("BAAI/bge-m3")

# === Generate Multi-vector Embeddings ===
logger.info("🧠 Generating multi-vector embeddings...")
doc_embeddings = []
for i, doc in enumerate(tqdm(docs, desc="Embedding texts")):
    vectors = bge_m3.encode(doc.page_content, normalize_embeddings=True, show_progress_bar=False)
    doc_embeddings.append(vectors)
    logger.info(f"   → Embedded doc {i+1}")

# === Save to Milvus ===
logger.info("🧬 Initializing Milvus vectorstore...")
URI = "http://localhost:19530"
vectorstore = Milvus(
    embedding_function=None,
    connection_args={"uri": URI, "db_name": db_name},
    collection_name="multivector_example",
    index_params={"index_type": "FLAT", "metric_type": "COSINE"},
    drop_old=True,
)

ids = [str(uuid.uuid4()) for _ in docs]
vectorstore.add_embeddings(texts=[doc.page_content for doc in docs], embeddings=doc_embeddings, ids=ids)
logger.info("✅ Saved documents with embeddings to Milvus.")

# === Perform Similarity Search ===
logger.info("🔎 Starting interactive retrieval loop.")
print("\n🔍 Multi-Vector Retrieval (type 'exit' to quit)")
while True:
    user_query = input("Enter your query: ").strip()
    if user_query.lower() in ["exit", "quit"]:
        print("Exiting retrieval loop.")
        logger.info("🛑 Exiting retrieval loop.")
        break

    logger.info(f"🔁 Processing query: {user_query}")
    query_embeds = bge_m3.encode(user_query, normalize_embeddings=True, show_progress_bar=False)

    results = vectorstore.similarity_search_by_vector(
        embedding=query_embeds,
        k=3,
    )

    print("\nTop Similar Documents:")
    for doc in results:
        print(f"- {doc.page_content[:200]}... [{doc.metadata}]")
    print("-" * 80)
    logger.info(f"✅ Returned {len(results)} result(s) for query.")
