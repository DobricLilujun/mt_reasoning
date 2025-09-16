from dotenv import load_dotenv
import os; load_dotenv()
from fast_graphrag import GraphRAG
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CONCURRENT_TASK_LIMIT = os.getenv("CONCURRENT_TASK_LIMIT", 8)


DOMAIN = (
    "Extract grammar knowledge from linguistics text book. Identify each GrammarPoint "
    "(name and concise definition), its Rules (conditions, morphology, syntax), "
    "Terms used, Examples (original, translation, gloss), Phenomena (e.g., ablaut/umlaut), "
    "and Variations/Exceptions. Build explicit relationships among them."
)

EXAMPLE_QUERIES = [
    "List all GrammarPoints and their core Rules.",
    "Find Examples that illustrate final devoicing and suffix assimilation.",
    "What Variations exist for the t-variant before consonants and vowels?",
    "How does Wechselflexion relate to strong vs weak verbs?",
    "Show contrasts between perfect vs preterite in usage frequency."
]

ENTITY_TYPES = [
    "GrammarPoint", "Rule", "Term", "Example", "Phenomenon", "Variation"
]


grag = GraphRAG(
    working_dir="/home/snt/projects_lujun/mt_reasoning/data/graph_rag_dir",
    domain=DOMAIN,
    example_queries="\n".join(EXAMPLE_QUERIES),
    entity_types=ENTITY_TYPES,            
)

with open("/home/snt/projects_lujun/mt_reasoning/data/graph_rag_dir/grammer_4_7_test.txt", "r", encoding="utf-8") as f:
    grag.insert(f.read())

